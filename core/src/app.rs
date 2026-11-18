//! The convert job and exports, shared by the CLI and the desktop app. Work is queued in
//! `<out>/.makaar/` (SQLite plus one JSON per finished page), so an interrupted run resumes;
//! progress is reported through a callback.
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::document::{BlockType, Document, Page};
pub use crate::engine::CancelFlag;
use crate::engine::{Engine, OpenAiEngine};
use crate::layout::PpDocLayoutV3;
use crate::models::{self, REQUIRED};
use crate::ocr::PpOcr;
use crate::pipeline::{self, PipelineConfig};
use crate::queue::{PageJob, Queue};
use crate::sidecar::{self, LazyServer, SidecarConfig};
use crate::{docx, export, ingest, latex};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

const JOB: &str = "convert";

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    pub name: String,
    /// Any of `md`, `tex`, `docx`.
    pub formats: Vec<String>,
    /// A running OpenAI-compatible server to use instead of starting llama-server.
    pub server: Option<String>,
    pub threads: usize,
    pub gpu_layers: Option<u32>,
    pub models: PathBuf,
    pub tectonic: Option<PathBuf>,
    /// Discard earlier job state in `out` and start over.
    pub fresh: bool,
    pub no_specialist: bool,
    /// Don't load the handwriting adapter even if it is installed.
    pub no_adapter: bool,
    /// Handwritten notes: read each block once, with the handwriting adapter (if installed),
    /// instead of twice. About twice as fast on a CPU; printed maths may be restyled.
    pub handwritten: bool,
    /// Set from another thread to stop after the page in hand; finished pages are exported and
    /// the rest stay queued, so converting again resumes.
    pub cancel: Option<CancelFlag>,
}

impl ConvertOptions {
    /// Defaults for `inputs` → `out`: all formats, half the cores, the default models folder.
    pub fn new(inputs: Vec<PathBuf>, out: PathBuf) -> Self {
        Self {
            inputs,
            out,
            name: "notes".into(),
            formats: ["md", "tex", "docx"].map(String::from).to_vec(),
            server: None,
            threads: std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1)),
            gpu_layers: None,
            models: models::default_dir(),
            tectonic: None,
            fresh: false,
            no_specialist: false,
            no_adapter: false,
            handwritten: false,
            cancel: None,
        }
    }

    fn stopped(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed))
    }

    pub fn json_path(&self) -> PathBuf {
        self.out.join(format!("{}.json", self.name))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Progress {
    Status {
        message: String,
    },
    Page {
        done: usize,
        total: usize,
        path: String,
        blocks: usize,
        flagged: usize,
        seconds: f64,
    },
    PageFailed {
        done: usize,
        total: usize,
        path: String,
        error: String,
    },
    Finished {
        json: PathBuf,
        pages: usize,
        flagged: usize,
        failed: Vec<String>,
    },
    /// Stopped early. `json` holds the `done` finished pages (none if no page finished); the
    /// other `total - done` pages stay queued for a resume.
    Stopped {
        json: Option<PathBuf>,
        done: usize,
        total: usize,
    },
}

/// Runs (or resumes) the job and writes the exports; returns the document JSON path.
pub fn convert(c: &ConvertOptions, progress: &mut dyn FnMut(Progress)) -> Result<PathBuf> {
    let status =
        |progress: &mut dyn FnMut(Progress), m: String| progress(Progress::Status { message: m });
    let pages = ingest::list_pages(&c.inputs)?;
    if pages.is_empty() {
        return Err("no page images found in the inputs".into());
    }
    let state = c.out.join(".makaar");
    if c.fresh && state.exists() {
        std::fs::remove_dir_all(&state)?;
    }
    std::fs::create_dir_all(state.join("pages"))?;
    let queue = Queue::open(&state.join("queue.sqlite").to_string_lossy())?;
    let paths: Vec<String> = pages.iter().map(|p| p.display().to_string()).collect();
    let known: Vec<String> = queue.pages(JOB)?.into_iter().map(|(p, _)| p.path).collect();
    if known.is_empty() {
        for p in &paths {
            queue.enqueue(JOB, p)?;
        }
    } else if known != paths {
        return Err(format!(
            "{} holds a job for different pages; start over (--fresh) or pick another output folder",
            c.out.display()
        )
        .into());
    } else {
        let (crashed, failed) = (queue.recover()?, queue.retry_failed(JOB)?);
        if crashed + failed > 0 {
            status(
                progress,
                format!(
                    "resuming: {crashed} interrupted and {failed} failed page(s) will be redone"
                ),
            );
        }
    }
    let rows = queue.pages(JOB)?;
    let pending = rows.iter().filter(|(_, s)| s == "pending").count();
    status(progress, format!("{} page(s), {pending} to do", rows.len()));
    if pending > 0 {
        recognize_pending(c, &queue, &rows, &state, progress)?;
    }
    let stopped = c.stopped() && queue.pages(JOB)?.iter().any(|(_, s)| s == "pending");
    assemble_and_export(c, &queue, &state, progress, stopped)
}

/// Model files the job needs that aren't in `c.models`.
pub fn missing_models(c: &ConvertOptions) -> Vec<&'static str> {
    REQUIRED
        .iter()
        .filter(|m| c.server.is_none() || !m.dest.starts_with("paddleocr-vl"))
        .filter(|m| !m.present(&c.models))
        .map(|m| m.dest)
        .collect()
}

fn recognize_pending(
    c: &ConvertOptions,
    queue: &Queue,
    rows: &[(PageJob, String)],
    state: &Path,
    progress: &mut dyn FnMut(Progress),
) -> Result<()> {
    let missing = missing_models(c);
    if !missing.is_empty() {
        return Err(format!(
            "missing model files in {}: {}; run `makaar setup`",
            c.models.display(),
            missing.join(", ")
        )
        .into());
    }
    let mut layout = PpDocLayoutV3::load(&models::LAYOUT.path(&c.models), c.threads)?;
    let mut ocr = PpOcr::load(
        models::OCR_DET.path(&c.models).parent().unwrap(),
        models::OCR_REC.path(&c.models).parent().unwrap(),
        c.threads,
    )?;
    let adapter = models::handwriting_adapter(&c.models).filter(|_| !c.no_adapter);
    let adapter_only = c.handwritten && adapter.is_some();
    let mut engine: Box<dyn Engine> = match &c.server {
        Some(u) => {
            let mut e = OpenAiEngine::new("paddleocr-vl-1.6", u);
            e.cancel = c.cancel.clone();
            Box::new(e)
        }
        None => Box::new(LazyServer::new(
            "paddleocr-vl-1.6",
            sidecar::find_binary(&c.models).ok_or(sidecar::SidecarError::NotFound)?,
            SidecarConfig {
                model: models::PADDLE_VL.path(&c.models),
                mmproj: models::PADDLE_VL_MMPROJ.path(&c.models),
                lora: adapter,
                threads: c.threads,
                gpu_layers: c.gpu_layers,
                ctx: 8192,
                log: state.join("llama-server.log"),
                cancel: c.cancel.clone(),
            },
        )),
    };
    let cfg = PipelineConfig {
        adapter_only,
        flag_below: if adapter_only {
            pipeline::FLAG_BELOW_ADAPTER
        } else {
            pipeline::FLAG_BELOW
        },
        ..Default::default()
    };
    let index = |id: i64| rows.iter().position(|(p, _)| p.id == id).unwrap_or(0);
    let total = rows.len();
    let files = c.out.join(format!("{}_files", c.name));
    loop {
        if c.stopped() {
            break;
        }
        let Some(job) = queue.claim(JOB)? else { break };
        let i = index(job.id);
        let t = Instant::now();
        let res = ingest::load_page(Path::new(&job.path))
            .map_err(|e| e.to_string())
            .and_then(|img| {
                let mut page = pipeline::recognize_page_routed(
                    &img,
                    i as u32,
                    &mut layout,
                    (!c.no_specialist).then_some(&mut ocr as &mut dyn Engine),
                    engine.as_mut(),
                    &cfg,
                )
                .map_err(|e| e.to_string())?;
                save_figures(&img, &mut page, &files, &c.name).map_err(|e| e.to_string())?;
                // absolute, so the review screen finds the scan from any working directory; a
                // PDF page is saved as rendered, since the review page can't show a PDF page
                let scan = if ingest::pdf_page(Path::new(&job.path)).is_some() {
                    let png = state.join("pages").join(format!("{}.png", job.id));
                    img.save(&png).map_err(|e| e.to_string())?;
                    png
                } else {
                    PathBuf::from(&job.path)
                };
                page.image = std::fs::canonicalize(scan).ok().map(|p| p.display().to_string());
                Ok(page)
            });
        match res {
            Ok(page) => {
                let json = serde_json::to_string_pretty(&page)?;
                std::fs::write(state.join("pages").join(format!("{}.json", job.id)), json)?;
                queue.finish(job.id, true)?;
                progress(Progress::Page {
                    done: i + 1,
                    total,
                    path: job.path.clone(),
                    blocks: page.blocks.len(),
                    flagged: page
                        .blocks
                        .iter()
                        .filter(|b| b.flagged == Some(true))
                        .count(),
                    seconds: t.elapsed().as_secs_f64(),
                });
            }
            Err(_) if c.stopped() => {
                // the read was cut short by Stop, not a bad page: queue it again
                queue.recover()?;
                break;
            }
            Err(error) => {
                queue.finish(job.id, false)?;
                progress(Progress::PageFailed {
                    done: i + 1,
                    total,
                    path: job.path.clone(),
                    error,
                });
            }
        }
    }
    Ok(())
}

/// Saves figure regions as PNGs beside the outputs and points `crop` at them (relative to the
/// output folder, which every exporter resolves against).
fn save_figures(img: &image::RgbImage, page: &mut Page, dir: &Path, name: &str) -> Result<()> {
    for b in page
        .blocks
        .iter_mut()
        .filter(|b| b.kind == BlockType::Figure)
    {
        std::fs::create_dir_all(dir)?;
        let file = format!("{}.png", b.id);
        pipeline::crop(img, b.bbox).save(dir.join(&file))?;
        b.crop = Some(format!("{name}_files/{file}"));
    }
    Ok(())
}

/// Writes `<out>/<name>.{md,tex,pdf,docx}` for the requested formats, then the document JSON
/// last so it carries the flags the exporters add (the review screen reads it). Returns a note
/// when the LaTeX compile check was skipped.
pub fn write_exports(
    doc: &mut Document,
    out: &Path,
    name: &str,
    formats: &[String],
    tectonic: Option<&Path>,
) -> Result<Option<String>> {
    let stem = out.join(name);
    let want = |f: &str| formats.iter().any(|x| x == f);
    let mut note = None;
    if want("md") {
        std::fs::write(stem.with_extension("md"), export::markdown(doc))?;
    }
    if want("tex") {
        let tectonic = tectonic
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("TECTONIC").map(PathBuf::from))
            .unwrap_or_else(|| "tectonic".into());
        match latex::export_checked(doc, &tectonic, out, name) {
            Ok(_) => {}
            Err(latex::LatexError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                std::fs::write(stem.with_extension("tex"), latex::latex(doc))?;
                note = Some(format!(
                    "tectonic not found: wrote {name}.tex without a compile check"
                ));
            }
            Err(e) => return Err(e.into()),
        }
    }
    if want("docx") {
        docx::export_docx(doc, out, &stem.with_extension("docx"))?;
    }
    std::fs::write(
        stem.with_extension("json"),
        serde_json::to_string_pretty(doc)?,
    )?;
    Ok(note)
}

pub fn flagged(doc: &Document) -> usize {
    doc.pages
        .iter()
        .flat_map(|p| &p.blocks)
        .filter(|b| b.flagged == Some(true))
        .count()
}

/// Removes headers and footers that repeat word for word on at least 3 pages and half the
/// document: scanner watermarks ("Scanned with CamScanner") and running heads, not notes.
pub fn drop_repeated_margins(doc: &mut Document) {
    let key = |b: &crate::document::Block| {
        matches!(b.label.as_deref(), Some("header" | "footer"))
            .then(|| b.text.as_deref().unwrap_or("").split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase())
            .filter(|t| !t.is_empty())
    };
    let mut seen = std::collections::HashMap::<String, usize>::new();
    for p in &doc.pages {
        let on_page: std::collections::HashSet<String> = p.blocks.iter().filter_map(key).collect();
        for t in on_page {
            *seen.entry(t).or_default() += 1;
        }
    }
    let pages = doc.pages.len();
    for p in &mut doc.pages {
        p.blocks.retain(|b| key(b).is_none_or(|t| seen[&t] < 3 || 2 * seen[&t] < pages));
    }
}

fn assemble_and_export(
    c: &ConvertOptions,
    queue: &Queue,
    state: &Path,
    progress: &mut dyn FnMut(Progress),
    stopped: bool,
) -> Result<PathBuf> {
    let mut doc = Document {
        version: 1,
        id: c.name.clone(),
        source: None,
        pages: vec![],
    };
    let mut failed = vec![];
    for (job, st) in queue.pages(JOB)? {
        if st == "done" {
            let json =
                std::fs::read_to_string(state.join("pages").join(format!("{}.json", job.id)))?;
            doc.pages.push(serde_json::from_str(&json)?);
        } else {
            failed.push(job.path);
        }
    }
    let total = doc.pages.len() + failed.len();
    if stopped && doc.pages.is_empty() {
        progress(Progress::Stopped { json: None, done: 0, total });
        return Ok(c.json_path());
    }
    drop_repeated_margins(&mut doc);
    if let Some(note) = write_exports(&mut doc, &c.out, &c.name, &c.formats, c.tectonic.as_deref())?
    {
        progress(Progress::Status { message: note });
    }
    let json = c.json_path();
    if stopped {
        progress(Progress::Stopped { json: Some(json.clone()), done: doc.pages.len(), total });
        return Ok(json);
    }
    progress(Progress::Finished {
        json: json.clone(),
        pages: doc.pages.len(),
        flagged: flagged(&doc),
        failed,
    });
    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopping_exports_the_finished_pages_and_keeps_the_rest_queued() {
        let dir = std::env::temp_dir().join(format!("makaar-stop-{}", std::process::id()));
        let state = dir.join(".makaar");
        std::fs::create_dir_all(state.join("pages")).unwrap();
        let queue = Queue::open(":memory:").unwrap();
        for p in ["a.png", "b.png", "c.png"] {
            queue.enqueue(JOB, p).unwrap();
        }
        let first = queue.claim(JOB).unwrap().unwrap();
        let page = serde_json::json!({"index": 0, "width": 10, "height": 10, "blocks": [
            {"id": "a", "type": "text", "label": "text", "bbox": [0, 0, 1, 1], "order": 0, "text": "hello"}]});
        std::fs::write(state.join("pages").join(format!("{}.json", first.id)), page.to_string()).unwrap();
        queue.finish(first.id, true).unwrap();
        let mut c = ConvertOptions::new(vec![], dir.clone());
        c.formats = vec!["md".into()];
        let mut events = vec![];
        assemble_and_export(&c, &queue, &state, &mut |p| events.push(p), true).unwrap();
        assert!(matches!(events.last(), Some(Progress::Stopped { json: Some(_), done: 1, total: 3 })));
        assert!(std::fs::read_to_string(dir.join("notes.md")).unwrap().contains("hello"));
        // the other two pages are still pending, so a resume picks them up
        assert_eq!(queue.pages(JOB).unwrap().iter().filter(|(_, s)| s == "pending").count(), 2);

        // stopped before any page finished: nothing is written
        let empty = Queue::open(":memory:").unwrap();
        empty.enqueue(JOB, "a.png").unwrap();
        let mut events = vec![];
        let c = ConvertOptions::new(vec![], dir.join("none"));
        assemble_and_export(&c, &empty, &state, &mut |p| events.push(p), true).unwrap();
        assert!(matches!(events.last(), Some(Progress::Stopped { json: None, done: 0, total: 1 })));
        assert!(!dir.join("none").exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drops_watermarks_but_keeps_page_numbers_and_notes() {
        let page = |i: usize| {
            serde_json::json!({"index": i, "width": 10, "height": 10, "blocks": [
                {"id": "a", "type": "text", "label": "text", "bbox": [0, 0, 1, 1], "order": 0, "text": "Scanned with CamScanner"},
                {"id": "b", "type": "text", "label": "footer", "bbox": [0, 0, 1, 1], "order": 1, "text": "Scanned  with CamScanner"},
                {"id": "c", "type": "text", "label": "footer", "bbox": [0, 0, 1, 1], "order": 2, "text": format!("{i}")},
            ]})
        };
        let mut doc: Document = serde_json::from_value(serde_json::json!({
            "version": 1, "id": "n", "pages": (0..4).map(page).collect::<Vec<_>>()
        }))
        .unwrap();
        drop_repeated_margins(&mut doc);
        let ids: Vec<&str> = doc.pages[0].blocks.iter().map(|b| b.id.as_str()).collect();
        assert_eq!(ids, ["a", "c"]);
        // two pages are too few to tell a watermark from a heading
        doc.pages.truncate(2);
        for p in &mut doc.pages {
            p.blocks = serde_json::from_value(page(0)["blocks"].clone()).unwrap();
        }
        drop_repeated_margins(&mut doc);
        assert_eq!(doc.pages[1].blocks.len(), 3);
    }
}
