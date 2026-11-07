//! The convert job and exports, shared by the CLI and the desktop app. Work is queued in
//! `<out>/.makaar/` (SQLite plus one JSON per finished page), so an interrupted run resumes;
//! progress is reported through a callback.
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::document::{BlockType, Document, Page};
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
        }
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
    assemble_and_export(c, &queue, &state, progress)
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
    let mut engine: Box<dyn Engine> = match &c.server {
        Some(u) => Box::new(OpenAiEngine::new("paddleocr-vl-1.6", u)),
        None => Box::new(LazyServer::new(
            "paddleocr-vl-1.6",
            sidecar::find_binary(&c.models).ok_or(sidecar::SidecarError::NotFound)?,
            SidecarConfig {
                model: models::PADDLE_VL.path(&c.models),
                mmproj: models::PADDLE_VL_MMPROJ.path(&c.models),
                threads: c.threads,
                gpu_layers: c.gpu_layers,
                ctx: 8192,
                log: state.join("llama-server.log"),
            },
        )),
    };
    let cfg = PipelineConfig::default();
    let index = |id: i64| rows.iter().position(|(p, _)| p.id == id).unwrap_or(0);
    let total = rows.len();
    let files = c.out.join(format!("{}_files", c.name));
    while let Some(job) = queue.claim(JOB)? {
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

fn assemble_and_export(
    c: &ConvertOptions,
    queue: &Queue,
    state: &Path,
    progress: &mut dyn FnMut(Progress),
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
    if let Some(note) = write_exports(&mut doc, &c.out, &c.name, &c.formats, c.tectonic.as_deref())?
    {
        progress(Progress::Status { message: note });
    }
    let json = c.json_path();
    progress(Progress::Finished {
        json: json.clone(),
        pages: doc.pages.len(),
        flagged: flagged(&doc),
        failed,
    });
    Ok(json)
}
