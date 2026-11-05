//! `makaar convert pages/ -o out/` → out/notes.{md,tex,pdf,docx,json}; `makaar models` checks or
//! downloads the model files. Work is queued in `<out>/.makaar/`, so an interrupted run resumes.
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use makaar_core::document::{BlockType, Document, Page};
use makaar_core::engine::OpenAiEngine;
use makaar_core::layout::PpDocLayoutV3;
use makaar_core::models::{self, REQUIRED};
use makaar_core::pipeline::{self, PipelineConfig};
use makaar_core::queue::Queue;
use makaar_core::sidecar::{self, LlamaServer, SidecarConfig};
use makaar_core::{docx, export, ingest, latex};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const JOB: &str = "convert";

#[derive(Parser)]
#[command(version, about = "Offline notes → Markdown, LaTeX and Word")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert page images (files or folders, in name order) into one document.
    Convert {
        inputs: Vec<PathBuf>,
        /// Output folder; also holds the resumable job state in `.makaar/`.
        #[arg(short, long)]
        out: PathBuf,
        /// Base name of the output files.
        #[arg(long, default_value = "notes")]
        name: String,
        /// Comma-separated: md, tex, docx.
        #[arg(long, default_value = "md,tex,docx")]
        formats: String,
        /// Use a running OpenAI-compatible server instead of starting llama-server.
        #[arg(long)]
        server: Option<String>,
        /// CPU threads for layout and llama-server (default: half the cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Model layers on the GPU; 0 = CPU only (default: llama-server decides).
        #[arg(long)]
        gpu_layers: Option<u32>,
        #[arg(long)]
        models: Option<PathBuf>,
        /// Tectonic binary for the LaTeX compile check (default: $TECTONIC or PATH).
        #[arg(long)]
        tectonic: Option<PathBuf>,
        /// Discard earlier job state in the output folder and start over.
        #[arg(long)]
        fresh: bool,
    },
    /// Show model status; download missing files or verify checksums.
    Models {
        #[arg(long)]
        dir: Option<PathBuf>,
        #[arg(long)]
        download: bool,
        #[arg(long)]
        verify: bool,
    },
}

fn main() {
    let res = match Cli::parse().cmd {
        Cmd::Models {
            dir,
            download,
            verify,
        } => run_models(&dir.unwrap_or_else(models::default_dir), download, verify),
        Cmd::Convert {
            inputs,
            out,
            name,
            formats,
            server,
            threads,
            gpu_layers,
            models,
            tectonic,
            fresh,
        } => convert(&Convert {
            inputs,
            out,
            name,
            formats: formats
                .split(',')
                .map(|f| f.trim().to_lowercase())
                .collect(),
            server,
            threads: threads.unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1))
            }),
            gpu_layers,
            models: models.unwrap_or_else(models::default_dir),
            tectonic,
            fresh,
        }),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run_models(dir: &Path, download: bool, verify: bool) -> Result<()> {
    println!("models directory: {}", dir.display());
    for m in REQUIRED {
        let state = if !m.present(dir) {
            if download {
                eprintln!(
                    "downloading {} ({} MB) from {}",
                    m.file,
                    m.size >> 20,
                    m.repo
                );
                m.download(dir, |done, total| {
                    eprint!("\r  {} / {} MB", done >> 20, total >> 20);
                })?;
                eprintln!();
                "downloaded, checksum ok"
            } else {
                "missing (run `makaar models --download`)"
            }
        } else if verify {
            if m.verify(dir)? {
                "checksum ok"
            } else {
                "CHECKSUM MISMATCH (delete it and download again)"
            }
        } else {
            "present"
        };
        println!("  {:<48} {state}", m.dest);
    }
    Ok(())
}

struct Convert {
    inputs: Vec<PathBuf>,
    out: PathBuf,
    name: String,
    formats: Vec<String>,
    server: Option<String>,
    threads: usize,
    gpu_layers: Option<u32>,
    models: PathBuf,
    tectonic: Option<PathBuf>,
    fresh: bool,
}

fn convert(c: &Convert) -> Result<()> {
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
            "{} holds a job for different pages; use --fresh to start over or another --out",
            c.out.display()
        )
        .into());
    } else {
        let (crashed, failed) = (queue.recover()?, queue.retry_failed(JOB)?);
        if crashed + failed > 0 {
            eprintln!("resuming: {crashed} interrupted and {failed} failed page(s) will be redone");
        }
    }
    let rows = queue.pages(JOB)?;
    let pending = rows.iter().filter(|(_, s)| s == "pending").count();
    eprintln!("{} page(s), {pending} to do", rows.len());

    if pending > 0 {
        recognize_pending(c, &queue, &rows, &state)?;
    }
    assemble_and_export(c, &queue, &state)
}

fn recognize_pending(
    c: &Convert,
    queue: &Queue,
    rows: &[(makaar_core::queue::PageJob, String)],
    state: &Path,
) -> Result<()> {
    let missing: Vec<&str> = REQUIRED
        .iter()
        .filter(|m| (c.server.is_none() || m.dest.ends_with(".onnx")) && !m.present(&c.models))
        .map(|m| m.dest)
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "missing model files in {}: {}; run `makaar models --download`",
            c.models.display(),
            missing.join(", ")
        )
        .into());
    }
    let mut layout = PpDocLayoutV3::load(&models::LAYOUT.path(&c.models), c.threads)?;
    let _server; // kept alive until recognition ends
    let url = match &c.server {
        Some(u) => u.clone(),
        None => {
            let bin = sidecar::find_binary().ok_or(sidecar::SidecarError::NotFound)?;
            eprintln!("starting {} …", bin.display());
            let t = Instant::now();
            let s = LlamaServer::start(
                &bin,
                &SidecarConfig {
                    model: models::PADDLE_VL.path(&c.models),
                    mmproj: models::PADDLE_VL_MMPROJ.path(&c.models),
                    threads: c.threads,
                    gpu_layers: c.gpu_layers,
                    ctx: 8192,
                    log: state.join("llama-server.log"),
                },
            )?;
            eprintln!("model server ready in {:.0}s", t.elapsed().as_secs_f64());
            let url = s.url.clone();
            _server = s;
            url
        }
    };
    let mut engine = OpenAiEngine::new("paddleocr-vl-1.6", &url);
    let cfg = PipelineConfig::default();
    let index = |id: i64| rows.iter().position(|(p, _)| p.id == id).unwrap_or(0) as u32;
    let total = rows.len();
    let files = c.out.join(format!("{}_files", c.name));
    while let Some(job) = queue.claim(JOB)? {
        let i = index(job.id);
        let t = Instant::now();
        let res = ingest::load_page(Path::new(&job.path))
            .map_err(|e| e.to_string())
            .and_then(|img| {
                let mut page = pipeline::recognize_page(&img, i, &mut layout, &mut engine, &cfg)
                    .map_err(|e| e.to_string())?;
                save_figures(&img, &mut page, &files, &c.name).map_err(|e| e.to_string())?;
                Ok(page)
            });
        match res {
            Ok(page) => {
                let json = serde_json::to_string_pretty(&page)?;
                std::fs::write(state.join("pages").join(format!("{}.json", job.id)), json)?;
                queue.finish(job.id, true)?;
                let flagged = page
                    .blocks
                    .iter()
                    .filter(|b| b.flagged == Some(true))
                    .count();
                eprintln!(
                    "[{}/{total}] {}: {} blocks, {flagged} flagged, {:.1}s",
                    i + 1,
                    job.path,
                    page.blocks.len(),
                    t.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                queue.finish(job.id, false)?;
                eprintln!("[{}/{total}] {}: FAILED: {e}", i + 1, job.path);
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

fn assemble_and_export(c: &Convert, queue: &Queue, state: &Path) -> Result<()> {
    let rows = queue.pages(JOB)?;
    let mut doc = Document {
        version: 1,
        id: c.name.clone(),
        source: None,
        pages: vec![],
    };
    let mut failed = vec![];
    for (job, st) in &rows {
        if st == "done" {
            let json =
                std::fs::read_to_string(state.join("pages").join(format!("{}.json", job.id)))?;
            doc.pages.push(serde_json::from_str(&json)?);
        } else {
            failed.push(job.path.clone());
        }
    }
    let stem = c.out.join(&c.name);
    let want = |f: &str| c.formats.iter().any(|x| x == f);
    if want("md") {
        std::fs::write(stem.with_extension("md"), export::markdown(&doc))?;
    }
    if want("tex") {
        let tectonic = c
            .tectonic
            .clone()
            .or_else(|| std::env::var_os("TECTONIC").map(PathBuf::from))
            .unwrap_or_else(|| "tectonic".into());
        match latex::export_checked(&mut doc, &tectonic, &c.out, &c.name) {
            Ok(_) => {}
            Err(latex::LatexError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                std::fs::write(stem.with_extension("tex"), latex::latex(&doc))?;
                eprintln!(
                    "tectonic not found: wrote {}.tex without a compile check",
                    c.name
                );
            }
            Err(e) => return Err(e.into()),
        }
    }
    if want("docx") {
        docx::export_docx(&mut doc, &c.out, &stem.with_extension("docx"))?;
    }
    // last, so it carries the flags the exporters added; the review screen reads this
    std::fs::write(
        stem.with_extension("json"),
        serde_json::to_string_pretty(&doc)?,
    )?;

    let flagged = doc
        .pages
        .iter()
        .flat_map(|p| &p.blocks)
        .filter(|b| b.flagged == Some(true))
        .count();
    eprintln!(
        "wrote {} ({} page(s), {flagged} block(s) flagged for review)",
        stem.display(),
        doc.pages.len()
    );
    if !failed.is_empty() {
        eprintln!(
            "{} page(s) failed and were left out; rerun the same command to retry:",
            failed.len()
        );
        for f in failed {
            eprintln!("  {f}");
        }
    }
    Ok(())
}
