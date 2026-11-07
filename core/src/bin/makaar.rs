//! `makaar convert pages/ -o out/` → out/notes.{md,tex,pdf,docx,json}; `makaar models` checks or
//! downloads the model files. Work is queued in `<out>/.makaar/`, so an interrupted run resumes.
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use makaar_core::app::{self, ConvertOptions, Progress, Result};
use makaar_core::document::Document;
use makaar_core::models::{self, REQUIRED};
use makaar_core::runtime;

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
        /// Send every region to the general model (no PP-OCR specialist).
        #[arg(long)]
        no_specialist: bool,
    },
    /// Open the review screen for a converted document (`<out>/<name>.json`) in the browser;
    /// saving re-exports the other formats beside it.
    Review {
        json: PathBuf,
        /// Port on 127.0.0.1 (0 picks a free one).
        #[arg(long, default_value_t = 0)]
        port: u16,
        #[arg(long, default_value = "md,tex,docx")]
        formats: String,
        #[arg(long)]
        tectonic: Option<PathBuf>,
    },
    /// First-run setup: download the models and llama.cpp's llama-server for this platform
    /// (pinned builds, SHA-256 checked).
    Setup {
        #[arg(long)]
        dir: Option<PathBuf>,
        /// llama.cpp build: cpu (works everywhere) or vulkan (most laptop GPUs).
        #[arg(long, default_value = "cpu")]
        backend: String,
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
        Cmd::Review {
            json,
            port,
            formats,
            tectonic,
        } => review(&json, port, &formats, tectonic.as_deref()),
        Cmd::Setup { dir, backend } => setup(&dir.unwrap_or_else(models::default_dir), &backend),
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
            no_specialist,
        } => {
            let mut c = ConvertOptions::new(inputs, out);
            c.name = name;
            c.formats = formats
                .split(',')
                .map(|f| f.trim().to_lowercase())
                .collect();
            c.server = server;
            c.threads = threads.unwrap_or(c.threads);
            c.gpu_layers = gpu_layers;
            c.models = models.unwrap_or(c.models);
            c.tectonic = tectonic;
            c.fresh = fresh;
            c.no_specialist = no_specialist;
            app::convert(&c, &mut print_progress).map(|_| ())
        }
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn review(json: &Path, port: u16, formats: &str, tectonic: Option<&Path>) -> Result<()> {
    let doc: Document = serde_json::from_str(&std::fs::read_to_string(json)?)?;
    let out = json.parent().unwrap_or(Path::new(".")).to_path_buf();
    let name = json
        .file_stem()
        .ok_or("not a file")?
        .to_string_lossy()
        .into_owned();
    let formats: Vec<String> = formats
        .split(',')
        .map(|f| f.trim().to_lowercase())
        .collect();
    let tectonic = tectonic.map(Path::to_path_buf);
    let mut save = |doc: &mut Document| -> std::result::Result<String, String> {
        let note = app::write_exports(doc, &out, &name, &formats, tectonic.as_deref())
            .map_err(|e| e.to_string())?;
        let msg = format!("saved and exported; {} still flagged", app::flagged(doc));
        Ok(note.map_or(msg.clone(), |n| format!("{msg} ({n})")))
    };
    makaar_core::review::Review {
        doc,
        json_path: json.to_path_buf(),
    }
    .serve(port, &mut save)?;
    Ok(())
}

fn setup(dir: &Path, backend: &str) -> Result<()> {
    let backend = match backend {
        "cpu" => runtime::Backend::Cpu,
        "vulkan" => runtime::Backend::Vulkan,
        other => return Err(format!("unknown backend {other}; use cpu or vulkan").into()),
    };
    run_models(dir, true, false)?;
    match runtime::installed_server(dir) {
        Some(p) => println!("llama-server: {} (already installed)", p.display()),
        None => {
            eprintln!("downloading llama.cpp {} ({backend:?})", runtime::LLAMA_TAG);
            let p = runtime::install(dir, backend, |done, total| {
                eprint!("\r  {} / {} MB", done >> 20, total >> 20);
            })?;
            eprintln!();
            println!("llama-server: {}", p.display());
        }
    }
    println!("ready: makaar convert <pages> -o <out>");
    Ok(())
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

fn print_progress(p: Progress) {
    match p {
        Progress::Status { message } => eprintln!("{message}"),
        Progress::Page {
            done,
            total,
            path,
            blocks,
            flagged,
            seconds,
        } => {
            eprintln!("[{done}/{total}] {path}: {blocks} blocks, {flagged} flagged, {seconds:.1}s")
        }
        Progress::PageFailed {
            done,
            total,
            path,
            error,
        } => {
            eprintln!("[{done}/{total}] {path}: FAILED: {error}")
        }
        Progress::Finished {
            json,
            pages,
            flagged,
            failed,
        } => {
            eprintln!(
                "wrote {} ({pages} page(s), {flagged} block(s) flagged for review)",
                json.with_extension("").display()
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
        }
    }
}
