//! makaar desktop: a window over the same core as the CLI. Home screen: first-run model
//! download, pick pages (images or PDFs) and an output folder, convert with live progress, then
//! open the review screen (core::review, served on loopback) in its own window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use makaar_core::app::{self, ConvertOptions, Progress};
use makaar_core::document::Document;
use makaar_core::review::Review;
use makaar_core::{models, runtime, sidecar};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;

/// One job at a time: conversion and setup both load models and use the CPU/GPU fully.
static BUSY: AtomicBool = AtomicBool::new(false);
static REVIEWS: AtomicUsize = AtomicUsize::new(0);

#[derive(Serialize)]
struct Status {
    models_dir: String,
    missing: Vec<&'static str>,
    llama_server: Option<String>,
    download_mb: u64,
}

#[tauri::command]
fn status() -> Status {
    let dir = models::default_dir();
    let missing: Vec<_> = models::REQUIRED.iter().filter(|m| !m.present(&dir)).collect();
    let runtime_mb = runtime::current_asset(runtime::Backend::Cpu).map_or(0, |a| a.size >> 20);
    let server = sidecar::find_binary(&dir);
    Status {
        models_dir: dir.display().to_string(),
        download_mb: missing.iter().map(|m| m.size >> 20).sum::<u64>()
            + if server.is_none() { runtime_mb } else { 0 },
        missing: missing.iter().map(|m| m.dest).collect(),
        llama_server: server.map(|p| p.display().to_string()),
    }
}

#[derive(Clone, Serialize)]
struct SetupProgress {
    item: String,
    done: u64,
    total: u64,
}

struct Busy;
impl Busy {
    fn take() -> Result<Self, String> {
        if BUSY.swap(true, Ordering::SeqCst) {
            return Err("another job is running".into());
        }
        Ok(Busy)
    }
}
impl Drop for Busy {
    fn drop(&mut self) {
        BUSY.store(false, Ordering::SeqCst);
    }
}

/// Downloads missing models and the llama.cpp runtime, emitting `setup` progress events.
#[tauri::command]
async fn setup(app: AppHandle) -> Result<(), String> {
    let busy = Busy::take()?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = busy;
        let dir = models::default_dir();
        for m in models::REQUIRED.iter().filter(|m| !m.present(&dir)) {
            m.download(&dir, |done, total| {
                let _ = app.emit("setup", SetupProgress { item: m.dest.into(), done, total });
            })
            .map_err(|e| e.to_string())?;
        }
        if sidecar::find_binary(&dir).is_none() {
            runtime::install(&dir, runtime::Backend::Cpu, |done, total| {
                let _ = app.emit("setup", SetupProgress { item: format!("llama.cpp {}", runtime::LLAMA_TAG), done, total });
            })
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Pages given on the command line (`makaar-desktop notes.pdf scans/`, or "Open with makaar").
#[tauri::command]
fn launch_inputs() -> Vec<String> {
    std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-') && Path::new(a).exists())
        .filter_map(|a| std::fs::canonicalize(a).ok())
        .map(|p| p.display().to_string())
        .collect()
}

#[tauri::command]
async fn pick_inputs(app: AppHandle, folder: bool) -> Vec<String> {
    let dialog = app.dialog().file();
    let picked = if folder {
        dialog.blocking_pick_folder().map(|f| vec![f])
    } else {
        dialog
            .add_filter("Pages", &["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "pdf"])
            .blocking_pick_files()
    };
    picked
        .unwrap_or_default()
        .into_iter()
        .filter_map(|f| f.into_path().ok())
        .map(|p| p.display().to_string())
        .collect()
}

#[tauri::command]
async fn pick_output(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|f| f.into_path().ok())
        .map(|p| p.display().to_string())
}

/// `~/Documents/makaar/<first input's name>`, or beside the first input.
#[tauri::command]
fn default_output(app: AppHandle, inputs: Vec<String>) -> String {
    let first = inputs.first().map(PathBuf::from).unwrap_or_default();
    let stem = first.file_stem().map_or("notes".into(), |s| s.to_string_lossy().into_owned());
    let base = app.path().document_dir().ok().map(|d| d.join("makaar"));
    base.unwrap_or_else(|| first.parent().unwrap_or(Path::new(".")).to_path_buf())
        .join(stem)
        .display()
        .to_string()
}

/// Runs a conversion on a worker thread; progress arrives as `progress` events. Returns the
/// document JSON path when done.
#[tauri::command]
async fn convert(app: AppHandle, inputs: Vec<String>, out: String, formats: Vec<String>, gpu: bool) -> Result<String, String> {
    let busy = Busy::take()?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = busy;
        let mut c = ConvertOptions::new(inputs.into_iter().map(PathBuf::from).collect(), out.into());
        c.formats = formats;
        if !gpu {
            c.gpu_layers = Some(0);
        }
        let json = app::convert(&c, &mut |p: Progress| {
            let _ = app.emit("progress", p);
        })
        .map_err(|e| e.to_string())?;
        Ok(json.display().to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Opens the review screen for `json` in a new window; saving re-exports the formats beside it.
#[tauri::command]
fn open_review(app: AppHandle, json: String) -> Result<(), String> {
    let path = PathBuf::from(&json);
    let doc: Document = serde_json::from_str(&std::fs::read_to_string(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let out = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let name = path.file_stem().map_or("notes".into(), |s| s.to_string_lossy().into_owned());
    let formats: Vec<String> = ["md", "tex", "docx"].map(String::from).to_vec();
    let save = move |doc: &mut Document| -> Result<String, String> {
        let note = app::write_exports(doc, &out, &name, &formats, None).map_err(|e| e.to_string())?;
        let msg = format!("saved and exported; {} still flagged", app::flagged(doc));
        Ok(note.map_or(msg.clone(), |n| format!("{msg} ({n})")))
    };
    let url = Review { doc, json_path: path }
        .spawn(Box::new(save))
        .map_err(|e| e.to_string())?;
    let n = REVIEWS.fetch_add(1, Ordering::SeqCst);
    WebviewWindowBuilder::new(&app, format!("review-{n}"), WebviewUrl::External(url.parse().map_err(|e| format!("{e}"))?))
        .title("makaar review")
        .inner_size(1300.0, 850.0)
        .build()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn open_path(app: AppHandle, path: String) -> Result<(), String> {
    app.opener().open_path(path, None::<&str>).map_err(|e| e.to_string())
}

fn main() {
    // WebKitGTK's DMA-BUF renderer draws garbage on some Linux GPU/driver setups (NVIDIA, VMs,
    // remote displays); the fallback renderer is fine for this UI.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            status,
            setup,
            launch_inputs,
            pick_inputs,
            pick_output,
            default_output,
            convert,
            open_review,
            open_path
        ])
        .run(tauri::generate_context!())
        .expect("error while running makaar");
}
