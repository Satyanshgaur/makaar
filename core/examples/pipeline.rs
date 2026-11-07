//! Layout + per-region recognition over an eval dataset (`<root>/<bucket>/<page>.png`). Writes
//! `<out>/<bucket>/<page>.md` (score with `runner --predictions`), the page JSON beside it, and
//! per-page seconds to `<out>/timings.json`. Routes like `makaar convert` (PP-OCR specialist first)
//! unless MAKAAR_NO_SPECIALIST is set.
//!   cargo run --release --example pipeline -- models http://127.0.0.1:8091 eval/data eval/preds/routed
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use makaar_core::engine::{Engine, OpenAiEngine};
use makaar_core::layout::PpDocLayoutV3;
use makaar_core::models;
use makaar_core::ocr::PpOcr;
use makaar_core::{export, ingest, pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [models, url, root, out] = args.as_slice() else {
        return Err("usage: pipeline <models-dir> <server-url> <dataset> <out>".into());
    };
    let threads = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1));
    let models = PathBuf::from(models);
    let mut layout = PpDocLayoutV3::load(&models::LAYOUT.path(&models), threads)?;
    let mut ocr = match std::env::var_os("MAKAAR_NO_SPECIALIST") {
        Some(_) => None,
        None => Some(PpOcr::load(
            models::OCR_DET.path(&models).parent().unwrap(),
            models::OCR_REC.path(&models).parent().unwrap(),
            threads,
        )?),
    };
    let mut engine = OpenAiEngine::new("paddleocr-vl-1.6+layout", url);
    let (root, out) = (PathBuf::from(root), PathBuf::from(out));
    let cfg = pipeline::PipelineConfig::default();

    let mut buckets: Vec<PathBuf> = std::fs::read_dir(&root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    buckets.sort();
    let mut timings: BTreeMap<String, f64> = std::fs::read_to_string(out.join("timings.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    for bucket in buckets {
        let name = bucket.file_name().unwrap().to_owned();
        std::fs::create_dir_all(out.join(&name))?;
        for path in ingest::list_pages(std::slice::from_ref(&bucket))? {
            // strip only the image extension: `paper.pdf_1.jpg` → `paper.pdf_1.md`, as the
            // runner looks it up
            let file = path.file_name().unwrap().to_string_lossy();
            let base = file.rsplit_once('.').map_or(&*file, |(b, _)| b);
            let (md, json) = (
                out.join(&name).join(format!("{base}.md")),
                out.join(&name).join(format!("{base}.json")),
            );
            if md.exists() && json.exists() {
                continue; // done in an earlier run
            }
            let img = ingest::load_page(&path)?;
            let t = Instant::now();
            let page = pipeline::recognize_page_routed(
                &img,
                0,
                &mut layout,
                ocr.as_mut().map(|o| o as &mut dyn Engine),
                &mut engine,
                &cfg,
            )?;
            let secs = t.elapsed().as_secs_f64();
            std::fs::write(&md, export::page_markdown(&page))?;
            std::fs::write(&json, serde_json::to_string_pretty(&page)?)?;
            eprintln!(
                "{}: {} blocks {secs:.1}s",
                path.display(),
                page.blocks.len()
            );
            timings.insert(path.strip_prefix(&root)?.display().to_string(), secs);
        }
    }
    std::fs::write(
        out.join("timings.json"),
        serde_json::to_string_pretty(&timings)?,
    )?;
    Ok(())
}
