//! Layout + per-region recognition over an eval dataset (`<root>/<bucket>/<page>.png`). Writes
//! `<out>/<bucket>/<page>.md` (score with `runner --predictions`), the page JSON beside it, and
//! per-page seconds to `<out>/timings.json`.
//!   cargo run --release --example pipeline -- models/pp-doclayoutv3/inference.onnx http://127.0.0.1:8091 eval/data eval/preds/paddle-layout
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use makaar_core::engine::OpenAiEngine;
use makaar_core::layout::PpDocLayoutV3;
use makaar_core::{export, ingest, pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, url, root, out] = args.as_slice() else {
        return Err("usage: pipeline <layout.onnx> <server-url> <dataset> <out>".into());
    };
    let threads = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1));
    let mut layout = PpDocLayoutV3::load(model.as_ref(), threads)?;
    let mut engine = OpenAiEngine::new("paddleocr-vl-1.6+layout", url);
    let (root, out) = (PathBuf::from(root), PathBuf::from(out));
    let cfg = pipeline::PipelineConfig::default();

    let mut buckets: Vec<PathBuf> = std::fs::read_dir(&root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    buckets.sort();
    let mut timings = BTreeMap::new();
    for bucket in buckets {
        let name = bucket.file_name().unwrap().to_owned();
        std::fs::create_dir_all(out.join(&name))?;
        for path in ingest::list_pages(std::slice::from_ref(&bucket))? {
            let img = ingest::load_page(&path)?;
            let t = Instant::now();
            let page = pipeline::recognize_page(&img, 0, &mut layout, &mut engine, &cfg)?;
            let secs = t.elapsed().as_secs_f64();
            let stem = out.join(&name).join(path.file_stem().unwrap());
            std::fs::write(stem.with_extension("md"), export::page_markdown(&page))?;
            std::fs::write(
                stem.with_extension("json"),
                serde_json::to_string_pretty(&page)?,
            )?;
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
