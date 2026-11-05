//! Calibrates the specialist threshold offline: replays stored general-model pages
//! (`<preds>/<bucket>/<page>.json` from the pipeline example), runs PP-OCR on each eligible region,
//! and writes `<out>/<accept>/<bucket>/<page>.md` with the specialist's text wherever it would be
//! accepted. Score each folder with `runner --predictions`.
//!   cargo run --release --example route_sim -- models eval/data eval/preds/paddle-layout eval/preds/route 0.9,0.95,0.98
use std::path::PathBuf;

use makaar_core::document::Page;
use makaar_core::engine::{Engine, Task};
use makaar_core::export::page_markdown;
use makaar_core::ocr::PpOcr;
use makaar_core::{ingest, pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [models, data, preds, out, accepts] = args.as_slice() else {
        return Err("usage: route_sim <models> <dataset> <stored-preds> <out> <t1,t2,…>".into());
    };
    let accepts: Vec<f64> = accepts
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let models = PathBuf::from(models);
    let mut ocr = PpOcr::load(
        &models.join("pp-ocrv6-small-det"),
        &models.join("pp-ocrv6-small-rec"),
        4,
    )?;
    let (data, preds, out) = (
        PathBuf::from(data),
        PathBuf::from(preds),
        PathBuf::from(out),
    );
    let mut buckets: Vec<PathBuf> = std::fs::read_dir(&data)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    buckets.sort();
    for img_path in ingest::list_pages(&buckets)? {
        let rel = img_path.strip_prefix(&data)?.with_extension("json");
        let page: Page = serde_json::from_str(&std::fs::read_to_string(preds.join(&rel))?)?;
        let img = ingest::load_page(&img_path)?;
        let mut spec = vec![None; page.blocks.len()];
        for (i, b) in page.blocks.iter().enumerate() {
            if pipeline::specialist_eligible(b) {
                let rec = ocr.recognize(&pipeline::crop(&img, b.bbox), Task::Text, 0)?;
                spec[i] = Some((pipeline::reflow(&rec.text), rec.confidence()));
            }
        }
        let confs: Vec<String> = spec
            .iter()
            .flatten()
            .map(|(_, c)| format!("{c:.3}"))
            .collect();
        eprintln!(
            "{}: {} eligible [{}]",
            rel.display(),
            confs.len(),
            confs.join(" ")
        );
        for &t in &accepts {
            let mut p = page.clone();
            for (b, s) in p.blocks.iter_mut().zip(&spec) {
                if let Some((text, c)) = s.as_ref().filter(|(text, c)| !text.is_empty() && *c >= t)
                {
                    b.text = Some(text.clone());
                    b.confidence = Some(*c);
                }
            }
            let dest = out.join(format!("{t}")).join(&rel).with_extension("md");
            std::fs::create_dir_all(dest.parent().unwrap())?;
            std::fs::write(dest, page_markdown(&p))?;
        }
    }
    Ok(())
}
