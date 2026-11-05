//! Runs the PP-OCRv6 specialist on whole pages of an eval dataset (`<root>/<bucket>/<page>.png`)
//! and writes `<out>/<bucket>/<page>.md`, one line per detected text line, for `runner --predictions`.
//!   cargo run --release --example ocr -- models eval/data eval/preds/ppocr
use std::path::PathBuf;
use std::time::Instant;

use makaar_core::{ingest, ocr::PpOcr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [models, root, out] = args.as_slice() else {
        return Err("usage: ocr <models-dir> <dataset> <out>".into());
    };
    let models = PathBuf::from(models);
    let mut ocr = PpOcr::load(
        &models.join("pp-ocrv6-small-det"),
        &models.join("pp-ocrv6-small-rec"),
        4,
    )?;
    let (root, out) = (PathBuf::from(root), PathBuf::from(out));
    let mut buckets: Vec<PathBuf> = std::fs::read_dir(&root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    buckets.sort();
    for page in ingest::list_pages(&buckets)? {
        let img = ingest::load_page(&page)?;
        let t = Instant::now();
        let lines = ocr.read(&img)?;
        let conf: f32 = lines.iter().flat_map(|l| &l.probs).sum::<f32>()
            / lines.iter().map(|l| l.probs.len()).sum::<usize>().max(1) as f32;
        let dest = out.join(page.strip_prefix(&root)?).with_extension("md");
        std::fs::create_dir_all(dest.parent().unwrap())?;
        std::fs::write(
            &dest,
            lines
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        )?;
        eprintln!(
            "{}: {} lines, conf {conf:.3}, {:.2}s",
            page.display(),
            lines.len(),
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
