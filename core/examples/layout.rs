//! Runs ingest + layout and prints the document JSON.
//!   cargo run --release --example layout -- models/pp-doclayoutv3/inference.onnx eval/data/printed [--crops out/]
use std::path::PathBuf;
use std::time::Instant;

use makaar_core::document::{Document, Page};
use makaar_core::ingest;
use makaar_core::layout::{to_blocks, LayoutDetector, PpDocLayoutV3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let crops = args.iter().position(|a| a == "--crops").map(|i| {
        let d = PathBuf::from(args.remove(i + 1));
        args.remove(i);
        d
    });
    let (model, inputs) = args
        .split_first()
        .ok_or("usage: layout <model.onnx> <inputs...> [--crops dir]")?;
    let threads = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1));
    let mut det = PpDocLayoutV3::load(model.as_ref(), threads)?;

    let inputs: Vec<PathBuf> = inputs.iter().map(PathBuf::from).collect();
    let mut doc = Document {
        version: 1,
        id: "layout".into(),
        source: None,
        pages: vec![],
    };
    for (i, path) in ingest::list_pages(&inputs)?.iter().enumerate() {
        let img = ingest::load_page(path)?;
        let t = Instant::now();
        let regions = det.detect(&img)?;
        eprintln!(
            "{} {} regions {:.2}s",
            path.display(),
            regions.len(),
            t.elapsed().as_secs_f32()
        );
        let mut blocks = to_blocks(i as u32, &regions);
        if let Some(dir) = &crops {
            std::fs::create_dir_all(dir)?;
            for b in &mut blocks {
                let [x0, y0, x1, y1] = b.bbox.map(|v| v as u32);
                let name = format!("{}-{}.png", b.id, b.label.as_deref().unwrap_or("region"));
                image::imageops::crop_imm(&img, x0, y0, x1 - x0, y1 - y0)
                    .to_image()
                    .save(dir.join(&name))?;
                b.crop = Some(name);
            }
        }
        let (width, height) = img.dimensions();
        doc.pages.push(Page {
            index: i as u32,
            width,
            height,
            image: Some(path.display().to_string()),
            blocks,
        });
    }
    println!("{}", serde_json::to_string_pretty(&doc)?);
    Ok(())
}
