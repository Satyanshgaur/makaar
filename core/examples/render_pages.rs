//! Renders every page of the inputs (images or PDFs) as `<out>/<name>-p<N>.png`, exactly as the
//! pipeline sees them (EXIF applied, PDFs at the ingest resolution).
//!   cargo run --release --example render_pages -- datasets/humyn/Handwritten-Physics-Notes-Dataset out/
use std::path::PathBuf;

use makaar_core::ingest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let out = args.pop().ok_or("usage: render_pages <inputs…> <out-dir>")?;
    std::fs::create_dir_all(&out)?;
    for page in ingest::list_pages(&args)? {
        let (file, n) = ingest::pdf_page(&page).unwrap_or((page.clone(), 1));
        let stem = file.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let dest = out.join(format!("{stem}-p{n}.png"));
        if !dest.exists() {
            ingest::load_page(&page)?.save(&dest)?;
        }
    }
    Ok(())
}
