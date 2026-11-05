//! Exports page JSONs (as written by the pipeline example) to a compile-checked LaTeX file.
//! usage: cargo run --example latex -- <out_dir> <name> <page.json>... (tectonic on PATH or $TECTONIC)
use std::path::PathBuf;
use std::time::Instant;

use makaar_core::document::{Document, Page};
use makaar_core::latex::export_checked;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [out, name, pages @ ..] = args.as_slice() else {
        return Err("usage: latex <out_dir> <name> <page.json>...".into());
    };
    let pages = pages
        .iter()
        .map(|p| Ok(serde_json::from_str::<Page>(&std::fs::read_to_string(p)?)?))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let mut doc = Document {
        version: 1,
        id: name.clone(),
        source: None,
        pages,
    };
    let tectonic = std::env::var_os("TECTONIC").map_or("tectonic".into(), PathBuf::from);
    let t = Instant::now();
    let bad = export_checked(&mut doc, &tectonic, out.as_ref(), name)?;
    println!(
        "{name}: flagged {bad:?} in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
