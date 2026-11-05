//! Exports page JSONs (as written by the pipeline example) to Markdown, compile-checked LaTeX
//! (plus PDF) and Word. Figure crops resolve against the first JSON's folder.
//! usage: cargo run --example export -- <out_dir> <name> <page.json>... (tectonic on PATH or $TECTONIC)
use std::path::{Path, PathBuf};
use std::time::Instant;

use makaar_core::document::{Document, Page};
use makaar_core::docx::export_docx;
use makaar_core::export::markdown;
use makaar_core::latex::export_checked;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [out, name, pages @ ..] = args.as_slice() else {
        return Err("usage: export <out_dir> <name> <page.json>...".into());
    };
    let base = pages
        .first()
        .and_then(|p| Path::new(p).parent())
        .unwrap_or(Path::new("."))
        .to_path_buf();
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
    let out = Path::new(out);
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join(format!("{name}.md")), markdown(&doc))?;

    let tectonic = std::env::var_os("TECTONIC").map_or("tectonic".into(), PathBuf::from);
    let t = Instant::now();
    let flagged = export_checked(&mut doc, &tectonic, out, name)?;
    println!("{name}.tex: flagged {flagged:?} in {:.1}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let flagged = export_docx(&mut doc, &base, &out.join(format!("{name}.docx")))?;
    println!("{name}.docx: flagged {flagged:?} in {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}
