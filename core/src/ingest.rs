//! Page ingest: list the inputs of a job and load each page upright. Phone photos carry their
//! rotation in EXIF, so it is applied here once and every later stage sees the pixels as shown.
use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("unsupported input: {0}")]
    Unsupported(PathBuf),
}

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"];

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Expands files and folders into page images in a stable order (folders sorted by name).
/// PDFs are not handled yet and are rejected rather than skipped silently.
pub fn list_pages(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, IngestError> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.is_file() && is_image(f))
                .collect();
            files.sort();
            out.extend(files);
        } else if is_image(p) {
            out.push(p.clone());
        } else {
            return Err(IngestError::Unsupported(p.clone()));
        }
    }
    Ok(out)
}

/// Loads a page as RGB with its EXIF orientation applied.
pub fn load_page(path: &Path) -> Result<RgbImage, IngestError> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img.into_rgb8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_folder_sorted_and_rejects_pdf() {
        let dir = std::env::temp_dir().join(format!("makaar-ingest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["b.png", "a.JPG", "notes.txt"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        let pages = list_pages(std::slice::from_ref(&dir)).unwrap();
        let names: Vec<_> = pages
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["a.JPG", "b.png"]);
        assert!(matches!(
            list_pages(&[dir.join("x.pdf")]),
            Err(IngestError::Unsupported(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn loads_rgb() {
        let p = std::env::temp_dir().join(format!("makaar-ingest-{}.png", std::process::id()));
        RgbImage::new(3, 2).save(&p).unwrap();
        let img = load_page(&p).unwrap();
        assert_eq!(img.dimensions(), (3, 2));
        std::fs::remove_file(&p).unwrap();
    }
}
