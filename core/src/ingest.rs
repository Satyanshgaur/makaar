//! Page ingest: list the inputs of a job and load each page upright. Phone photos carry their
//! rotation in EXIF, so it is applied here once and every later stage sees the pixels as shown.
//! PDFs (scanned or digital) expand to one page per entry, named `file.pdf#page=N`, and are
//! rendered with hayro (pure Rust) at scan-like resolution.
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
    #[error("pdf {0}: {1}")]
    Pdf(PathBuf, String),
}

/// Long side of a rendered PDF page, about a 240 dpi scan of A4.
const PDF_LONG_SIDE: f32 = 2000.0;
const PAGE_MARK: &str = "#page=";

fn is_pdf(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
}

/// `file.pdf#page=3` → (`file.pdf`, 3).
pub fn pdf_page(p: &Path) -> Option<(PathBuf, usize)> {
    let s = p.to_str()?;
    let (file, n) = s.rsplit_once(PAGE_MARK)?;
    Some((PathBuf::from(file), n.parse().ok()?))
}

fn open_pdf(path: &Path) -> Result<hayro::hayro_syntax::Pdf, IngestError> {
    hayro::hayro_syntax::Pdf::new(std::fs::read(path)?)
        .map_err(|e| IngestError::Pdf(path.into(), format!("{e:?}")))
}

fn render_pdf_page(path: &Path, n: usize) -> Result<RgbImage, IngestError> {
    use hayro::{render, PixmapSettings, RenderCache, RenderSettings};
    let pdf = open_pdf(path)?;
    let page = pdf
        .pages()
        .get(n.wrapping_sub(1))
        .ok_or_else(|| IngestError::Pdf(path.into(), format!("no page {n}")))?;
    let (w, h) = page.render_dimensions();
    let scale = (PDF_LONG_SIDE / w.max(h)).clamp(0.5, 8.0);
    let pixmap = render(
        page,
        &RenderCache::new(),
        &Default::default(),
        &RenderSettings::default(),
        &PixmapSettings {
            x_scale: scale,
            y_scale: scale,
            bg_color: hayro::vello_cpu::color::palette::css::WHITE,
        },
    );
    let (pw, ph) = (u32::from(pixmap.width()), u32::from(pixmap.height()));
    // opaque (white background), so premultiplied RGBA is plain RGBA
    let rgb: Vec<u8> = pixmap
        .data_as_u8_slice()
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    RgbImage::from_raw(pw, ph, rgb)
        .ok_or_else(|| IngestError::Pdf(path.into(), "bad pixmap size".into()))
}

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"];

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Expands files and folders into page images in a stable order (folders sorted by name).
/// Other file types are rejected rather than skipped silently.
pub fn list_pages(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, IngestError> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.is_file() && (is_image(f) || is_pdf(f)))
                .collect();
            files.sort();
            out.extend(list_pages(&files)?);
        } else if !p.exists() {
            return Err(IngestError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} not found", p.display()),
            )));
        } else if is_image(p) {
            out.push(p.clone());
        } else if is_pdf(p) {
            let n = open_pdf(p)?.pages().len();
            out.extend((1..=n).map(|i| PathBuf::from(format!("{}{PAGE_MARK}{i}", p.display()))));
        } else {
            return Err(IngestError::Unsupported(p.clone()));
        }
    }
    Ok(out)
}

/// Loads a page as RGB with its EXIF orientation applied.
pub fn load_page(path: &Path) -> Result<RgbImage, IngestError> {
    if let Some((pdf, n)) = pdf_page(path) {
        return render_pdf_page(&pdf, n);
    }
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

    /// A minimal PDF: `pages` A4 pages, each with a black bar at a page-specific height.
    fn tiny_pdf(pages: usize) -> Vec<u8> {
        let mut objs = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!(
                "<< /Type /Pages /Count {pages} /Kids [{}] >>",
                (0..pages)
                    .map(|i| format!("{} 0 R", 3 + 2 * i))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        ];
        for i in 0..pages {
            objs.push(format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {} 0 R >>",
                4 + 2 * i
            ));
            let content = format!("0 0 0 rg 100 {} 300 40 re f", 700 - 100 * i);
            objs.push(format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ));
        }
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = vec![];
        for (i, o) in objs.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).bytes());
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).bytes());
        for o in offsets {
            out.extend(format!("{o:010} 00000 n \n").bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objs.len() + 1
            )
            .bytes(),
        );
        out
    }

    #[test]
    fn pdf_pages_expand_and_render() {
        let dir = std::env::temp_dir().join(format!("makaar-pdf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.pdf"), tiny_pdf(2)).unwrap();
        image::RgbImage::new(4, 4).save(dir.join("b.png")).unwrap();
        let pages = list_pages(std::slice::from_ref(&dir)).unwrap();
        let names: Vec<String> = pages
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a.pdf#page=1", "a.pdf#page=2", "b.png"]);
        let img = load_page(&pages[1]).unwrap();
        assert!(img.height().abs_diff(2000) <= 1, "{}", img.height());
        // page 2's bar is at PDF y=600..640 → image rows (842-640..842-600) × scale
        let scale = 2000.0 / 842.0;
        let y = ((842.0 - 620.0) * scale) as u32;
        assert_eq!(img.get_pixel((250.0 * scale) as u32, y).0, [0, 0, 0]);
        assert_eq!(img.get_pixel(10, 10).0, [255, 255, 255]);
        assert!(matches!(
            load_page(&dir.join("a.pdf#page=9")),
            Err(IngestError::Pdf(..))
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn lists_folder_sorted_and_rejects_others() {
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
            list_pages(&[dir.join("notes.txt")]),
            Err(IngestError::Unsupported(_))
        ));
        assert!(matches!(
            list_pages(&[dir.join("missing.png")]),
            Err(IngestError::Io(_))
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
