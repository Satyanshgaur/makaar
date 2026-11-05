//! Word (.docx) export, written directly as WordprocessingML: headings as Word heading styles,
//! math as native equations (via `omml`), code in a monospace style, tables as Word tables and
//! figure crops embedded. Math that can't be converted is kept as LaTeX source and flagged.
use std::io::Write;
use std::path::Path;

use zip::write::SimpleFileOptions;

use crate::document::{Block, BlockType, Document};
use crate::export::strip_math_delims;
use crate::latex::{math_segments, table_rows, Seg};
use crate::omml::omml;

#[derive(Debug, thiserror::Error)]
pub enum DocxError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("zip: {0}")]
    Zip(#[from] zip::result::ZipError),
}

const NS: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture""#;

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="jpeg" ContentType="image/jpeg"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>"#;

const ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#;

const STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/><w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style><w:style w:type="paragraph" w:customStyle="1" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="20"/></w:rPr></w:style><w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders></w:tblPr></w:style></w:styles>"#;

/// 6.5 inches: the text width of a Letter or A4 page with 1-inch margins.
const MAX_WIDTH_EMU: u64 = 5_943_600;
const EMU_PER_PX: u64 = 9_525;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn text_run(s: &str) -> String {
    format!("<w:r><w:t xml:space=\"preserve\">{}</w:t></w:r>", esc(s))
}

fn para(style: Option<&str>, inner: &str) -> String {
    match style {
        Some(s) => format!("<w:p><w:pPr><w:pStyle w:val=\"{s}\"/></w:pPr>{inner}</w:p>"),
        None => format!("<w:p>{inner}</w:p>"),
    }
}

struct Writer {
    body: String,
    rels: String,
    media: Vec<(String, Vec<u8>)>,
    /// Set when the current block had math kept as LaTeX source.
    lossy: bool,
}

impl Writer {
    /// Equation XML for `latex`, or its source as text (and the block marked lossy).
    fn math(&mut self, latex: &str, display: bool) -> String {
        match omml(latex.trim()) {
            Ok(m) if display => format!("<m:oMathPara>{m}</m:oMathPara>"),
            Ok(m) => m,
            Err(_) => {
                self.lossy = true;
                text_run(&if display {
                    format!("\\[{latex}\\]")
                } else {
                    format!("\\({latex}\\)")
                })
            }
        }
    }

    /// Prose with inline and display math; display math gets its own paragraph.
    fn rich(&mut self, style: Option<&str>, text: &str) {
        let mut inner = String::new();
        for seg in math_segments(text) {
            match seg {
                Seg::Text(t) => inner.push_str(&text_run(t)),
                Seg::Inline(m) => inner.push_str(&self.math(m, false)),
                Seg::Display(m) => {
                    if !inner.trim().is_empty() {
                        self.body
                            .push_str(&para(style, &std::mem::take(&mut inner)));
                    }
                    let eq = self.math(m, true);
                    self.body.push_str(&para(None, &eq));
                }
            }
        }
        if !inner.is_empty() {
            self.body.push_str(&para(style, &inner));
        }
    }

    /// Runs for prose with math kept inline (table cells, line-by-line blocks).
    fn inline(&mut self, text: &str) -> String {
        let mut inner = String::new();
        for seg in math_segments(text) {
            match seg {
                Seg::Text(t) => inner.push_str(&text_run(t)),
                Seg::Inline(m) | Seg::Display(m) => inner.push_str(&self.math(m, false)),
            }
        }
        inner
    }

    fn cell(&mut self, text: &str) -> String {
        let inner = self.inline(text);
        format!(
            "<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>{}</w:tc>",
            para(None, &inner)
        )
    }

    fn table(&mut self, rows: &[Vec<String>]) {
        let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut t = String::from(
            "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr><w:tblGrid>",
        );
        t.push_str(&"<w:gridCol/>".repeat(cols));
        t.push_str("</w:tblGrid>");
        for row in rows {
            t.push_str("<w:tr>");
            for c in 0..cols {
                let cell = self.cell(row.get(c).map_or("", String::as_str));
                t.push_str(&cell);
            }
            t.push_str("</w:tr>");
        }
        t.push_str("</w:tbl>");
        self.body.push_str(&t);
        self.body.push_str("<w:p/>"); // Word needs a paragraph between adjacent tables
    }

    fn code(&mut self, text: &str) {
        let runs: Vec<String> = text.lines().map(text_run).collect();
        self.body
            .push_str(&para(Some("Code"), &runs.join("<w:r><w:br/></w:r>")));
    }

    fn image(&mut self, path: &Path) {
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let Ok((w, h)) = image::image_dimensions(path) else {
            return;
        };
        let ext = match image::guess_format(&bytes) {
            Ok(image::ImageFormat::Png) => "png",
            Ok(image::ImageFormat::Jpeg) => "jpeg",
            _ => return,
        };
        let n = self.media.len() + 1;
        let name = format!("image{n}.{ext}");
        let (mut cx, mut cy) = (w as u64 * EMU_PER_PX, h as u64 * EMU_PER_PX);
        if cx > MAX_WIDTH_EMU {
            cy = cy * MAX_WIDTH_EMU / cx;
            cx = MAX_WIDTH_EMU;
        }
        self.rels.push_str(&format!(
            "<Relationship Id=\"rIdImg{n}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/{name}\"/>"
        ));
        self.body.push_str(&para(None, &format!(
            "<w:r><w:drawing><wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\"><wp:extent cx=\"{cx}\" cy=\"{cy}\"/><wp:docPr id=\"{n}\" name=\"Picture {n}\"/><a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/picture\"><pic:pic><pic:nvPicPr><pic:cNvPr id=\"{n}\" name=\"{name}\"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed=\"rIdImg{n}\"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"
        )));
        self.media.push((name, bytes));
    }

    fn block(&mut self, b: &Block, base: &Path) {
        if b.kind == BlockType::Figure {
            if let Some(c) = &b.crop {
                self.image(&base.join(c));
            }
            return;
        }
        let Some(text) = b.text.as_deref().map(str::trim).filter(|t| !t.is_empty()) else {
            return;
        };
        match (&b.kind, b.label.as_deref()) {
            (BlockType::Heading, Some("doc_title")) => self.rich(Some("Heading1"), text),
            (BlockType::Heading, _) => self.rich(Some("Heading2"), text),
            (BlockType::Formula, Some("inline_formula")) => {
                let m = self.math(strip_math_delims(text), false);
                self.body.push_str(&para(None, &m));
            }
            (BlockType::Formula, _) => {
                let m = self.math(strip_math_delims(text), true);
                self.body.push_str(&para(None, &m));
            }
            (BlockType::Table, _) => {
                let rows = table_rows(text);
                if rows.iter().any(|r| !r.is_empty()) {
                    self.table(&rows);
                } else {
                    self.rich(None, text);
                }
            }
            (BlockType::Text, Some("algorithm")) if !text.contains("\\(") => self.code(
                text.trim_start_matches("```")
                    .trim_end_matches("```")
                    .trim_matches('\n'),
            ),
            (BlockType::Text, Some("algorithm")) => {
                let lines: Vec<String> = text
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| self.inline(l))
                    .collect();
                self.body
                    .push_str(&para(None, &lines.join("<w:r><w:br/></w:r>")));
            }
            _ => {
                for p in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
                    self.rich(None, p);
                }
            }
        }
    }
}

/// Starts the content written from `start` on a new page. A page-break property on its first
/// paragraph, not a break paragraph: that one spills onto a blank page when the previous page
/// is full.
fn break_before(body: &mut String, start: usize) {
    let rest = &body[start..];
    if rest.starts_with("<w:p><w:pPr>") {
        body.insert_str(start + "<w:p><w:pPr>".len(), "<w:pageBreakBefore/>");
    } else if rest.starts_with("<w:p>") {
        body.insert_str(start + "<w:p>".len(), "<w:pPr><w:pageBreakBefore/></w:pPr>");
    } else {
        // empty page or a leading table
        body.insert_str(start, "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>");
    }
}

/// Writes `doc` to `out` as .docx. Figure crops are resolved against `base`. Blocks whose math
/// couldn't be converted (kept as LaTeX source) are flagged; returns them as (page, block id).
pub fn export_docx(
    doc: &mut Document,
    base: &Path,
    out: &Path,
) -> Result<Vec<(usize, String)>, DocxError> {
    let mut w = Writer {
        body: String::new(),
        rels: String::new(),
        media: vec![],
        lossy: false,
    };
    let mut flagged = vec![];
    for (p, page) in doc.pages.iter_mut().enumerate() {
        let start = w.body.len();
        let mut order: Vec<usize> = (0..page.blocks.len()).collect();
        order.sort_by_key(|&i| page.blocks[i].order);
        for i in order {
            w.lossy = false;
            w.block(&page.blocks[i], base);
            if w.lossy {
                page.blocks[i].flagged = Some(true);
                flagged.push((p, page.blocks[i].id.clone()));
            }
        }
        if p > 0 {
            break_before(&mut w.body, start);
        }
    }
    let document = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:document {NS}><w:body>{}<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1134\" w:right=\"1134\" w:bottom=\"1134\" w:left=\"1134\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/></w:sectPr></w:body></w:document>",
        w.body
    );
    let doc_rels = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rIdStyles\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>{}</Relationships>",
        w.rels
    );
    let mut zip = zip::ZipWriter::new(std::fs::File::create(out)?);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, data) in [
        ("[Content_Types].xml", CONTENT_TYPES.as_bytes()),
        ("_rels/.rels", ROOT_RELS.as_bytes()),
        ("word/document.xml", document.as_bytes()),
        ("word/styles.xml", STYLES.as_bytes()),
        ("word/_rels/document.xml.rels", doc_rels.as_bytes()),
    ] {
        zip.start_file(name, opts)?;
        zip.write_all(data)?;
    }
    for (name, data) in &w.media {
        zip.start_file(format!("word/media/{name}"), opts)?;
        zip.write_all(data)?;
    }
    zip.finish()?;
    Ok(flagged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Page;
    use std::io::Read;

    fn block(kind: BlockType, label: &str, order: u32, text: &str) -> Block {
        Block {
            id: format!("b{order}"),
            kind,
            label: Some(label.into()),
            bbox: [0.0; 4],
            order,
            crop: None,
            text: Some(text.into()),
            script: None,
            engine: None,
            confidence: None,
            verified: None,
            flagged: None,
            edited: None,
        }
    }

    #[test]
    fn writes_equations_tables_and_flags_bad_math() {
        let mut d = Document {
            version: 1,
            id: "d".into(),
            source: None,
            pages: vec![Page {
                index: 0,
                width: 1,
                height: 1,
                image: None,
                blocks: vec![
                    block(BlockType::Heading, "paragraph_title", 0, "Big-O & friends"),
                    block(
                        BlockType::Text,
                        "text",
                        1,
                        "If \\(f(n) \\le c\\,g(n)\\) then\n\n$$\\frac{a}{b}$$ done",
                    ),
                    block(BlockType::Formula, "display_formula", 2, "\\[\\frac{1}{\\]"),
                    block(
                        BlockType::Table,
                        "table",
                        3,
                        "<fcel>n<fcel>\\(n^2\\)<nl><fcel>2<fcel>4<nl>",
                    ),
                    block(BlockType::Text, "algorithm", 4, "for(i=0;i<n;i++)\n  x++;"),
                ],
            }],
        };
        let out = std::env::temp_dir().join(format!("makaar-{}.docx", std::process::id()));
        let flagged = export_docx(&mut d, Path::new("."), &out).unwrap();
        assert_eq!(flagged, vec![(0, "b2".to_string())]);
        assert_eq!(d.pages[0].blocks[2].flagged, Some(true));

        let mut z = zip::ZipArchive::new(std::fs::File::open(&out).unwrap()).unwrap();
        let mut xml = String::new();
        z.by_name("word/document.xml")
            .unwrap()
            .read_to_string(&mut xml)
            .unwrap();
        assert!(xml.contains("<w:pStyle w:val=\"Heading2\"/></w:pPr><w:r><w:t xml:space=\"preserve\">Big-O &amp; friends"));
        assert!(xml.contains("<m:oMathPara><m:oMath><m:f>"));
        assert!(xml.contains("\\[\\frac{1}{\\]"), "bad math kept as source");
        assert_eq!(xml.matches("<w:tc>").count(), 4);
        assert!(xml.contains("for(i=0;i&lt;n;i++)</w:t></w:r><w:r><w:br/></w:r>"));
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn embeds_figure_crops() {
        let dir = std::env::temp_dir().join(format!("makaar-fig-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbImage::new(1200, 300)
            .save(dir.join("fig.png"))
            .unwrap();
        let mut fig = block(BlockType::Figure, "image", 0, "");
        fig.crop = Some("fig.png".into());
        let mut d = Document {
            version: 1,
            id: "d".into(),
            source: None,
            pages: vec![Page {
                index: 0,
                width: 1,
                height: 1,
                image: None,
                blocks: vec![fig],
            }],
        };
        let out = dir.join("out.docx");
        export_docx(&mut d, &dir, &out).unwrap();
        let mut z = zip::ZipArchive::new(std::fs::File::open(&out).unwrap()).unwrap();
        assert!(z.by_name("word/media/image1.png").is_ok());
        let mut xml = String::new();
        z.by_name("word/document.xml")
            .unwrap()
            .read_to_string(&mut xml)
            .unwrap();
        // 1200 px is wider than the text, so it is scaled to MAX_WIDTH_EMU keeping the aspect
        assert!(xml.contains(&format!(
            "cx=\"{MAX_WIDTH_EMU}\" cy=\"{}\"",
            MAX_WIDTH_EMU / 4
        )));
        let mut rels = String::new();
        z.by_name("word/_rels/document.xml.rels")
            .unwrap()
            .read_to_string(&mut rels)
            .unwrap();
        assert!(rels.contains("Target=\"media/image1.png\""));
        let _ = std::fs::remove_dir_all(dir);
    }
}
