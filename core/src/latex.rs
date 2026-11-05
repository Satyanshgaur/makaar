//! LaTeX export with a compile check. Engines return prose with inline `\(…\)` math, display
//! formulas, code, and tables as HTML or OTSL. Prose is escaped, math passes through, and any
//! block that stops Tectonic is replaced by its escaped source and flagged, so the exported
//! file always compiles.
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::document::{Block, BlockType, Document};
use crate::export::strip_math_delims;

const PREAMBLE: &str = "\\documentclass{article}\n\
\\usepackage{amsmath,amssymb,graphicx}\n\
\\usepackage[margin=2cm]{geometry}\n\
\\setlength{\\parindent}{0pt}\n\
\\setlength{\\parskip}{0.6em}\n\
\\begin{document}\n";
const END: &str = "\\end{document}\n";

/// Symbols common in notes that Latin Modern has no text glyph for.
const SYMBOLS: [(char, &str); 22] = [
    ('→', "\\(\\rightarrow\\)"),
    ('←', "\\(\\leftarrow\\)"),
    ('⇒', "\\(\\Rightarrow\\)"),
    ('⇔', "\\(\\Leftrightarrow\\)"),
    ('↔', "\\(\\leftrightarrow\\)"),
    ('≤', "\\(\\le\\)"),
    ('≥', "\\(\\ge\\)"),
    ('≠', "\\(\\ne\\)"),
    ('≈', "\\(\\approx\\)"),
    ('×', "\\(\\times\\)"),
    ('÷', "\\(\\div\\)"),
    ('±', "\\(\\pm\\)"),
    ('∴', "\\(\\therefore\\)"),
    ('∵', "\\(\\because\\)"),
    ('∞', "\\(\\infty\\)"),
    ('∈', "\\(\\in\\)"),
    ('•', "\\(\\bullet\\)"),
    ('·', "\\(\\cdot\\)"),
    ('°', "\\(^\\circ\\)"),
    ('θ', "\\(\\theta\\)"),
    ('Θ', "\\(\\Theta\\)"),
    ('Ω', "\\(\\Omega\\)"),
];

/// Escapes prose for LaTeX, including the symbols above.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' | '}' | '$' | '&' | '%' | '#' | '_' => {
                out.push('\\');
                out.push(c);
            }
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            _ => match SYMBOLS.iter().find(|(u, _)| *u == c) {
                Some((_, tex)) => out.push_str(tex),
                None => out.push(c),
            },
        }
    }
    out
}

/// A piece of engine text: prose, inline math, or display math (bodies without delimiters).
#[derive(Debug, PartialEq)]
pub enum Seg<'a> {
    Text(&'a str),
    Inline(&'a str),
    Display(&'a str),
}

/// Splits `s` at `\(…\)`, `\[…\]`, `$$…$$` and `$…$`; an unclosed opener stays prose.
pub fn math_segments(s: &str) -> Vec<Seg<'_>> {
    let mut out = vec![];
    let mut rest = s;
    while let Some((start, open, close)) = next_math(rest) {
        let body_from = start + open.len();
        let Some(end) = rest[body_from..].find(close) else {
            break;
        };
        if start > 0 {
            out.push(Seg::Text(&rest[..start]));
        }
        let body = &rest[body_from..body_from + end];
        out.push(if open == "$$" || open == "\\[" {
            Seg::Display(body)
        } else {
            Seg::Inline(body)
        });
        rest = &rest[body_from + end + close.len()..];
    }
    if !rest.is_empty() {
        out.push(Seg::Text(rest));
    }
    out
}

/// Escapes the prose of `s` but keeps `\(…\)`, `\[…\]`, `$$…$$` and `$…$` math as is.
pub fn text_with_math(s: &str) -> String {
    math_segments(s)
        .into_iter()
        .map(|seg| match seg {
            Seg::Text(t) => escape(t),
            Seg::Inline(m) => format!("\\({m}\\)"),
            Seg::Display(m) => format!("\\[{m}\\]"),
        })
        .collect()
}

/// True if `s` opens `\(`, `\[` or `$$` math it never closes: the engine likely lost text, and
/// the source is shown escaped. A lone `$` is ignored since it is usually currency.
pub fn has_unclosed_math(s: &str) -> bool {
    let mut rest = s;
    while let Some((start, open, close)) = next_math(rest) {
        let from = start + open.len();
        match rest[from..].find(close) {
            Some(end) => rest = &rest[from + end + close.len()..],
            None => return open != "$",
        }
    }
    false
}

fn next_math(s: &str) -> Option<(usize, &'static str, &'static str)> {
    [("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$")]
        .into_iter()
        .filter_map(|(o, c)| s.find(o).map(|i| (i, o, c)))
        // earliest opener wins; at the same index the longer one (`$$` over `$`)
        .min_by_key(|(i, o, _)| (*i, usize::MAX - o.len()))
}

/// Table cells from HTML (`<tr><td>`) or OTSL (`<fcel>`, `<ecel>`, `<nl>`, …) output.
pub fn table_rows(s: &str) -> Vec<Vec<String>> {
    if s.contains("<tr") {
        return s
            .split("<tr")
            .skip(1)
            .map(|row| {
                row.split("<t")
                    .skip(1)
                    .filter(|c| c.starts_with('d') || c.starts_with('h'))
                    .map(|c| {
                        let inner = c.split_once('>').map_or("", |(_, v)| v);
                        inner.split("</t").next().unwrap_or("").trim().to_string()
                    })
                    .collect()
            })
            .collect();
    }
    let mut rows = vec![];
    for line in s.split("<nl>") {
        let mut row = vec![];
        for (i, cell) in line.split('<').enumerate() {
            if i == 0 {
                continue;
            }
            let (tag, text) = cell.split_once('>').unwrap_or((cell, ""));
            match tag {
                "fcel" | "ched" | "rhed" | "srow" => row.push(text.trim().to_string()),
                "ecel" | "lcel" | "ucel" | "xcel" => row.push(String::new()),
                _ => {}
            }
        }
        if !row.is_empty() {
            rows.push(row);
        }
    }
    rows
}

fn tabular(rows: &[Vec<String>]) -> Option<String> {
    let cols = rows.iter().map(Vec::len).max().filter(|&n| n > 0)?;
    let mut out = format!(
        "\\begin{{center}}\n\\begin{{tabular}}{{|{}}}\n\\hline\n",
        "l|".repeat(cols)
    );
    for row in rows {
        let mut cells: Vec<String> = row.iter().map(|c| text_with_math(c)).collect();
        cells.resize(cols, String::new());
        out.push_str(&cells.join(" & "));
        out.push_str(" \\\\ \\hline\n");
    }
    out.push_str("\\end{tabular}\n\\end{center}");
    Some(out)
}

fn verbatim(text: &str) -> String {
    // `\end{verbatim}` inside the text would end the environment early
    format!(
        "\\begin{{verbatim}}\n{}\n\\end{{verbatim}}",
        text.replace("\\end{verbatim}", "\\end {verbatim}")
    )
}

/// A block's LaTeX as units that can fail independently: one per paragraph for prose (so a
/// whole-page block isn't lost to one bad formula), otherwise one per block. Each unit keeps
/// its source for the verbatim fallback.
fn block_units(b: &Block) -> Vec<(&str, String)> {
    if b.kind == BlockType::Figure {
        return b.crop.iter().map(|c| {
            ("", format!(
                "\\begin{{center}}\n\\includegraphics[width=0.8\\linewidth,height=0.4\\textheight,keepaspectratio]{{{c}}}\n\\end{{center}}"
            ))
        }).collect();
    }
    let Some(text) = b.text.as_deref().map(str::trim).filter(|t| !t.is_empty()) else {
        return vec![];
    };
    let one = |tex: String| vec![(text, tex)];
    match (&b.kind, b.label.as_deref()) {
        (BlockType::Heading, Some("doc_title")) => {
            one(format!("\\section*{{{}}}", text_with_math(text)))
        }
        (BlockType::Heading, _) => one(format!("\\subsection*{{{}}}", text_with_math(text))),
        (BlockType::Formula, Some("inline_formula")) => {
            one(format!("\\({}\\)", strip_math_delims(text)))
        }
        (BlockType::Formula, _) => one(format!("\\[\n{}\n\\]", strip_math_delims(text))),
        (BlockType::Table, _) => {
            one(tabular(&table_rows(text)).unwrap_or_else(|| text_with_math(text)))
        }
        (BlockType::Text, Some("algorithm")) if !text.contains("\\(") => one(verbatim(
            text.trim_start_matches("```")
                .trim_end_matches("```")
                .trim_matches('\n'),
        )),
        (BlockType::Text, Some("algorithm")) => one(text
            .lines()
            .filter(|l| !l.trim().is_empty()) // `\\` on an empty line is an error
            .map(text_with_math)
            .collect::<Vec<_>>()
            .join("\\\\\n")),
        _ => text
            .split("\n\n")
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| (p, text_with_math(p)))
            .collect(),
    }
}

/// A unit's position: (page, block index in `page.blocks`, unit). Block ids aren't used since
/// pages exported separately can reuse them.
type UnitKey = (usize, usize, usize);

/// Blocks with their position in `page.blocks`, in reading order.
fn ordered(blocks: &[Block]) -> Vec<(usize, &Block)> {
    let mut v: Vec<(usize, &Block)> = blocks.iter().enumerate().collect();
    v.sort_by_key(|(_, b)| b.order);
    v
}

/// LaTeX for `doc`, with the units in `bad` replaced by their verbatim source.
fn render(doc: &Document, bad: &[UnitKey]) -> String {
    let mut out = String::from(PREAMBLE);
    for (p, page) in doc.pages.iter().enumerate() {
        if p > 0 {
            out.push_str("\\clearpage\n\n");
        }
        for (i, b) in ordered(&page.blocks) {
            for (k, (src, tex)) in block_units(b).into_iter().enumerate() {
                if bad.contains(&(p, i, k)) {
                    out.push_str(&verbatim(src));
                } else {
                    out.push_str(&tex);
                }
                out.push_str("\n\n");
            }
        }
    }
    out.push_str(END);
    out
}

/// LaTeX for the whole document, without a compile check.
pub fn latex(doc: &Document) -> String {
    render(doc, &[])
}

#[derive(Debug, thiserror::Error)]
pub enum LatexError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("still does not compile after replacing failing blocks: {0}")]
    Compile(String),
}

/// Runs Tectonic on `tex` in `dir`; `Err` holds the tail of its log.
pub fn compile(tectonic: &Path, dir: &Path, name: &str, tex: &str) -> Result<PathBuf, LatexError> {
    std::fs::create_dir_all(dir)?;
    let file = dir.join(format!("{name}.tex"));
    std::fs::write(&file, tex)?;
    let out = Command::new(tectonic)
        .args(["-X", "compile", "--untrusted"])
        .arg(&file)
        .current_dir(dir)
        .output()?;
    if out.status.success() {
        return Ok(dir.join(format!("{name}.pdf")));
    }
    let log = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
    let tail: Vec<&str> = log.lines().rev().take(12).collect();
    Err(LatexError::Compile(
        tail.into_iter().rev().collect::<Vec<_>>().join("\n"),
    ))
}

/// Writes `<dir>/<name>.tex` (and the PDF) so that it compiles. If the whole document fails,
/// each unit with a command is compiled alone and failures fall back to verbatim source. Those
/// blocks, and prose paragraphs with unclosed math, are flagged in `doc`. Returns the flagged
/// blocks as (page position, block id).
pub fn export_checked(
    doc: &mut Document,
    tectonic: &Path,
    dir: &Path,
    name: &str,
) -> Result<Vec<(usize, String)>, LatexError> {
    let mut flag: Vec<(usize, usize)> = if compile(tectonic, dir, name, &latex(doc)).is_ok() {
        vec![]
    } else {
        isolate_failures(doc, tectonic, dir, name)?
            .into_iter()
            .map(|(p, i, _)| (p, i))
            .collect()
    };
    for (p, page) in doc.pages.iter().enumerate() {
        for (i, b) in page.blocks.iter().enumerate() {
            let prose = matches!(
                b.kind,
                BlockType::Text | BlockType::Heading | BlockType::Caption
            );
            let unclosed = b
                .text
                .as_deref()
                .is_some_and(|t| t.split("\n\n").any(has_unclosed_math));
            if prose && unclosed {
                flag.push((p, i));
            }
        }
    }
    flag.sort();
    flag.dedup();
    Ok(flag
        .into_iter()
        .map(|(p, i)| {
            let b = &mut doc.pages[p].blocks[i];
            b.flagged = Some(true);
            (p, b.id.clone())
        })
        .collect())
}

/// Compiles each unit with a command alone, then writes the document with the failing units
/// as verbatim source.
fn isolate_failures(
    doc: &Document,
    tectonic: &Path,
    dir: &Path,
    name: &str,
) -> Result<Vec<UnitKey>, LatexError> {
    let probe = dir.join(".probe");
    let mut bad = vec![];
    for (p, page) in doc.pages.iter().enumerate() {
        for (i, b) in page.blocks.iter().enumerate() {
            for (k, (_, tex)) in block_units(b).into_iter().enumerate() {
                // escaped prose without commands can't fail on its own
                if tex.contains('\\')
                    && compile(tectonic, &probe, "unit", &format!("{PREAMBLE}{tex}\n{END}"))
                        .is_err()
                {
                    bad.push((p, i, k));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&probe);
    compile(tectonic, dir, name, &render(doc, &bad))?;
    Ok(bad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Page;

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

    fn doc(blocks: Vec<Block>) -> Document {
        Document {
            version: 1,
            id: "d".into(),
            source: None,
            pages: vec![Page {
                index: 0,
                width: 1,
                height: 1,
                image: None,
                blocks,
            }],
        }
    }

    #[test]
    fn escapes_prose_keeps_math() {
        assert_eq!(
            escape("50% of a_b & {c} → d"),
            "50\\% of a\\_b \\& \\{c\\} \\(\\rightarrow\\) d"
        );
        assert_eq!(
            text_with_math("cost \\(c_1\\) is $$x^2$$ and $y$ 10%"),
            "cost \\(c_1\\) is \\[x^2\\] and \\(y\\) 10\\%"
        );
        assert_eq!(text_with_math("costs $5 only"), "costs \\$5 only");
        assert!(has_unclosed_math("a \\(x^2 and b"));
        assert!(!has_unclosed_math("a \\(x\\) costs $5"));
    }

    #[test]
    fn tables_from_html_and_otsl() {
        let html = "<table><tr><th>a</th><td>b_1</td></tr><tr><td>c</td><td></td></tr></table>";
        assert_eq!(table_rows(html), vec![vec!["a", "b_1"], vec!["c", ""]]);
        let otsl = "<fcel>a<fcel>b<nl><fcel>c<ecel><nl>";
        assert_eq!(table_rows(otsl), vec![vec!["a", "b"], vec!["c", ""]]);
        let t = tabular(&table_rows(html)).unwrap();
        assert!(t.contains("{|l|l|}") && t.contains("a & b\\_1 \\\\"));
    }

    #[test]
    fn blocks_in_order() {
        let d = doc(vec![
            block(BlockType::Formula, "display_formula", 1, "$$E=mc^2$$"),
            block(BlockType::Heading, "paragraph_title", 0, "Energy & mass"),
            block(BlockType::Text, "algorithm", 2, "for(i=0;i<n;i++) {}"),
        ]);
        let tex = latex(&d);
        let body = &tex[PREAMBLE.len()..tex.len() - END.len()];
        assert_eq!(
            body,
            "\\subsection*{Energy \\& mass}\n\n\\[\nE=mc^2\n\\]\n\n\\begin{verbatim}\nfor(i=0;i<n;i++) {}\n\\end{verbatim}\n\n"
        );
    }

    fn tectonic() -> Option<PathBuf> {
        let p = std::env::var_os("TECTONIC")
            .map(PathBuf::from)
            .unwrap_or("tectonic".into());
        Command::new(&p).arg("--version").output().ok().map(|_| p)
    }

    #[test]
    #[ignore = "needs tectonic; run with --ignored"]
    fn broken_block_falls_back_and_is_flagged() {
        let tectonic = tectonic().expect("tectonic on PATH or $TECTONIC");
        let dir = std::env::temp_dir().join(format!("makaar-latex-{}", std::process::id()));
        let mut d = doc(vec![
            block(
                BlockType::Text,
                "text",
                0,
                "fine \\(x^2\\) → ok\n\nbroken \\(\\frac{a}\\)",
            ),
            block(BlockType::Formula, "display_formula", 1, "\\frac{a}{b"),
            block(BlockType::Formula, "display_formula", 2, "\\sum_i a_i"),
        ]);
        let bad = export_checked(&mut d, &tectonic, &dir, "out").unwrap();
        assert_eq!(bad, vec![(0, "b0".to_string()), (0, "b1".to_string())]);
        assert_eq!(d.pages[0].blocks[1].flagged, Some(true));
        let tex = std::fs::read_to_string(dir.join("out.tex")).unwrap();
        // only the bad paragraph of b0 falls back
        assert!(tex.contains("fine \\(x^2\\) \\(\\rightarrow\\) ok"));
        assert!(tex.contains("\\begin{verbatim}\nbroken \\(\\frac{a}\\)\n\\end{verbatim}"));
        assert_eq!(d.pages[0].blocks[2].flagged, None);
        assert!(dir.join("out.pdf").exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
