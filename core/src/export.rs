//! Markdown export: blocks in reading order, display formulas as `$$…$$`, code (`algorithm`)
//! fenced, tables as the HTML the engine returned, figures as image links when a crop was saved.
use crate::document::{Block, BlockType, Document, Page};

/// Removes one layer of `$$…$$`, `\[…\]` or `$…$` that some engines add around LaTeX.
pub fn strip_math_delims(s: &str) -> &str {
    let t = s.trim();
    for (open, close) in [("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")] {
        if t.len() >= open.len() + close.len() && t.starts_with(open) && t.ends_with(close) {
            return t[open.len()..t.len() - close.len()].trim();
        }
    }
    t
}

fn block_md(b: &Block) -> Option<String> {
    if b.kind == BlockType::Figure {
        return b.crop.as_ref().map(|c| format!("![]({c})"));
    }
    let text = b.text.as_deref()?.trim();
    if text.is_empty() {
        return None;
    }
    Some(match (&b.kind, b.label.as_deref()) {
        (BlockType::Heading, Some("doc_title")) => format!("# {text}"),
        (BlockType::Heading, _) => format!("## {text}"),
        (BlockType::Formula, Some("inline_formula")) => format!("${}$", strip_math_delims(text)),
        (BlockType::Formula, _) => format!("$$\n{}\n$$", strip_math_delims(text)),
        (BlockType::Text, Some("algorithm")) if !text.starts_with("```") => {
            format!("```\n{text}\n```")
        }
        _ => text.to_string(),
    })
}

pub fn page_markdown(page: &Page) -> String {
    let mut blocks: Vec<&Block> = page.blocks.iter().collect();
    blocks.sort_by_key(|b| b.order);
    let parts: Vec<String> = blocks.into_iter().filter_map(block_md).collect();
    parts.join("\n\n") + "\n"
}

pub fn markdown(doc: &Document) -> String {
    doc.pages
        .iter()
        .map(page_markdown)
        .collect::<Vec<_>>()
        .join("\n---\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(kind: BlockType, label: &str, order: u32, text: Option<&str>) -> Block {
        Block {
            id: format!("b{order}"),
            kind,
            label: Some(label.into()),
            bbox: [0.0; 4],
            order,
            crop: None,
            text: text.map(String::from),
            script: None,
            engine: None,
            confidence: None,
            verified: None,
            flagged: None,
            edited: None,
        }
    }

    #[test]
    fn strips_one_layer_of_delimiters() {
        assert_eq!(strip_math_delims(" $$x^2$$ "), "x^2");
        assert_eq!(strip_math_delims("\\[ a+b \\]"), "a+b");
        assert_eq!(strip_math_delims("x$"), "x$");
        assert_eq!(strip_math_delims("$"), "$");
    }

    #[test]
    fn page_in_reading_order() {
        let page = Page {
            index: 0,
            width: 1,
            height: 1,
            image: None,
            blocks: vec![
                block(
                    BlockType::Formula,
                    "display_formula",
                    2,
                    Some("\\[E=mc^2\\]"),
                ),
                block(BlockType::Heading, "paragraph_title", 0, Some("Energy")),
                block(BlockType::Figure, "image", 3, None),
                block(BlockType::Text, "text", 1, Some("It is ")),
                block(BlockType::Formula, "inline_formula", 4, Some("c")),
                block(BlockType::Text, "algorithm", 5, Some("x++;")),
            ],
        };
        assert_eq!(
            page_markdown(&page),
            "## Energy\n\nIt is\n\n$$\nE=mc^2\n$$\n\n$c$\n\n```\nx++;\n```\n"
        );
    }
}
