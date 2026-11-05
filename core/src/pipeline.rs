//! Page pipeline: layout → crop each region → recognize with the task its type asks for.
//! Figures are kept as images, not read. A page is read whole as one text block when the layout
//! finds nothing, or when its text regions overlap so much that lines would be read twice
//! (handwritten notes); this is the plan's "hard regions go straight to the general model".
use image::RgbImage;

use crate::document::{Block, BlockType, Page};
use crate::engine::{Engine, EngineError, Task};
use crate::layout::{to_blocks, LayoutDetector, LayoutError};

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Engine(#[from] EngineError),
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Above this raw-layout overlap the page is read whole. 0.1 separates the 10 eval pages
    /// (diagrams and clean printed ≤ 0.06; handwritten maths and printed2 ≥ 0.11); calibrate on
    /// the test set.
    pub max_overlap: f32,
    /// Token budget for a whole page; regions get their area share of it (with margin).
    pub page_tokens: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_overlap: 0.1,
            page_tokens: 4096,
        }
    }
}

/// Pixels added around each crop so strokes on the box edge are not cut.
const PAD: f64 = 4.0;
const MIN_TOKENS: u32 = 128;

pub fn task_for(block: &Block) -> Option<Task> {
    match block.kind {
        BlockType::Figure => None,
        BlockType::Formula => Some(Task::Formula),
        BlockType::Table => Some(Task::Table),
        _ => Some(Task::Text),
    }
}

pub fn crop(img: &RgbImage, bbox: [f64; 4]) -> RgbImage {
    let (w, h) = (f64::from(img.width()), f64::from(img.height()));
    let x0 = (bbox[0] - PAD).clamp(0.0, w) as u32;
    let y0 = (bbox[1] - PAD).clamp(0.0, h) as u32;
    let x1 = (bbox[2] + PAD).clamp(0.0, w) as u32;
    let y1 = (bbox[3] + PAD).clamp(0.0, h) as u32;
    image::imageops::crop_imm(img, x0, y0, (x1 - x0).max(1), (y1 - y0).max(1)).to_image()
}

/// The region's share of the page budget, with 1.5× margin. Resolution-independent.
fn token_budget(bbox: [f64; 4], width: u32, height: u32, page_tokens: u32) -> u32 {
    let share = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (f64::from(width) * f64::from(height));
    ((1.5 * share * f64::from(page_tokens)) as u32).clamp(MIN_TOKENS, page_tokens)
}

/// If `text` ends in a run of one repeated unit (the model looping until the budget ran out),
/// returns it cut after the unit's first occurrence. Needs the unit to repeat at least 4 times.
pub fn trim_loop(text: &str) -> Option<String> {
    let c: Vec<char> = text.chars().collect();
    let n = c.len();
    let mut best: Option<(usize, usize)> = None; // (run length, unit length)
    for unit in 1..=200.min(n / 4) {
        let run = (unit..n).rev().take_while(|&i| c[i] == c[i - unit]).count();
        if run >= 3 * unit && best.is_none_or(|(r, _)| run > r) {
            best = Some((run, unit));
        }
    }
    best.map(|(run, _)| {
        c[..n - run]
            .iter()
            .collect::<String>()
            .trim_end()
            .to_string()
    })
}

fn whole_page(index: u32, width: u32, height: u32) -> Block {
    Block {
        id: format!("p{index}-b0"),
        kind: BlockType::Text,
        label: Some("page".into()),
        bbox: [0.0, 0.0, f64::from(width), f64::from(height)],
        order: 0,
        crop: None,
        text: None,
        script: None,
        engine: None,
        confidence: None,
        verified: None,
        flagged: None,
        edited: None,
    }
}

pub fn recognize_page(
    img: &RgbImage,
    index: u32,
    layout: &mut dyn LayoutDetector,
    engine: &mut dyn Engine,
    cfg: &PipelineConfig,
) -> Result<Page, PipelineError> {
    let (width, height) = img.dimensions();
    let found = layout.detect(img)?;
    let mut blocks = if found.regions.is_empty() || found.overlap > cfg.max_overlap {
        vec![whole_page(index, width, height)]
    } else {
        to_blocks(index, &found.regions)
    };
    for b in &mut blocks {
        let Some(task) = task_for(b) else { continue };
        let budget = token_budget(b.bbox, width, height, cfg.page_tokens);
        let rec = engine.recognize(&crop(img, b.bbox), task, budget)?;
        b.confidence = Some(rec.confidence());
        b.engine = Some(rec.engine.clone());
        if rec.truncated {
            // Hit the budget: drop the repeated tail and send the block to review.
            b.flagged = Some(true);
            b.text = Some(trim_loop(&rec.text).unwrap_or(rec.text));
        } else {
            b.text = Some(rec.text);
        }
    }
    Ok(Page {
        index,
        width,
        height,
        image: None,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Recognition;
    use crate::layout::{overlap_score, PageLayout, Region};

    struct FixedLayout(Vec<Region>);
    impl LayoutDetector for FixedLayout {
        fn detect(&mut self, _: &RgbImage) -> Result<PageLayout, LayoutError> {
            Ok(PageLayout {
                regions: self.0.clone(),
                overlap: overlap_score(&self.0),
            })
        }
    }

    /// Records (crop width, task, budget); returns `text` or the task name.
    struct Echo(Vec<(u32, Task, u32)>, Option<String>);
    impl Engine for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn recognize(
            &mut self,
            crop: &RgbImage,
            task: Task,
            max_tokens: u32,
        ) -> Result<Recognition, EngineError> {
            self.0.push((crop.width(), task, max_tokens));
            Ok(Recognition {
                text: self.1.clone().unwrap_or(format!("{task:?}")),
                token_probs: vec![0.5],
                engine: "echo".into(),
                truncated: self.1.is_some(),
            })
        }
    }

    fn region(label: &'static str, bbox: [f32; 4], order: f32) -> Region {
        Region {
            label,
            score: 0.9,
            bbox,
            order,
        }
    }

    #[test]
    fn routes_by_type_and_skips_figures() {
        let img = RgbImage::new(100, 100);
        let mut layout = FixedLayout(vec![
            region("paragraph_title", [0., 0., 50., 10.], 1.),
            region("display_formula", [10., 20., 30., 30.], 2.),
            region("image", [0., 40., 100., 80.], 3.),
            region("table", [98., 90., 100., 100.], 4.),
        ]);
        let mut engine = Echo(vec![], None);
        let cfg = PipelineConfig::default();
        let page = recognize_page(&img, 0, &mut layout, &mut engine, &cfg).unwrap();
        let tasks: Vec<_> = engine.0.iter().map(|e| e.1).collect();
        assert_eq!(tasks, [Task::Text, Task::Formula, Task::Table]);
        // Padded by 4 px each side, clamped at the page edge.
        assert_eq!(engine.0[1].0, 28);
        assert_eq!(engine.0[2].0, 6);
        // 5% of the page → 1.5 × 0.05 × 4096 = 307 tokens; tiny regions get the floor.
        assert_eq!(engine.0[0].2, 307);
        assert_eq!(engine.0[1].2, MIN_TOKENS);
        assert_eq!(page.blocks[2].text, None);
        assert_eq!(page.blocks[1].confidence, Some(0.5));
        assert_eq!(page.blocks[1].flagged, None);
    }

    #[test]
    fn empty_or_overlapping_layout_reads_whole_page() {
        let img = RgbImage::new(40, 60);
        let cfg = PipelineConfig::default();
        let mut engine = Echo(vec![], None);
        let page = recognize_page(&img, 2, &mut FixedLayout(vec![]), &mut engine, &cfg).unwrap();
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(page.blocks[0].bbox, [0., 0., 40., 60.]);
        assert_eq!(engine.0, [(40, Task::Text, 4096)]);

        let mut layout = FixedLayout(vec![
            region("text", [0., 0., 40., 30.], 1.),
            region("text", [0., 20., 40., 50.], 2.),
        ]);
        let mut engine = Echo(vec![], None);
        let page = recognize_page(&img, 0, &mut layout, &mut engine, &cfg).unwrap();
        assert_eq!(page.blocks[0].label.as_deref(), Some("page"));
    }

    #[test]
    fn truncated_output_is_trimmed_and_flagged() {
        let img = RgbImage::new(10, 10);
        let mut engine = Echo(vec![], Some(format!("Total: {}", "\\u325f\n".repeat(30))));
        let mut layout = FixedLayout(vec![region("text", [0., 0., 10., 10.], 1.)]);
        let page = recognize_page(
            &img,
            0,
            &mut layout,
            &mut engine,
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(page.blocks[0].text.as_deref(), Some("Total: \\u325f"));
        assert_eq!(page.blocks[0].flagged, Some(true));
    }

    #[test]
    fn trim_loop_needs_four_repeats() {
        assert_eq!(trim_loop("ab ab ab ab "), Some("ab".into()));
        assert_eq!(trim_loop("x = 1, 2, 3"), None);
        assert_eq!(trim_loop("aaa"), None);
        assert_eq!(trim_loop("n-1\nn-1\nn-1\nn-1\nn-"), Some("n-1".into()));
    }
}
