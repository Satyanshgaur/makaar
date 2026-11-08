//! Page pipeline: layout → crop each region → recognize with the task its type asks for.
//! Figures are kept as images, not read. A page is read whole as one text block when the layout
//! finds nothing, or when its text regions overlap so much that lines would be read twice
//! (handwritten notes); this is the plan's "hard regions go straight to the general model".
//! With a specialist, prose regions try it first and keep its text only when its character
//! confidence clears `accept`; everything else goes to the general model.
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
    /// A specialist's text is kept when its mean character probability is at least this.
    pub accept: f64,
    /// General-model blocks below this mean token probability are flagged for review.
    pub flag_below: f64,
    /// When the specialist was rejected and the general model read the block, flag it if the
    /// two readings differ by more than this (normalized edit distance).
    pub flag_disagree: f64,
    /// Re-read short text-prompt readings with the formula prompt (see `FORMULA_MAX_WORDS`).
    pub formula_retry: bool,
    /// Widen text and formula boxes along their line to take in ink next to them (see `grow_to_ink`).
    pub grow_to_ink: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_overlap: 0.1,
            page_tokens: 4096,
            accept: ACCEPT,
            flag_below: FLAG_BELOW,
            flag_disagree: FLAG_DISAGREE,
            formula_retry: true,
            grow_to_ink: true,
        }
    }
}

/// Default specialist acceptance; see DECISIONS for the calibration.
pub const ACCEPT: f64 = 0.99;

/// On 2,068 OmniDocBench blocks read by both models, disagreement above 0.3 flags 4% of blocks
/// with 0.48 precision; token confidence alone barely separates bad printed blocks.
pub const FLAG_DISAGREE: f64 = 0.3;

/// Edit distance over characters divided by the longer length (0 = same, 1 = nothing shared).
pub fn disagreement(a: &str, b: &str) -> f64 {
    let (a, b): (Vec<char>, Vec<char>) = (
        a.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .collect(),
        b.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .collect(),
    );
    let n = a.len().max(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push(
                (prev[j + 1] + 1)
                    .min(cur[j] + 1)
                    .min(prev[j] + usize::from(ca != cb)),
            );
        }
        prev = cur;
    }
    prev[b.len()] as f64 / n as f64
}

/// PaddleOCR-VL's mean token probability is 0.75–0.89 on handwritten blocks and 0.91–1.00 on
/// printed ones in the eval set; provisional until the test set calibrates it.
pub const FLAG_BELOW: f64 = 0.9;

/// Prose regions a line reader can handle. Code (`algorithm`) needs its layout kept, and the
/// whole-page fallback is handwriting, so both go to the general model.
pub fn specialist_eligible(b: &Block) -> bool {
    matches!(
        b.kind,
        BlockType::Text | BlockType::Heading | BlockType::Caption
    ) && !matches!(
        b.label.as_deref(),
        Some("algorithm" | "page" | "formula_number")
    )
}

/// Joins a line reader's lines into one paragraph, undoing end-of-line hyphenation.
pub fn reflow(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if out.ends_with('-') && line.starts_with(|c: char| c.is_lowercase()) {
            out.pop();
        } else if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(line);
    }
    out
}

/// Words of three or more letters, the test for "this reading is not prose".
fn words(text: &str) -> usize {
    text.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| w.len() >= 3)
        .count()
}

/// A formula reading that is really prose: a word of three or more letters set in `\mathrm`,
/// `\text` and the like (the formula prompt turns printed `Culture` into `\mathrm{C u l t u r e}`),
/// or CJK text.
fn wordy_formula(f: &str) -> bool {
    const WRAPPERS: [&str; 6] = [
        "\\mathrm{",
        "\\text{",
        "\\texttt{",
        "\\textbf{",
        "\\mathbf{",
        "\\operatorname{",
    ];
    let cjk = f.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
    cjk || WRAPPERS.iter().any(|w| {
        f.match_indices(w).any(|(i, _)| {
            let inner = f[i + w.len()..].split('}').next().unwrap_or("");
            inner.chars().filter(char::is_ascii_alphabetic).count() >= 3
        })
    })
}

/// A text-prompt reading with at most this many words is re-read as a formula. The layout model
/// often calls a lone handwritten expression a title or a paragraph, and the text prompt then
/// returns `## 39` for `3 9`, or CJK look-alikes (`工工` for `I I`).
pub const FORMULA_MAX_WORDS: usize = 2;
/// The formula reading wins unless its confidence is this much below the text reading's. On 450
/// handwritten crops (CROHME, MathWriting, IAM lines) this takes compact NED from 0.306 to 0.201
/// on CROHME and 0.247 to 0.179 on MathWriting, with IAM unchanged (0.061 → 0.058).
pub const FORMULA_MARGIN: f64 = 0.05;

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

/// Ink: pixels well below the page's median brightness (grey paper and photos included).
fn ink_mask(img: &RgbImage) -> Vec<bool> {
    let luma: Vec<u8> = img
        .pixels()
        .map(|p| {
            ((u32::from(p[0]) * 299 + u32::from(p[1]) * 587 + u32::from(p[2]) * 114) / 1000) as u8
        })
        .collect();
    let mut hist = [0usize; 256];
    for &l in &luma {
        hist[usize::from(l)] += 1;
    }
    let (mut seen, mut median) = (0, 255u8);
    for (v, &n) in hist.iter().enumerate() {
        seen += n;
        if seen * 2 >= luma.len() {
            median = v as u8;
            break;
        }
    }
    let threshold = median.saturating_sub(40).min(128);
    luma.into_iter().map(|l| l < threshold).collect()
}

/// The layout model often boxes only part of a handwritten line (`=0` of `N[u(x)]=0`), and the
/// rest is never read. Widen each text and formula box left and right along its own rows while
/// there is ink within a gap of about half the box height, stopping at other boxes on that line.
fn grow_to_ink(img: &RgbImage, blocks: &mut [Block]) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let ink = ink_mask(img);
    let boxes: Vec<[f64; 4]> = blocks.iter().map(|b| b.bbox).collect();
    for (i, b) in blocks.iter_mut().enumerate() {
        if !matches!(
            b.kind,
            BlockType::Text | BlockType::Heading | BlockType::Formula
        ) {
            continue;
        }
        let [x0, y0, x1, y1] = b.bbox;
        let (r0, r1) = ((y0.max(0.0) as usize).min(h), (y1.max(0.0) as usize).min(h));
        if r1 <= r0 {
            continue;
        }
        let max_gap = ((y1 - y0) * 0.6).max(8.0) as usize;
        let inked = |x: usize| (r0..r1).filter(|&y| ink[y * w + x]).count() >= 2;
        // columns covered by another box that shares these rows
        let blocked = |x: usize| {
            let x = x as f64;
            boxes
                .iter()
                .enumerate()
                .any(|(j, o)| j != i && o[0] <= x && x <= o[2] && o[1] < y1 && y0 < o[3])
        };
        let walk = |cols: &mut dyn Iterator<Item = usize>| {
            let (mut last, mut gap) = (None, 0);
            for x in cols {
                if blocked(x) {
                    break;
                }
                if inked(x) {
                    last = Some(x);
                    gap = 0;
                } else {
                    gap += 1;
                    if gap > max_gap {
                        break;
                    }
                }
            }
            last
        };
        let left = (x0.max(0.0) as usize).min(w);
        let right = (x1.max(0.0).ceil() as usize).min(w);
        if let Some(x) = walk(&mut (0..left).rev()) {
            b.bbox[0] = x as f64;
        }
        if let Some(x) = walk(&mut (right..w)) {
            b.bbox[2] = (x + 1) as f64;
        }
    }
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
    recognize_page_routed(img, index, layout, None, engine, cfg)
}

pub fn recognize_page_routed(
    img: &RgbImage,
    index: u32,
    layout: &mut dyn LayoutDetector,
    mut specialist: Option<&mut dyn Engine>,
    engine: &mut dyn Engine,
    cfg: &PipelineConfig,
) -> Result<Page, PipelineError> {
    let (width, height) = img.dimensions();
    let found = layout.detect(img)?;
    let mut blocks = if found.regions.is_empty() || found.overlap > cfg.max_overlap {
        vec![whole_page(index, width, height)]
    } else if found.regions.iter().all(|r| r.kind() == BlockType::Figure) {
        // Only figures: a lone handwritten formula or derivation is often taken for a drawing,
        // and figures aren't read. Keep the crops and read the page too, so nothing is dropped.
        let mut b = to_blocks(index, &found.regions);
        let mut page = whole_page(index, width, height);
        page.id = format!("p{index}-b{}", b.len());
        page.order = b.len() as u32;
        b.push(page);
        b
    } else {
        let mut b = to_blocks(index, &found.regions);
        if cfg.grow_to_ink {
            grow_to_ink(img, &mut b);
        }
        b
    };
    for b in &mut blocks {
        let Some(task) = task_for(b) else { continue };
        let budget = token_budget(b.bbox, width, height, cfg.page_tokens);
        let region = crop(img, b.bbox);
        let mut draft = None;
        if let Some(s) = specialist.as_mut().filter(|_| specialist_eligible(b)) {
            let rec = s.recognize(&region, Task::Text, budget)?;
            draft = Some(reflow(&rec.text));
            if !rec.text.trim().is_empty() && rec.confidence() >= cfg.accept {
                b.text = Some(reflow(&rec.text));
                b.confidence = Some(rec.confidence());
                b.engine = Some(rec.engine);
                b.verified = Some(true); // passed the confidence check
                continue;
            }
        }
        let mut rec = engine.recognize(&region, task, budget)?;
        if cfg.formula_retry
            && task == Task::Text
            && b.label.as_deref() != Some("algorithm")
            && !rec.truncated
            && words(&rec.text) <= FORMULA_MAX_WORDS
        {
            let f = engine.recognize(&region, Task::Formula, budget)?;
            if !f.text.trim().is_empty()
                && !f.truncated
                && !wordy_formula(&f.text)
                && f.confidence() >= rec.confidence() - FORMULA_MARGIN
            {
                b.kind = BlockType::Formula;
                rec = f;
            }
        }
        b.confidence = Some(rec.confidence());
        b.engine = Some(rec.engine.clone());
        let disagrees = draft.is_some_and(|d| disagreement(&d, &rec.text) > cfg.flag_disagree);
        if rec.confidence() < cfg.flag_below || disagrees {
            b.flagged = Some(true);
        }
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
                text: self.1.clone().unwrap_or(format!("{task:?} read as prose")),
                token_probs: vec![0.5],
                engine: "echo".into(),
                truncated: self.1.is_some(),
            })
        }
    }

    /// A specialist that returns fixed text at a fixed confidence.
    struct Fixed(&'static str, f64);
    impl Engine for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }
        fn recognize(&mut self, _: &RgbImage, _: Task, _: u32) -> Result<Recognition, EngineError> {
            Ok(Recognition {
                text: self.0.into(),
                token_probs: vec![self.1],
                engine: "fixed".into(),
                truncated: false,
            })
        }
    }

    #[test]
    fn specialist_kept_only_when_confident() {
        let img = RgbImage::new(100, 100);
        let regions = vec![
            region("text", [0., 0., 50., 10.], 1.),
            region("display_formula", [10., 20., 30., 30.], 2.),
            region("algorithm", [0., 40., 100., 60.], 3.),
        ];
        let cfg = PipelineConfig::default();
        let mut general = Echo(vec![], None);
        let mut sure = Fixed("hyphen-\nated line", 0.99);
        let page = recognize_page_routed(
            &img,
            0,
            &mut FixedLayout(regions.clone()),
            Some(&mut sure),
            &mut general,
            &cfg,
        )
        .unwrap();
        assert_eq!(page.blocks[0].text.as_deref(), Some("hyphenated line"));
        assert_eq!(page.blocks[0].verified, Some(true));
        // formula and code still go to the general model
        assert_eq!(general.0.len(), 2);

        let mut general = Echo(vec![], None);
        let mut unsure = Fixed("guess", 0.6);
        let page = recognize_page_routed(
            &img,
            0,
            &mut FixedLayout(regions),
            Some(&mut unsure),
            &mut general,
            &cfg,
        )
        .unwrap();
        assert_eq!(page.blocks[0].text.as_deref(), Some("Text read as prose"));
        assert_eq!(general.0.len(), 3);
        // Echo reports probability 0.5, below FLAG_BELOW
        assert!(page.blocks.iter().all(|b| b.flagged == Some(true)));
    }

    #[test]
    fn figures_only_page_is_also_read_whole() {
        let img = RgbImage::new(100, 100);
        let mut layout = FixedLayout(vec![region("image", [10., 10., 90., 60.], 1.)]);
        let mut engine = Echo(vec![], None);
        let page = recognize_page(
            &img,
            0,
            &mut layout,
            &mut engine,
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(page.blocks.len(), 2);
        assert_eq!(page.blocks[0].kind, BlockType::Figure);
        assert_eq!(page.blocks[1].label.as_deref(), Some("page"));
        assert_eq!(page.blocks[1].id, "p0-b1");
        assert_eq!(engine.0.len(), 1); // only the page is read
    }

    /// Reads `## 39` with the text prompt and `\[3 9\]` with the formula prompt.
    struct ByTask(f64, f64);
    impl Engine for ByTask {
        fn name(&self) -> &str {
            "by-task"
        }
        fn recognize(
            &mut self,
            _: &RgbImage,
            task: Task,
            _: u32,
        ) -> Result<Recognition, EngineError> {
            let (text, p) = match task {
                Task::Formula => ("\\[3 9\\]", self.1),
                _ => ("## 39", self.0),
            };
            Ok(Recognition {
                text: text.into(),
                token_probs: vec![p],
                engine: "by-task".into(),
                truncated: false,
            })
        }
    }

    #[test]
    fn short_text_readings_are_retried_as_formulas() {
        let img = RgbImage::new(100, 100);
        let read = |text_p, formula_p| {
            let regions = vec![region("paragraph_title", [0., 0., 50., 10.], 1.)];
            recognize_page(
                &img,
                0,
                &mut FixedLayout(regions),
                &mut ByTask(text_p, formula_p),
                &PipelineConfig::default(),
            )
            .unwrap()
            .blocks
            .remove(0)
        };
        let b = read(0.8, 0.78);
        assert_eq!(b.kind, BlockType::Formula);
        assert_eq!(b.text.as_deref(), Some("\\[3 9\\]"));
        assert!(wordy_formula("\\[\\mathrm{C u l t u r e}\\]"));
        assert!(wordy_formula("\\text{then } x"));
        assert!(!wordy_formula("\\int \\mathrm{d} A + \\sin x"));
        let b = read(0.95, 0.85); // formula reading much less sure: keep the text
        assert_ne!(b.kind, BlockType::Formula);
        assert_eq!(b.text.as_deref(), Some("## 39"));
    }

    #[test]
    fn boxes_grow_to_adjacent_ink_on_their_line() {
        // white page; ink at x 10..30 and 40..60 on rows 20..40, and far away at x 150..160
        let mut img = RgbImage::from_pixel(200, 60, image::Rgb([250, 250, 250]));
        for y in 20..40 {
            for x in (10..30).chain(40..60).chain(150..160) {
                img.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        let mut blocks = to_blocks(0, &[region("text", [40., 18., 60., 42.], 1.)]);
        grow_to_ink(&img, &mut blocks);
        assert_eq!(blocks[0].bbox, [10.0, 18.0, 60.0, 42.0]); // took in 10..30, not 150..160

        // a neighbouring box on the same line stops the growth
        let mut blocks = to_blocks(
            0,
            &[
                region("text", [40., 18., 60., 42.], 1.),
                region("text", [25., 18., 32., 42.], 2.),
            ],
        );
        grow_to_ink(&img, &mut blocks);
        assert_eq!(blocks[0].bbox[0], 40.0);
    }

    #[test]
    fn disagreement_flags_rejected_blocks() {
        assert_eq!(disagreement("a  b", "a b"), 0.0);
        assert!((disagreement("abcd", "abxy") - 0.5).abs() < 1e-9);
        let img = RgbImage::new(100, 100);
        let regions = vec![region("text", [0., 0., 50., 10.], 1.)];
        let mut general = Fixed("the general reading", 0.99);
        let mut unsure_but_close = Fixed("the general readinq", 0.6);
        let page = recognize_page_routed(
            &img,
            0,
            &mut FixedLayout(regions.clone()),
            Some(&mut unsure_but_close),
            &mut general,
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(page.blocks[0].flagged, None);
        let mut unsure_and_different = Fixed("something else entirely", 0.6);
        let page = recognize_page_routed(
            &img,
            0,
            &mut FixedLayout(regions),
            Some(&mut unsure_and_different),
            &mut general,
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(page.blocks[0].flagged, Some(true));
    }

    #[test]
    fn reflow_joins_lines() {
        assert_eq!(
            reflow("a long para-\ngraph\n\nends\nhere"),
            "a long paragraph ends here"
        );
        assert_eq!(reflow("Self-\nAssessment"), "Self- Assessment");
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
        assert_eq!(page.blocks[1].flagged, Some(true)); // 0.5 < FLAG_BELOW
        assert_eq!(page.blocks[2].flagged, None); // figures aren't read
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
