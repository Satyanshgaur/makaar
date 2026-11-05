//! Layout stage: find regions on a page and their reading order. `LayoutDetector` is the swappable
//! interface (a notebook-trained model can replace PP-DocLayoutV3 later); `postprocess` is pure so
//! it is tested without the model.
use std::path::Path;

use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::document::{Block, BlockType};

#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("onnx runtime: {0}")]
    Ort(String),
    #[error("unexpected model output: {0}")]
    Output(String),
}

impl<T> From<ort::Error<T>> for LayoutError {
    fn from(e: ort::Error<T>) -> Self {
        LayoutError::Ort(e.to_string())
    }
}

/// A detected region in page pixels. `order` is the model's reading-order key, not yet dense.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub label: &'static str,
    pub score: f32,
    pub bbox: [f32; 4],
    pub order: f32,
}

impl Region {
    pub fn kind(&self) -> BlockType {
        block_type(self.label)
    }
}

/// Regions in reading order, plus how messy the raw layout was: `overlap_score` of the regions
/// before container boxes were removed. The pipeline reads high-overlap pages whole.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PageLayout {
    pub regions: Vec<Region>,
    pub overlap: f32,
}

pub trait LayoutDetector {
    fn detect(&mut self, page: &RgbImage) -> Result<PageLayout, LayoutError>;
}

#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Minimum score. 0.5 (the model's draw threshold) misses most regions on handwritten pages;
    /// 0.3 is eyeballed on the eval pages and must be calibrated on the test set.
    pub threshold: f32,
    /// Same-label boxes overlapping more than this are duplicates.
    pub same_label_iou: f32,
    /// Any two boxes overlapping more than this are one region with two guessed labels.
    pub any_label_iou: f32,
    /// Share of a box's area inside a kept box above which it counts as nested.
    pub contained: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            same_label_iou: 0.5,
            any_label_iou: 0.9,
            contained: 0.9,
        }
    }
}

/// PP-DocLayoutV3 class ids, from the model's `inference.yml`.
pub const LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

pub fn block_type(label: &str) -> BlockType {
    match label {
        "doc_title" | "paragraph_title" => BlockType::Heading,
        "display_formula" | "inline_formula" => BlockType::Formula,
        "table" => BlockType::Table,
        "image" | "chart" | "seal" | "header_image" | "footer_image" => BlockType::Figure,
        "figure_title" | "vision_footnote" | "formula_number" => BlockType::Caption,
        _ => BlockType::Text,
    }
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn intersection(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    area(&[
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ])
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let i = intersection(a, b);
    let u = area(a) + area(b) - i;
    if u > 0.0 {
        i / u
    } else {
        0.0
    }
}

/// Share of `inner`'s area that lies inside `outer`.
fn inside(inner: &[f32; 4], outer: &[f32; 4]) -> f32 {
    let a = area(inner);
    if a > 0.0 {
        intersection(inner, outer) / a
    } else {
        1.0
    }
}

/// Turns raw rows `[class, score, x0, y0, x1, y1, order]` (page pixels) into regions in reading
/// order. Rows are visited by descending score; a row is dropped if it duplicates or nests inside
/// a kept region, or if it is an inline formula inside kept text (recognized with that text).
pub fn postprocess(rows: &[[f32; 7]], width: u32, height: u32, cfg: &LayoutConfig) -> PageLayout {
    let (w, h) = (width as f32, height as f32);
    let mut cands: Vec<Region> = rows
        .iter()
        .filter(|r| r[1] >= cfg.threshold && r[0] >= 0.0 && (r[0] as usize) < LABELS.len())
        .map(|r| Region {
            label: LABELS[r[0] as usize],
            score: r[1],
            bbox: [
                r[2].clamp(0.0, w),
                r[3].clamp(0.0, h),
                r[4].clamp(0.0, w),
                r[5].clamp(0.0, h),
            ],
            order: r[6],
        })
        .filter(|r| area(&r.bbox) > 0.0)
        .collect();
    cands.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut kept: Vec<Region> = Vec::new();
    for c in cands {
        let dup = kept.iter().any(|k| {
            let same = k.label == c.label;
            let o = iou(&c.bbox, &k.bbox);
            (same && o > cfg.same_label_iou)
                || o > cfg.any_label_iou
                || (same && inside(&c.bbox, &k.bbox) > cfg.contained)
                || (c.label == "inline_formula"
                    && k.kind() == BlockType::Text
                    && inside(&c.bbox, &k.bbox) > cfg.contained)
        });
        if !dup {
            kept.push(c);
        }
    }
    let overlap = overlap_score(&kept);
    // A non-figure box around two or more kept regions repeats their text (e.g. a bullet list
    // also boxed as `algorithm`); keep the finer regions. Figures keep their inner labels.
    let containers: Vec<usize> = (0..kept.len())
        .filter(|&i| {
            kept[i].kind() != BlockType::Figure
                && kept
                    .iter()
                    .enumerate()
                    .filter(|&(j, r)| j != i && inside(&r.bbox, &kept[i].bbox) > cfg.contained)
                    .count()
                    >= 2
        })
        .collect();
    let mut kept: Vec<Region> = kept
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !containers.contains(i))
        .map(|(_, r)| r)
        .collect();
    kept.sort_by(|a, b| a.order.total_cmp(&b.order));
    PageLayout {
        regions: kept,
        overlap,
    }
}

/// Overlap between non-figure regions: summed pairwise intersection over summed area. Clean
/// printed layouts score ~0; handwritten pages whose text bands overlap score 0.1–0.4, and their
/// regions would read the same lines twice.
pub fn overlap_score(regions: &[Region]) -> f32 {
    let rs: Vec<&Region> = regions
        .iter()
        .filter(|r| r.kind() != BlockType::Figure)
        .collect();
    let total: f32 = rs.iter().map(|r| area(&r.bbox)).sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut over = 0.0;
    for (i, a) in rs.iter().enumerate() {
        for b in &rs[i + 1..] {
            over += intersection(&a.bbox, &b.bbox);
        }
    }
    over / total
}

/// Regions → document blocks with dense reading order. Ids are `p{page}-b{n}`.
pub fn to_blocks(page: u32, regions: &[Region]) -> Vec<Block> {
    regions
        .iter()
        .enumerate()
        .map(|(n, r)| Block {
            id: format!("p{page}-b{n}"),
            kind: r.kind(),
            label: Some(r.label.to_string()),
            bbox: r.bbox.map(f64::from),
            order: n as u32,
            crop: None,
            text: None,
            script: None,
            engine: None,
            confidence: Some(f64::from(r.score)),
            verified: None,
            flagged: None,
            edited: None,
        })
        .collect()
}

const SIDE: u32 = 800;

/// PP-DocLayoutV3 on ONNX Runtime (CPU). Input is the page squashed to 800×800 in 0–1 RGB;
/// `scale_factor` makes the model return boxes in page pixels.
pub struct PpDocLayoutV3 {
    session: Session,
    pub config: LayoutConfig,
}

impl PpDocLayoutV3 {
    /// `threads` caps ONNX Runtime's intra-op pool so a batch leaves the laptop usable.
    pub fn load(model: &Path, threads: usize) -> Result<Self, LayoutError> {
        // ONNX Runtime's own telemetry is off: nothing leaves the laptop.
        let env = ort::init()
            .with_name("makaar")
            .with_telemetry(false)
            .build()?;
        let session = Session::builder(&env)?
            .with_intra_threads(threads)?
            .commit_from_file(model)?;
        Ok(Self {
            session,
            config: LayoutConfig::default(),
        })
    }
}

impl LayoutDetector for PpDocLayoutV3 {
    fn detect(&mut self, page: &RgbImage) -> Result<PageLayout, LayoutError> {
        let (w, h) = page.dimensions();
        let small = image::imageops::resize(page, SIDE, SIDE, FilterType::CatmullRom);
        let plane = (SIDE * SIDE) as usize;
        let mut chw = vec![0f32; 3 * plane];
        for (i, px) in small.pixels().enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = f32::from(px[c]) / 255.0;
            }
        }
        let side = SIDE as f32;
        let outputs = self.session.run(ort::inputs![
            "image" => Tensor::from_array(([1usize, 3, SIDE as usize, SIDE as usize], chw))?,
            "im_shape" => Tensor::from_array(([1usize, 2], vec![side, side]))?,
            "scale_factor" => Tensor::from_array(([1usize, 2], vec![side / h as f32, side / w as f32]))?,
        ])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 2 || shape[1] != 7 {
            return Err(LayoutError::Output(format!(
                "boxes shape {shape:?}, want [N, 7]"
            )));
        }
        let (rows, _) = data.as_chunks::<7>();
        Ok(postprocess(rows, w, h, &self.config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(label: &str, score: f32, b: [f32; 4], order: f32) -> [f32; 7] {
        let c = LABELS.iter().position(|l| *l == label).unwrap() as f32;
        [c, score, b[0], b[1], b[2], b[3], order]
    }

    #[test]
    fn dedups_same_label_and_nested() {
        // From printed1.png: three overlapping `algorithm` boxes for one code listing.
        let rows = [
            row("algorithm", 0.54, [98., 75., 526., 330.], 29.),
            row("algorithm", 0.71, [112., 77., 452., 335.], 30.),
            row("algorithm", 0.43, [117., 76., 357., 288.], 27.),
            row("paragraph_title", 0.57, [110., 32., 441., 63.], 10.),
        ];
        let r = postprocess(&rows, 607, 806, &LayoutConfig::default()).regions;
        let got: Vec<_> = r.iter().map(|r| (r.label, r.score)).collect();
        assert_eq!(got, [("paragraph_title", 0.57), ("algorithm", 0.71)]);
    }

    #[test]
    fn container_of_two_regions_is_dropped() {
        // From printed2.png: the bullet list is boxed as `algorithm` and as separate `text`s.
        let rows = [
            row("algorithm", 0.99, [96., 61., 549., 349.], 20.),
            row("text", 0.96, [106., 67., 517., 129.], 10.),
            row("text", 0.98, [105., 133., 518., 196.], 30.),
            row("image", 0.9, [300., 400., 600., 700.], 40.),
            row("text", 0.9, [320., 420., 400., 440.], 41.),
            row("text", 0.9, [320., 450., 400., 470.], 42.),
        ];
        let r = postprocess(&rows, 607, 806, &LayoutConfig::default()).regions;
        let got: Vec<_> = r.iter().map(|r| r.label).collect();
        assert_eq!(got, ["text", "text", "image", "text", "text"]);
        assert!(overlap_score(&r) < 0.01);
        let raw = postprocess(&rows, 607, 806, &LayoutConfig::default()).overlap;
        assert!(raw > 0.2, "{raw}");
    }

    #[test]
    fn overlap_score_counts_shared_area() {
        let rows = [
            row("text", 0.9, [0., 0., 100., 100.], 1.),
            row("text", 0.8, [0., 75., 100., 175.], 2.),
        ];
        let r = postprocess(&rows, 200, 200, &LayoutConfig::default()).regions;
        assert_eq!(r.len(), 2);
        assert!((overlap_score(&r) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn one_box_two_labels_keeps_higher() {
        let rows = [
            row("display_formula", 0.32, [53., 35., 283., 64.], 5.),
            row("text", 0.36, [53., 35., 283., 64.], 5.),
        ];
        let r = postprocess(&rows, 607, 806, &LayoutConfig::default()).regions;
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].label, "text");
    }

    #[test]
    fn inline_formula_inside_text_is_dropped_but_alone_kept() {
        let rows = [
            row("text", 0.75, [43., 247., 564., 402.], 128.),
            row("inline_formula", 0.38, [271., 256., 408., 289.], 131.),
            row("inline_formula", 0.33, [411., 500., 470., 520.], 186.),
        ];
        let r = postprocess(&rows, 607, 806, &LayoutConfig::default()).regions;
        let got: Vec<_> = r.iter().map(|r| r.bbox[1]).collect();
        assert_eq!(got, [247., 500.]);
    }

    #[test]
    fn threshold_clamp_and_order() {
        let rows = [
            row("text", 0.9, [-3., 400., 700., 420.], 50.),
            row("text", 0.29, [0., 0., 10., 10.], 1.),
            row("doc_title", 0.8, [10., 10., 200., 40.], 2.),
        ];
        let r = postprocess(&rows, 607, 806, &LayoutConfig::default()).regions;
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].label, "doc_title");
        assert_eq!(r[1].bbox, [0., 400., 607., 420.]);
        let blocks = to_blocks(3, &r);
        assert_eq!(blocks[1].id, "p3-b1");
        assert_eq!(blocks[1].order, 1);
        assert_eq!(blocks[0].kind, BlockType::Heading);
        assert_eq!(blocks[0].label.as_deref(), Some("doc_title"));
    }

    /// Needs the model: `hf download PaddlePaddle/PP-DocLayoutV3_onnx --local-dir models/pp-doclayoutv3`.
    #[test]
    #[ignore]
    fn model_on_eval_page() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut m =
            PpDocLayoutV3::load(&root.join("models/pp-doclayoutv3/inference.onnx"), 4).unwrap();
        let page = crate::ingest::load_page(&root.join("eval/data/printed/printed3.png")).unwrap();
        let r = m.detect(&page).unwrap().regions;
        assert!(r.iter().any(|r| r.label == "paragraph_title"), "{r:?}");
        assert!(r.iter().filter(|r| r.label == "text").count() >= 5, "{r:?}");
    }
}
