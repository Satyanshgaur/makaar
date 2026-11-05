//! PP-OCRv6 small (text detection + line recognition, Apache-2.0, ~31 MB) on ONNX Runtime: the
//! printed-text specialist. It finds text lines in a region crop with DB, reads each with CTC,
//! and reports per-character probabilities so the router can decide whether to trust it.
use std::path::Path;

use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::engine::{Engine, EngineError, Recognition, Task};

#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error("onnx runtime: {0}")]
    Ort(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model files: {0}")]
    Model(String),
}

impl<T> From<ort::Error<T>> for OcrError {
    fn from(e: ort::Error<T>) -> Self {
        OcrError::Ort(e.to_string())
    }
}

/// DB post-processing settings from the model's `inference.yml`.
const BIN_THRESH: f32 = 0.2;
const BOX_THRESH: f32 = 0.45;
const UNCLIP: f32 = 1.4;
/// Detection input: longest side at most this, both sides multiples of 32.
const DET_MAX_SIDE: u32 = 960;
/// White border added before detection so lines touching the crop edge are found.
const DET_PAD: u32 = 8;
const REC_HEIGHT: u32 = 48;
const REC_MAX_WIDTH: u32 = 3200;

pub struct PpOcr {
    det: Session,
    rec: Session,
    /// CTC classes: 0 is blank, 1..=n the dictionary, n+1 a space.
    dict: Vec<String>,
}

/// One text line: its box in crop pixels, text and per-character probabilities.
#[derive(Debug, Clone)]
pub struct Line {
    pub bbox: [u32; 4],
    pub text: String,
    pub probs: Vec<f32>,
}

/// Reads `character_dict` from a PaddleOCR `inference.yml` (plain or single-quoted scalars).
pub fn parse_dict(yml: &str) -> Vec<String> {
    let mut lines = yml
        .lines()
        .skip_while(|l| l.trim() != "character_dict:")
        .skip(1);
    let mut out = vec![];
    for l in lines.by_ref() {
        let Some(v) = l
            .trim_start()
            .strip_prefix("- ")
            .or_else(|| (l.trim() == "-").then_some(""))
        else {
            break;
        };
        let v = match v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
            Some(q) => q.replace("''", "'"),
            None => v.to_string(),
        };
        out.push(v);
    }
    out
}

impl PpOcr {
    /// `det_dir` and `rec_dir` hold `inference.onnx` (and `inference.yml` for the dictionary).
    pub fn load(det_dir: &Path, rec_dir: &Path, threads: usize) -> Result<Self, OcrError> {
        let env = crate::layout::ort_env().map_err(OcrError::Ort)?;
        let session = |p: &Path| -> Result<Session, OcrError> {
            Ok(Session::builder(&env)?
                .with_intra_threads(threads)?
                .commit_from_file(p)?)
        };
        let dict = parse_dict(&std::fs::read_to_string(rec_dir.join("inference.yml"))?);
        if dict.is_empty() {
            return Err(OcrError::Model(
                "no character_dict in rec inference.yml".into(),
            ));
        }
        Ok(Self {
            det: session(&det_dir.join("inference.onnx"))?,
            rec: session(&rec_dir.join("inference.onnx"))?,
            dict,
        })
    }

    /// Line boxes in `img` pixels, in reading order.
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<[u32; 4]>, OcrError> {
        let (w, h) = img.dimensions();
        let mut padded =
            RgbImage::from_pixel(w + 2 * DET_PAD, h + 2 * DET_PAD, image::Rgb([255; 3]));
        image::imageops::replace(&mut padded, img, DET_PAD.into(), DET_PAD.into());
        let (pw, ph) = padded.dimensions();
        let scale = (DET_MAX_SIDE as f32 / pw.max(ph) as f32).min(1.0);
        let round32 = |v: u32| (((v as f32 * scale) / 32.0).round() as u32).max(1) * 32;
        let (dw, dh) = (round32(pw), round32(ph));
        let small = image::imageops::resize(&padded, dw, dh, FilterType::Triangle);
        let plane = (dw * dh) as usize;
        let mut chw = vec![0f32; 3 * plane];
        // BGR order with ImageNet mean/std, as PaddleOCR feeds it
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        for (i, px) in small.pixels().enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = (f32::from(px[2 - c]) / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let out = self.det.run(
            ort::inputs!["x" => Tensor::from_array(([1usize, 3, dh as usize, dw as usize], chw))?],
        )?;
        let (_, prob) = out[0].try_extract_tensor::<f32>()?;
        let (sx, sy) = (pw as f32 / dw as f32, ph as f32 / dh as f32);
        let boxes = db_boxes(prob, dw as usize, dh as usize)
            .into_iter()
            .filter_map(|[x0, y0, x1, y1]| {
                let to = |v: f32, s: f32, max: u32| {
                    ((v * s) as i64 - DET_PAD as i64).clamp(0, max as i64) as u32
                };
                let b = [to(x0, sx, w), to(y0, sy, h), to(x1, sx, w), to(y1, sy, h)];
                (b[2] > b[0] + 2 && b[3] > b[1] + 2).then_some(b)
            })
            .collect();
        Ok(reading_order(boxes))
    }

    /// Reads one line image.
    pub fn read_line(&mut self, line: &RgbImage) -> Result<(String, Vec<f32>), OcrError> {
        let (w, h) = line.dimensions();
        let rw = ((REC_HEIGHT as f32 * w as f32 / h.max(1) as f32).ceil() as u32)
            .clamp(16, REC_MAX_WIDTH);
        let small = image::imageops::resize(line, rw, REC_HEIGHT, FilterType::Triangle);
        let plane = (rw * REC_HEIGHT) as usize;
        let mut chw = vec![0f32; 3 * plane];
        for (i, px) in small.pixels().enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = (f32::from(px[2 - c]) / 255.0 - 0.5) / 0.5;
            }
        }
        let out = self.rec.run(ort::inputs![
            "x" => Tensor::from_array(([1usize, 3, REC_HEIGHT as usize, rw as usize], chw))?
        ])?;
        let (shape, probs) = out[0].try_extract_tensor::<f32>()?;
        let classes = *shape.last().unwrap_or(&0) as usize;
        if classes != self.dict.len() + 2 {
            return Err(OcrError::Model(format!(
                "rec has {classes} classes, dictionary {} (+blank, space)",
                self.dict.len()
            )));
        }
        Ok(ctc_decode(probs, classes, &self.dict))
    }

    pub fn read(&mut self, img: &RgbImage) -> Result<Vec<Line>, OcrError> {
        let mut lines = vec![];
        for bbox in self.detect(img)? {
            let [x0, y0, x1, y1] = bbox;
            let crop = image::imageops::crop_imm(img, x0, y0, x1 - x0, y1 - y0).to_image();
            let (text, probs) = self.read_line(&crop)?;
            if !text.trim().is_empty() {
                lines.push(Line { bbox, text, probs });
            }
        }
        Ok(lines)
    }
}

/// Greedy CTC: best class per step, repeats collapsed, blanks dropped. Returns the text and each
/// character's probability.
pub fn ctc_decode(probs: &[f32], classes: usize, dict: &[String]) -> (String, Vec<f32>) {
    let (mut text, mut out, mut prev) = (String::new(), vec![], 0usize);
    for step in probs.chunks_exact(classes) {
        let (best, p) =
            step.iter().enumerate().fold(
                (0, f32::MIN),
                |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc },
            );
        if best != 0 && best != prev {
            match dict.get(best - 1) {
                Some(ch) => text.push_str(ch),
                None => text.push(' '),
            }
            out.push(p);
        }
        prev = best;
    }
    (text.trim().to_string(), out)
}

/// DB post-processing on axis-aligned boxes: threshold, 4-connected components, mean score
/// filter, then unclip each box by area × ratio / perimeter. Coordinates in map pixels.
fn db_boxes(prob: &[f32], w: usize, h: usize) -> Vec<[f32; 4]> {
    let mut seen = vec![false; w * h];
    let mut out = vec![];
    let mut stack = vec![];
    for start in 0..w * h {
        if seen[start] || prob[start] <= BIN_THRESH {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        let (mut sum, mut n) = (0f32, 0usize);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            sum += prob[i];
            n += 1;
            let mut push = |j: usize| {
                if !seen[j] && prob[j] > BIN_THRESH {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                push(i - 1);
            }
            if x + 1 < w {
                push(i + 1);
            }
            if y > 0 {
                push(i - w);
            }
            if y + 1 < h {
                push(i + w);
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        if bw.min(bh) < 3.0 || sum / (n as f32) < BOX_THRESH {
            continue;
        }
        let d = bw * bh * UNCLIP / (2.0 * (bw + bh));
        out.push([
            x0 as f32 - d,
            y0 as f32 - d,
            x1 as f32 + 1.0 + d,
            y1 as f32 + 1.0 + d,
        ]);
    }
    out
}

/// Top-to-bottom lines (boxes sharing most of their height), left to right within a line.
fn reading_order(mut boxes: Vec<[u32; 4]>) -> Vec<[u32; 4]> {
    boxes.sort_by_key(|b| b[1] + b[3]);
    let mut rows: Vec<Vec<[u32; 4]>> = vec![];
    for b in boxes {
        let same = rows.last().is_some_and(|r| {
            let (t, bt) = (r[0][1].max(b[1]), r[0][3].min(b[3]));
            let overlap = bt.saturating_sub(t) as f32;
            overlap > 0.5 * (b[3] - b[1]).min(r[0][3] - r[0][1]) as f32
        });
        match rows.last_mut() {
            Some(r) if same => r.push(b),
            _ => rows.push(vec![b]),
        }
    }
    rows.into_iter()
        .flat_map(|mut r| {
            r.sort_by_key(|b| b[0]);
            r
        })
        .collect()
}

impl Engine for PpOcr {
    fn name(&self) -> &str {
        "pp-ocrv6-small"
    }

    /// Text only; lines are joined with newlines (the router reflows paragraphs).
    fn recognize(
        &mut self,
        crop: &RgbImage,
        task: Task,
        _max_tokens: u32,
    ) -> Result<Recognition, EngineError> {
        if task != Task::Text {
            return Err(EngineError::Response(format!(
                "pp-ocr reads text, not {task:?}"
            )));
        }
        let lines = self
            .read(crop)
            .map_err(|e| EngineError::Response(e.to_string()))?;
        Ok(Recognition {
            text: lines
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            token_probs: lines
                .iter()
                .flat_map(|l| l.probs.iter().map(|&p| f64::from(p)))
                .collect(),
            engine: self.name().into(),
            truncated: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_from_yaml() {
        let yml = "PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - '!'\n  - ''''\n  - a\n  - \\\n";
        assert_eq!(parse_dict(yml), ["!", "'", "a", "\\"]);
    }

    #[test]
    fn ctc_collapses_repeats_and_blanks() {
        let dict: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        // classes: blank, a, b, space
        let p = |i: usize| {
            let mut v = vec![0.0f32; 4];
            v[i] = 0.9;
            v
        };
        let probs: Vec<f32> = [1, 1, 0, 1, 2, 3, 2].iter().flat_map(|&i| p(i)).collect();
        let (text, conf) = ctc_decode(&probs, 4, &dict);
        assert_eq!(text, "aab b");
        assert_eq!(conf.len(), 5);
    }

    #[test]
    fn boxes_and_order() {
        let (w, h) = (40, 20);
        let mut prob = vec![0f32; w * h];
        for y in 2..6 {
            for x in 20..30 {
                prob[y * w + x] = 0.9; // right word, line 1
            }
            for x in 2..12 {
                prob[y * w + x] = 0.9; // left word, line 1
            }
        }
        for y in 12..16 {
            for x in 2..30 {
                prob[y * w + x] = 0.9; // line 2
            }
        }
        let boxes: Vec<[u32; 4]> = db_boxes(&prob, w, h)
            .into_iter()
            .map(|b| b.map(|v| v.max(0.0) as u32))
            .collect();
        assert_eq!(boxes.len(), 3);
        let ordered = reading_order(boxes);
        assert!(
            ordered[0][0] < ordered[1][0] && ordered[1][1] < ordered[2][1],
            "{ordered:?}"
        );
    }

    #[test]
    #[ignore = "needs models/pp-ocrv6-small-{det,rec}; run with --ignored"]
    fn reads_printed_page() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut ocr = PpOcr::load(
            &root.join("models/pp-ocrv6-small-det"),
            &root.join("models/pp-ocrv6-small-rec"),
            4,
        )
        .unwrap();
        let img = image::open(root.join("eval/data/printed/printed2.png"))
            .unwrap()
            .to_rgb8();
        let text: Vec<String> = ocr
            .read(&img)
            .unwrap()
            .into_iter()
            .map(|l| l.text)
            .collect();
        let all = text.join(" ");
        assert!(all.contains("Heap Sort"), "{all}");
    }
}
