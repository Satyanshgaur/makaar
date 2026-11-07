# Results

Scores are normalized edit distance (lower is better) after `clean_text`, which removes formatting, math delimiters and table markup. "Compact" also ignores whitespace. Measured 2026-11-07 on an i5-13420H with an RTX 3050 6 GB, PaddleOCR-VL-1.6 Q8 on llama.cpp. The datasets and how they were sampled are in DECISIONS 41.

## Printed pages: OmniDocBench, 140 English pages

| Source (20 pages each) | makaar (routed) | PaddleOCR-VL alone | Gemini 3.5 Flash-Lite* |
|---|---|---|---|
| Magazines | 0.049 | 0.045 | 0.106 |
| Newspapers | 0.086 | 0.087 | 0.034 (6 pages) |
| Exam papers | 0.099 | 0.075 | 0.138 |
| Colourful textbooks | 0.113 | 0.094 | 0.295 |
| Academic papers | 0.153 | 0.150 | 0.309 |
| Slides | 0.166 | 0.165 | 0.191 |
| Books | 0.229 | 0.214 | 0.255 |
| **All** | **0.128** | 0.118 | 0.207 (126 pages; makaar 0.130 on the same pages) |

GPU time per page, median: 5.7 s routed, 12.1 s with PaddleOCR-VL alone. Routing used the 0.98 threshold; it is now 0.99, which block-level calibration puts at about half the accuracy cost (DECISIONS 42).

## Handwriting

| Set (150 each) | makaar | Gemini 3.5 Flash-Lite* |
|---|---|---|
| IAM lines (text) | 0.061 compact, 36% exact | 0.027 |
| CROHME 2019 (math expressions) | 0.306 compact, 24% exact | 0.163 |
| MathWriting (math expressions) | 0.217 on the 68 Flash-Lite finished | 0.202 |

\*Flash-Lite is not the gate model. Gemini Flash's free tier allows 20 requests a day, so the gate comparison runs on a fixed 60-item sample over three days (`scripts/gemini_daily.sh`). Flash-Lite has a separate quota and is shown only as context.

## Speed (gate 2), CPU only, 6 threads

| Pages | Median s/page |
|---|---|
| Handwritten linear-algebra notes (HumynLabs, 20) | ~70 (gate: 60) |
| Original printed eval pages (3) | 1.5–38 |

## Reproduce

```
cd eval
uv run python scripts/prepare_omnidocbench.py ../datasets/omnidocbench ../datasets/eval-omni --per-source 20
uv run --with pyarrow --with pillow python scripts/prepare_parquet.py Neeze/CROHME-full data/2019-00000-of-00001.parquet label ../datasets/eval-hw crohme2019 --n 150 --math
cargo run --release --manifest-path ../core/Cargo.toml --example pipeline -- ../models http://127.0.0.1:8091 ../datasets/eval-omni preds/routed-omni
uv run python -m makaar_eval.runner ../datasets/eval-omni --name routed-omni --predictions preds/routed-omni
PYTHONPATH=scripts uv run python scripts/calibrate_blocks.py ../datasets/omnidocbench/OmniDocBench.json preds/vl-omni preds/route-omni/specialist
```
