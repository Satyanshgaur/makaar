"""Run an engine over a dataset and write per-page scores.

Dataset layout: <root>/<bucket>/<page>.{png,jpg} with ground truth <page>.md beside it.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from .engine import Engine, Region
from .metrics import has_loop, latex_compiles, normalized_edit_distance
from .openai_engine import PADDLE_PROMPTS, OpenAIEngine

IMG_EXT = {".png", ".jpg", ".jpeg"}
DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)


def pages(root: Path):
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT):
        truth = img.with_suffix(".md")
        if truth.exists():
            yield img.relative_to(root).parent.name, img, truth


def score_page(engine: Engine, img: Path, truth: Path, check_latex: bool) -> dict:
    t0 = time.perf_counter()
    rec = engine.recognize(Region(img.read_bytes(), (0, 0, 0, 0), "page"), "text")
    secs = time.perf_counter() - t0
    out = {"page": str(img), "seconds": secs, "ned": normalized_edit_distance(rec.text, truth.read_text()),
           "confidence": rec.confidence, "loop": has_loop(rec.text), "text": rec.text}
    if check_latex:
        f = DISPLAY_MATH.findall(rec.text)
        out["formulas"] = len(f)
        out["formulas_compiled"] = sum(latex_compiles(x) for x in f)
    return out


def summarize(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["bucket"], []).append(r)
    out = {}
    for b, rs in by.items():
        secs = sorted(r["seconds"] for r in rs)
        out[b] = {"pages": len(rs), "ned_mean": statistics.mean(r["ned"] for r in rs),
                  "sec_median": statistics.median(secs),
                  "sec_p90": secs[min(len(secs) - 1, int(0.9 * len(secs)))],
                  "loops": sum(r["loop"] for r in rs)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--name", required=True, help="label for this system, e.g. paddle-q8")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="default")
    ap.add_argument("--prompt", help="override the text prompt (TeleOCR)")
    ap.add_argument("--latex", action="store_true", help="also check formula compile rate")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)

    prompts = dict(PADDLE_PROMPTS)
    if a.prompt:
        prompts["text"] = a.prompt
    engine = OpenAIEngine(a.name, a.url, a.model, prompts)
    rows = []
    for bucket, img, truth in pages(a.dataset):
        r = score_page(engine, img, truth, a.latex)
        rows.append({"bucket": bucket, **r})
        print(f"{bucket}/{img.name}: ned={r['ned']:.3f} {r['seconds']:.1f}s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (a.out / f"{a.name}.json").write_text(json.dumps({"summary": summary, "pages": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
