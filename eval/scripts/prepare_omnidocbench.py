"""Builds an eval set (<out>/<source>/<page>.png + .md) from OmniDocBench's English pages.

OmniDocBench (opendatalab, research use only, not commercial) annotates every block of 1,651 PDF
pages. Ground truth here is each page's text in the annotators' reading order: headers first,
then ordered blocks, then footers. Display math stays `$$…$$`, tables stay HTML (clean_text
scores tables on cell text in any format), and figures and masked regions are left out.

    uv run python scripts/prepare_omnidocbench.py ../datasets/omnidocbench ../datasets/eval-omni --per-source 25
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

SOURCES = ["book", "exam_paper", "colorful_textbook", "academic_literature", "PPT2PDF", "magazine", "newspaper"]
TOP = {"header"}
BOTTOM = {"footer", "page_number", "page_footnote"}
SKIP = {"figure", "abandon", "text_mask", "chart_mask", "unknown_mask", "need_mask", "table_mask",
        "algorithm_mask", "list_group"}


def block_text(a: dict) -> str:
    cat = a["category_type"]
    if cat == "equation_isolated":
        latex = (a.get("latex") or "").strip()
        return latex if latex.startswith("$$") else f"$$\n{latex}\n$$"
    if cat == "table":
        return a.get("html") or a.get("latex") or ""
    if cat == "title":
        return "## " + (a.get("text") or "").strip()
    return (a.get("text") or "").strip()


def page_truth(page: dict) -> str:
    dets = [a for a in page["layout_dets"] if a["category_type"] not in SKIP and not a.get("ignore")]
    top = [a for a in dets if a["category_type"] in TOP]
    bottom = [a for a in dets if a["category_type"] in BOTTOM]
    body = [a for a in dets if a not in top and a not in bottom]
    body.sort(key=lambda a: (a.get("order") is None, a.get("order") or 0))
    parts = [block_text(a) for a in top + body + bottom]
    return "\n\n".join(p for p in parts if p.strip()) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="folder with OmniDocBench.json (images downloaded on demand)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-source", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    pages = json.loads((a.src / "OmniDocBench.json").read_text())
    rng = random.Random(a.seed)
    chosen = []
    for src in SOURCES:
        pool = [p for p in pages if p["page_info"]["page_attribute"].get("language") == "english"
                and p["page_info"]["page_attribute"].get("data_source") == src]
        chosen += rng.sample(pool, min(a.per_source, len(pool)))
    names = [p["page_info"]["image_path"] for p in chosen]
    missing = [n for n in names if not (a.src / "images" / n).exists()]
    if missing:
        subprocess.run(["hf", "download", "opendatalab/OmniDocBench", "--repo-type", "dataset",
                        "--local-dir", str(a.src), *[f"images/{n}" for n in missing]], check=True)
    for p in chosen:
        info = p["page_info"]
        bucket = a.out / info["page_attribute"]["data_source"]
        bucket.mkdir(parents=True, exist_ok=True)
        stem = Path(info["image_path"]).stem
        img = a.src / "images" / info["image_path"]
        shutil.copy(img, bucket / (stem + img.suffix.lower()))
        (bucket / (stem + ".md")).write_text(page_truth(p))
    print(f"{len(chosen)} pages → {a.out}")


if __name__ == "__main__":
    main()
