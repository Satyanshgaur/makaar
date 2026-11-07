"""Block-level calibration of the router's two thresholds on OmniDocBench.

Each of our blocks gets its own ground truth: the OmniDocBench annotations whose box centres fall
inside it, in the annotators' order. Then, over the blocks:

* specialist acceptance: for each threshold, the block NED of "specialist if its confidence ≥ t,
  else the general model" against the general model alone, and how many blocks skip it;
* review flags: for each threshold, the precision and recall of flagging general-model blocks
  whose mean token probability is below it, where a block is bad if its clean NED > --bad.

    uv run python scripts/calibrate_blocks.py ../datasets/omnidocbench/OmniDocBench.json \\
        preds/vl-omni preds/route-omni/specialist
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from makaar_eval.metrics import clean_text, normalized_edit_distance
from prepare_omnidocbench import SKIP, block_text


def box(a: dict) -> tuple[float, float, float, float]:
    xs, ys = a["poly"][0::2], a["poly"][1::2]
    return min(xs), min(ys), max(xs), max(ys)


def truth_for(block: dict, dets: list[dict], pad: float = 6.0) -> str | None:
    x0, y0, x1, y1 = block["bbox"]
    inside = []
    for a in dets:
        bx0, by0, bx1, by1 = box(a)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
            inside.append(a)
    if not inside:
        return None
    inside.sort(key=lambda a: (a.get("order") is None, a.get("order") or 0))
    return "\n\n".join(block_text(a).removeprefix("## ") for a in inside)


def ned(pred: str, truth: str) -> float:
    return normalized_edit_distance(clean_text(pred), clean_text(truth))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("omni", type=Path, help="OmniDocBench.json")
    ap.add_argument("general", type=Path, help="page JSONs from the pipeline example run without the specialist")
    ap.add_argument("specialist", type=Path, help="per-page {block id: [text, confidence]} from route_sim")
    ap.add_argument("--bad", type=float, default=0.2, help="block clean NED above which a block needs review")
    a = ap.parse_args(argv)

    gt = {Path(p["page_info"]["image_path"]).stem: p for p in json.loads(a.omni.read_text())}
    rows = []  # (general conf, general ned, specialist conf or None, specialist ned or None)
    for page_json in sorted(a.general.rglob("*.json")):
        if page_json.name == "timings.json":
            continue
        page = json.loads(page_json.read_text())
        g = gt.get(page_json.stem)
        if g is None:
            continue
        dets = [d for d in g["layout_dets"] if d["category_type"] not in SKIP and not d.get("ignore")]
        spec_file = a.specialist / page_json.relative_to(a.general)
        spec = json.loads(spec_file.read_text()) if spec_file.exists() else {}
        for b in page["blocks"]:
            if b.get("text") is None or b.get("type") == "figure":
                continue
            t = truth_for(b, dets)
            if not t or not clean_text(t):
                continue
            s = spec.get(b["id"])
            rows.append((b.get("confidence") or 0.0, ned(b["text"], t),
                         s[1] if s else None, ned(s[0], t) if s else None))

    eligible = [r for r in rows if r[2] is not None]
    print(f"{len(rows)} blocks with ground truth; {len(eligible)} eligible for the specialist")
    print("\nspecialist acceptance (eligible blocks): mean block NED, routed vs general only")
    general_only = statistics.mean(r[1] for r in eligible)
    for t in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]:
        chosen = [r[3] if r[2] >= t else r[1] for r in eligible]
        took = sum(r[2] >= t for r in eligible)
        print(f"  t={t:<5} routed={statistics.mean(chosen):.4f} general={general_only:.4f}"
              f" specialist takes {took}/{len(eligible)} ({100 * took / len(eligible):.0f}%)")

    print(f"\nreview flags (all general-model blocks, bad = clean NED > {a.bad})")
    bad = sum(r[1] > a.bad for r in rows)
    print(f"  {bad}/{len(rows)} blocks are bad")
    for th in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
        flagged = [r for r in rows if r[0] < th]
        hit = sum(r[1] > a.bad for r in flagged)
        prec = hit / len(flagged) if flagged else 0.0
        rec = hit / bad if bad else 0.0
        print(f"  flag below {th:<4}: flags {len(flagged):4d} ({100 * len(flagged) / len(rows):4.1f}%)"
              f"  precision {prec:.2f}  recall {rec:.2f}")


if __name__ == "__main__":
    main()
