"""Apply the research doc's pass/fail gates to eval results.

Usage: python -m makaar_eval.gates results/<system>.json [--baseline results/gemini.json] [--gates ../gates.json]
Exit 0 on pass, 1 on fail. Buckets without ground truth are never silently passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V1_BUCKETS = {"printed", "photo", "exam", "slides", "math"}  # handwriting/hindi are not v1 claims


def check(res: dict, gates: dict, baseline: dict | None) -> list[str]:
    fails = []
    summ = res["summary"]
    for b, s in summ.items():
        if b not in V1_BUCKETS:
            continue
        if s["sec_median"] > gates["median_seconds_per_page_max"]:
            fails.append(f"{b}: median {s['sec_median']:.1f}s > {gates['median_seconds_per_page_max']}s")
        if s["loops"] > gates["loops_max"]:
            fails.append(f"{b}: {s['loops']} looping pages")
        if baseline and b in baseline["summary"]:
            margin = s["ned_mean"] - baseline["summary"][b]["ned_mean"]
            if margin > gates["accuracy_margin_vs_baseline"]:
                fails.append(f"{b}: NED {s['ned_mean']:.3f} is {margin:.3f} worse than baseline")
    formulas = sum(p.get("formulas", 0) for p in res["pages"])
    if formulas:
        rate = sum(p.get("formulas_compiled", 0) for p in res["pages"]) / formulas
        if rate < gates["formula_compile_rate_min"]:
            fails.append(f"formula compile rate {rate:.2f} < {gates['formula_compile_rate_min']}")
    if not any(b in V1_BUCKETS for b in summ):
        fails.append("no v1 buckets scored")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--gates", type=Path, default=Path(__file__).resolve().parents[3] / "gates.json")
    a = ap.parse_args(argv)
    fails = check(json.loads(a.results.read_text()), json.loads(a.gates.read_text()),
                  json.loads(a.baseline.read_text()) if a.baseline else None)
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"{len(fails)} gate(s) failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
