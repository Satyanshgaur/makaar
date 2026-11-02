"""Metrics matching OmniDocBench conventions: normalized edit distance, reading order."""
from __future__ import annotations

import re
from collections.abc import Sequence
import shutil
import subprocess
import tempfile
from pathlib import Path


def edit_distance(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(pred: str, truth: str) -> float:
    """0.0 = identical, 1.0 = completely different. Whitespace-normalized."""
    pred, truth = " ".join(pred.split()), " ".join(truth.split())
    if not pred and not truth:
        return 0.0
    return edit_distance(pred, truth) / max(len(pred), len(truth))


def reading_order_distance(pred_ids: list[str], truth_ids: list[str]) -> float:
    if not pred_ids and not truth_ids:
        return 0.0
    return edit_distance(pred_ids, truth_ids) / max(len(pred_ids), len(truth_ids))


def flag_quality(blocks: list[tuple[bool, bool]]) -> dict[str, float]:
    """blocks: (is_wrong, is_flagged). Returns recall of wrong blocks and the flag rate."""
    if not blocks:
        return {"wrong_flagged": 1.0, "flag_rate": 0.0}
    wrong = [f for w, f in blocks if w]
    return {
        "wrong_flagged": sum(wrong) / len(wrong) if wrong else 1.0,
        "flag_rate": sum(f for _, f in blocks) / len(blocks),
    }


def has_loop(text: str, min_repeat: int = 8) -> bool:
    """Detect degenerate repetition, a known failure of small OCR VLMs."""
    return re.search(r"(.{3,}?)\1{%d,}" % (min_repeat - 1), text, re.S) is not None


def latex_compiles(formula: str, tectonic: str = "tectonic") -> bool:
    """True if the formula compiles inside a minimal document. False if Tectonic is missing."""
    exe = shutil.which(tectonic)
    if exe is None:
        return False
    doc = "\\documentclass{article}\\usepackage{amsmath,amssymb}\\begin{document}\n$$" + formula + "$$\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.tex").write_text(doc)
        r = subprocess.run([exe, "-X", "compile", "f.tex"], cwd=d, capture_output=True, timeout=120)
        return r.returncode == 0
