"""Links a fixed, stratified sample of prepared eval sets into one folder for the Gemini gate run.

The Gemini free tier allows 20 Flash requests a day, so the gate comparison uses a small fixed
sample that a daily run can finish in a few days (the runner resumes). Every system is scored on
the same folder.

    uv run python scripts/make_gate_sample.py ../datasets/gate ../datasets/eval-omni:6 ../datasets/eval-hw:6
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

IMAGE = {".png", ".jpg", ".jpeg"}


def main(argv=None):
    argv = argv or sys.argv[1:]
    out, sources = Path(argv[0]), argv[1:]
    rng = random.Random(11)
    n = 0
    for spec in sources:
        root, per = spec.rsplit(":", 1)
        for bucket in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            pages = sorted(p for p in bucket.iterdir() if p.suffix.lower() in IMAGE)
            for img in rng.sample(pages, min(int(per), len(pages))):
                dest = out / bucket.name
                dest.mkdir(parents=True, exist_ok=True)
                for f in (img, img.with_suffix(".md")):
                    link = dest / f.name
                    if not link.exists():
                        link.symlink_to(f.resolve())
                n += 1
    print(f"{n} pages → {out}")


if __name__ == "__main__":
    main()
