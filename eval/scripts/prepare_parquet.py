"""Samples an image + label dataset from the Hugging Face Hub into eval format
(<out>/<bucket>/<id>.png + .md). Labels are wrapped as display math with --math.

    uv run --with pyarrow python scripts/prepare_parquet.py Neeze/CROHME-full data/2019-00000-of-00001.parquet label ../datasets/eval-hw crohme2019 --n 150 --math
    uv run --with pyarrow python scripts/prepare_parquet.py Teklia/IAM-line data/test.parquet text ../datasets/eval-hw iam_line --n 150

Licences differ: CROHME and IAM are for research use, MathWriting is CC BY-NC-SA. Use these for
evaluation only; never train a shipped model on them.
"""
from __future__ import annotations

import argparse
import io
import random
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("file", help="parquet path inside the dataset repo")
    ap.add_argument("label", help="label column")
    ap.add_argument("out", type=Path)
    ap.add_argument("bucket")
    ap.add_argument("--image", default="image", help="image column (struct with bytes)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--math", action="store_true", help="wrap labels in $$…$$")
    ap.add_argument("--cache", type=Path, default=Path("../datasets/hub"))
    a = ap.parse_args(argv)
    local = a.cache / a.repo.replace("/", "__")
    if not (local / a.file).exists():
        subprocess.run(["hf", "download", a.repo, a.file, "--repo-type", "dataset", "--local-dir", str(local)],
                       check=True)
    table = pq.read_table(local / a.file, columns=[a.image, a.label])
    rows = list(range(table.num_rows))
    random.Random(a.seed).shuffle(rows)
    dest = a.out / a.bucket
    dest.mkdir(parents=True, exist_ok=True)
    images, labels = table.column(a.image), table.column(a.label)
    for k, i in enumerate(sorted(rows[: a.n])):
        img = images[i].as_py()
        data = img["bytes"] if isinstance(img, dict) else img
        label = labels[i].as_py().strip()
        stem = f"{a.bucket}_{i:05d}"
        from PIL import Image, ImageOps, ImageStat

        im = Image.open(io.BytesIO(data)).convert("RGB")
        if ImageStat.Stat(im.convert("L")).mean[0] < 128:  # CROHME renders white ink on black
            im = ImageOps.invert(im)
        im.save(dest / f"{stem}.png")
        (dest / f"{stem}.md").write_text(f"$$\n{label}\n$$\n" if a.math else label + "\n")
    print(f"{min(a.n, len(rows))} samples → {dest}")


if __name__ == "__main__":
    main()
