"""Write an edited GGUF: remove the loop direction from what chosen blocks write into the residual
stream, W_out <- W_out - alpha * d (d^T W_out), for attention output and MLP down projections.

    python make_edit.py base.gguf out.gguf --layers attn:7,attn:8,mlp:8 --alpha 1.0 [--dir-layer 9]
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/satyansh/tools/llama.cpp/gguf-py")
import gguf  # noqa: E402

NAMES = {"attn": "blk.{}.attn_output.weight", "mlp": "blk.{}.ffn_down.weight"}


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)  # round to nearest even


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--layers", required=True, help="comma list of attn:N / mlp:N")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--dir-layer", type=int, help="direction to remove (default: the best-separating layer)")
    ap.add_argument("--directions", default="directions.pt")
    a = ap.parse_args(argv)
    saved = torch.load(a.directions)
    d = saved["dirs"][a.dir_layer if a.dir_layer is not None else saved["ref"]].numpy().astype(np.float32)
    d /= np.linalg.norm(d)
    shutil.copyfile(a.base, a.out)
    r = gguf.GGUFReader(a.out, "r+")
    tensors = {t.name: t for t in r.tensors}
    changed = 0
    for spec in a.layers.split(","):
        kind, layer = spec.split(":")
        t = tensors[NAMES[kind].format(int(layer))]
        assert t.tensor_type == gguf.GGMLQuantizationType.BF16, t.tensor_type
        w = bf16_to_f32(np.asarray(t.data).view(np.uint16))  # the reader exposes BF16 as raw bytes
        # GGUF data is [rows = output dim, cols = input dim] like torch's weight
        assert w.shape[0] == d.shape[0], (t.name, w.shape)
        w2 = w - a.alpha * np.outer(d, d @ w)
        t.data[...] = f32_to_bf16(w2).view(np.uint8)
        changed += w.size
        print(f"{t.name}: removed direction, |dW|/|W| = {np.linalg.norm(w2 - w) / np.linalg.norm(w):.4f}")
    r.data.flush()
    print(f"edited {changed:,} weights in {len(a.layers.split(','))} tensors -> {a.out}")


if __name__ == "__main__":
    main()
