"""Find a "repetition loop" direction in the residual stream.

Teacher-forces the runtime's own readings of practice pages (text only, no image) and compares
token positions inside a loop (second and later repeats of the looping unit) with positions in
ordinary text (before the loop, and on pages that never looped). Per layer: the difference of
means as a direction, how well it separates the two (AUC), and how much each attention block and
MLP writes along it.

    python find_direction.py ~/.cache/makaar/loops/base.jsonl ~/.cache/makaar/loops/base-var.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from lm import LAYERS, LM


def loop_span(text: str, min_repeats: int = 4) -> tuple[int, int] | None:
    """(start of the looping region, start of its second repeat) by character, or None."""
    n, best = len(text), None
    for unit in range(1, min(200, n // min_repeats) + 1):
        run = 0
        while run + unit < n and text[n - 1 - run] == text[n - 1 - run - unit]:
            run += 1
        if run >= (min_repeats - 1) * unit and (best is None or run > best[0]):
            best = (run, unit)
    if best is None:
        return None
    run, unit = best
    return n - run - unit, n - run


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    s = torch.cat([pos, neg])
    r = s.argsort().argsort().float() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    rows = [json.loads(l) for f in sys.argv[1:] for l in Path(f).read_text().splitlines()]
    practice = [r for r in rows if r["split"] == "practice"]
    loops = [r for r in practice if r["loop"] and loop_span(r["text"])]
    clean = [r for r in practice if not r["loop"]][: max(2 * len(loops), 10)]
    print(f"practice readings: {len(practice)}, usable loops {len(loops)}, clean pages used {len(clean)}")
    m = LM()
    prefix = m.encode_chat("OCR:")
    in_loop = [[] for _ in range(LAYERS + 1)]
    normal = [[] for _ in range(LAYERS + 1)]
    writes = {"attn": [[[], []] for _ in range(LAYERS)], "mlp": [[[], []] for _ in range(LAYERS)]}
    for r in loops + clean:
        text = r["text"][:6000]
        enc = m.tok.encode(text, add_special_tokens=False)
        ids = prefix + enc.ids
        tr = m.run(ids, logits="none")
        offs = [o[0] for o in enc.offsets]
        span = loop_span(text) if r["loop"] else None
        for j, off in enumerate(offs):
            t = len(prefix) + j
            if span and off >= span[1] + 1:
                bucket = in_loop
            elif span is None or off < span[0]:
                bucket = normal
            else:
                continue  # first repeat: ambiguous
            for l in range(LAYERS + 1):
                bucket[l].append(tr.resid[l][t])
            for kind, outs in (("attn", tr.attn_out), ("mlp", tr.mlp_out)):
                for l in range(LAYERS):
                    writes[kind][l][0 if bucket is in_loop else 1].append(outs[l][t])
    print(f"tokens: in-loop {len(in_loop[0])}, normal {len(normal[0])}")
    dirs = []
    print("\nlayer  AUC(resid·d)  |mean diff|")
    for l in range(LAYERS + 1):
        a, b = torch.stack(in_loop[l]), torch.stack(normal[l])
        d = a.mean(0) - b.mean(0)
        dirs.append(d / d.norm())
        print(f"{l:5d}  {auc(a @ dirs[l], b @ dirs[l]):.3f}        {d.norm():.2f}")
    # which blocks write the late-layer loop direction
    ref = int(sys.argv[0] and max(range(LAYERS + 1), key=lambda l: auc(torch.stack(in_loop[l]) @ dirs[l], torch.stack(normal[l]) @ dirs[l])))
    print(f"\nwriters of the layer-{ref} direction: mean projection, in-loop minus normal")
    contrib = []
    for kind in ("attn", "mlp"):
        for l in range(LAYERS):
            a, b = torch.stack(writes[kind][l][0]), torch.stack(writes[kind][l][1])
            contrib.append(((a @ dirs[ref]).mean() - (b @ dirs[ref]).mean(), kind, l))
    for c, kind, l in sorted(contrib, reverse=True)[:10]:
        print(f"  {kind:4s} layer {l:2d}: {c:+.2f}")
    torch.save({"dirs": torch.stack(dirs), "ref": ref, "contrib": contrib}, "directions.pt")


if __name__ == "__main__":
    main()
