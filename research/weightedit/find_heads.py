"""Which attention heads drive repetition loops: inside a loop of period P tokens, an induction
head at position t attends to t - P + 1 (the token after the previous occurrence of the current
one) and copies it. For each head: the mean attention on t - P + 1 inside loops, and as a control
the same head's mean attention on "the token after the previous occurrence of the current token"
in ordinary (non-looping) practice text, where copying one's own earlier output is rarely right.

    python find_heads.py ~/.cache/makaar/loops/base.jsonl ~/.cache/makaar/loops/base-var.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from find_direction import loop_span
from lm import HEAD_DIM, HEADS, KV_HEADS, LAYERS, LM


def token_period(ids: list[int], start: int) -> int | None:
    """Smallest p such that ids[t] == ids[t - p] for all t in the tail from `start`."""
    tail = ids[start:]
    for p in range(1, len(tail) // 3):
        if all(tail[i] == tail[i - p] for i in range(p, len(tail))):
            return p
    return None


@torch.no_grad()
def head_attention(m: LM, ids: list[int], queries: list[int]) -> torch.Tensor:
    """[LAYERS, HEADS, len(queries), T] attention probabilities for the chosen query rows."""
    hooks, out = {}, []
    w, T = m.w, len(ids)
    x = w["model.embed_tokens.weight"][torch.tensor(ids, device=m.device)]
    pos = torch.arange(T, device=m.device, dtype=torch.float32)
    f = torch.outer(pos, m.inv_freq)
    emb = torch.cat((f, f), -1)
    cos, sin = emb.cos()[None], emb.sin()[None]
    qi = torch.tensor(queries, device=m.device)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        h = m._norm(x, w[p + "input_layernorm.weight"])
        q = (h @ w[p + "self_attn.q_proj.weight"].T).view(T, HEADS, HEAD_DIM).transpose(0, 1)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).view(T, KV_HEADS, HEAD_DIM).transpose(0, 1)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).view(T, KV_HEADS, HEAD_DIM).transpose(0, 1)
        q, k = m._rope(q, cos, sin), m._rope(k, cos, sin)
        kk = k.repeat_interleave(HEADS // KV_HEADS, 0)
        s = (q[:, qi].float() @ kk.float().transpose(1, 2)) * HEAD_DIM ** -0.5
        s = s.masked_fill(torch.arange(T, device=m.device)[None, None, :] > qi[None, :, None], float("-inf"))
        out.append(torch.softmax(s, -1).cpu())
        o = torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True, enable_gqa=True)[0]
        x = x + o.transpose(0, 1).reshape(T, HEADS * HEAD_DIM) @ w[p + "self_attn.o_proj.weight"].T
        h = m._norm(x, w[p + "post_attention_layernorm.weight"])
        x = x + (torch.nn.functional.silu(h @ w[p + "mlp.gate_proj.weight"].T) * (h @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
    return torch.stack(out)


def main():
    rows = [json.loads(l) for f in sys.argv[1:] for l in Path(f).read_text().splitlines()]
    practice = [r for r in rows if r["split"] == "practice"]
    loops = [r for r in practice if r["loop"] and loop_span(r["text"])]
    clean = [r for r in practice if not r["loop"]][:20]
    m = LM()
    prefix = m.encode_chat("OCR:")
    loop_score = torch.zeros(LAYERS, HEADS)
    n_loop = 0
    for r in loops:
        enc = m.tok.encode(r["text"][:6000], add_special_tokens=False)
        ids = prefix + enc.ids
        span = loop_span(r["text"][:6000])
        start = len(prefix) + next(j for j, o in enumerate(enc.offsets) if o[0] >= span[0])
        P = token_period(ids, start + 5)
        if not P or P < 2:
            continue
        qs = list(range(max(start + 2 * P, len(ids) - 200), len(ids)))[:200]
        att = head_attention(m, ids, qs)
        tgt = torch.tensor([q - P + 1 for q in qs])
        loop_score += att[:, :, torch.arange(len(qs)), tgt].mean(-1)
        n_loop += 1
    loop_score /= max(n_loop, 1)
    ctrl = torch.zeros(LAYERS, HEADS)
    n_ctrl = 0
    for r in clean:
        ids = prefix + m.tok.encode(r["text"][:6000], add_special_tokens=False).ids
        last = {}
        qs, tg = [], []
        for t in range(len(prefix), len(ids)):
            if ids[t] in last and t - last[ids[t]] > 2:
                qs.append(t)
                tg.append(last[ids[t]] + 1)
            last[ids[t]] = t
        if not qs:
            continue
        qs, tg = qs[-200:], tg[-200:]
        att = head_attention(m, ids, qs)
        ctrl += att[:, :, torch.arange(len(qs)), torch.tensor(tg)].mean(-1)
        n_ctrl += 1
    ctrl /= max(n_ctrl, 1)
    print(f"loops analysed {n_loop}, clean pages {n_ctrl}")
    print("head        in-loop copy  clean copy   ratio")
    order = sorted(((loop_score[l, h].item(), l, h) for l in range(LAYERS) for h in range(HEADS)), reverse=True)
    for s, l, h in order[:15]:
        c = ctrl[l, h].item()
        print(f"L{l:2d} H{h:2d}     {s:.3f}        {c:.3f}      {s / max(c, 1e-3):.1f}")
    torch.save({"loop": loop_score, "clean": ctrl}, "heads.pt")


if __name__ == "__main__":
    main()
