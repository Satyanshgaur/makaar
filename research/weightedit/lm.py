"""PaddleOCR-VL-1.6's language model (ERNIE-4.5-0.3B), written from scratch over the raw
safetensors so no downloaded code runs. Text only: without image tokens the model's 3-D RoPE
reduces to ordinary neox-style RoPE, since all three position components are equal.

Exposes what the weight-editing work needs: the residual stream after every layer, and what each
attention block and MLP wrote into it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

MODEL = Path("/home/satyansh/makaar/models/paddleocr-vl-1.6-hf")
LAYERS, HEADS, KV_HEADS, HEAD_DIM, EPS, THETA = 18, 16, 2, 128, 1e-5, 500000.0


@dataclass
class Trace:
    resid: list[torch.Tensor]  # input to each layer, then the final pre-norm stream: LAYERS + 1 x [T, d]
    attn_out: list[torch.Tensor]  # what each attention block added: LAYERS x [T, d]
    mlp_out: list[torch.Tensor]  # what each MLP added: LAYERS x [T, d]
    logits: torch.Tensor  # [T, vocab]


class LM:
    def __init__(self, weights: dict[str, torch.Tensor] | None = None, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        w = weights if weights is not None else load_file(MODEL / "model.safetensors")
        self.w = {k: v.to(device=device, dtype=dtype) for k, v in w.items()
                  if not k.startswith(("visual", "mlp_AR"))}
        self.device, self.dtype = device, dtype
        self.tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        inv = 1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
        self.inv_freq = inv

    def encode_chat(self, prompt: str, answer: str = "") -> list[int]:
        """The chat template without the image: `<cls>User: {prompt}\\nAssistant:\\n{answer}`."""
        return self.tok.encode(f"<|begin_of_sentence|>User: {prompt}\nAssistant:\n{answer}", add_special_tokens=False).ids

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2:]
        return (x.float() * cos + torch.cat((-x2, x1), -1).float() * sin).to(x.dtype)

    @staticmethod
    def _norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype)

    @torch.no_grad()
    def run(self, ids: list[int], keep: bool = True, hooks: dict | None = None, logits: str = "last") -> Trace:
        """Forward pass. `hooks[(layer, "attn"|"mlp")]` may replace a block's output (for ablations).
        `logits`: "all" positions, only the "last", or "none"."""
        w, T = self.w, len(ids)
        x = w["model.embed_tokens.weight"][torch.tensor(ids, device=self.device)]
        pos = torch.arange(T, device=self.device, dtype=torch.float32)
        f = torch.outer(pos, self.inv_freq)
        emb = torch.cat((f, f), -1)
        cos, sin = emb.cos()[None], emb.sin()[None]
        resid, attn_out, mlp_out = [], [], []
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            if keep:
                resid.append(x.float().cpu())
            h = self._norm(x, w[p + "input_layernorm.weight"])
            q = (h @ w[p + "self_attn.q_proj.weight"].T).view(T, HEADS, HEAD_DIM).transpose(0, 1)
            k = (h @ w[p + "self_attn.k_proj.weight"].T).view(T, KV_HEADS, HEAD_DIM).transpose(0, 1)
            v = (h @ w[p + "self_attn.v_proj.weight"].T).view(T, KV_HEADS, HEAD_DIM).transpose(0, 1)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            o = torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True,
                                                                 enable_gqa=True)[0]
            a = o.transpose(0, 1).reshape(T, HEADS * HEAD_DIM) @ w[p + "self_attn.o_proj.weight"].T
            if hooks and (i, "attn") in hooks:
                a = hooks[(i, "attn")](a)
            x = x + a
            h = self._norm(x, w[p + "post_attention_layernorm.weight"])
            g = torch.nn.functional.silu(h @ w[p + "mlp.gate_proj.weight"].T) * (h @ w[p + "mlp.up_proj.weight"].T)
            m = g @ w[p + "mlp.down_proj.weight"].T
            if hooks and (i, "mlp") in hooks:
                m = hooks[(i, "mlp")](m)
            x = x + m
            if keep:
                attn_out.append(a.float().cpu())
                mlp_out.append(m.float().cpu())
        if keep:
            resid.append(x.float().cpu())
        sel = {"all": x, "last": x[-1:], "none": x[:0]}[logits]
        out = self._norm(sel, w["model.norm.weight"]) @ w["lm_head.weight"].T
        return Trace(resid, attn_out, mlp_out, out.float())
