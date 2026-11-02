"""Engine adapter for any OpenAI-compatible vision server (llama-server, vLLM, SGLang)."""
from __future__ import annotations

import base64
import json
import math
import urllib.request

from .engine import Recognition, Region

# PaddleOCR-VL task prompts; TeleOCR's prompt is configurable per run.
PADDLE_PROMPTS = {
    "text": "OCR:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
    "handwriting": "OCR:",
}


class OpenAIEngine:
    def __init__(self, name: str, base_url: str, model: str = "default",
                 prompts: dict[str, str] | None = None, max_tokens: int = 4096,
                 timeout: float = 600.0):
        self.name = name
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.prompts = prompts or PADDLE_PROMPTS
        self.max_tokens = max_tokens
        self.timeout = timeout

    def recognize(self, region: Region, task: str, draft: str | None = None) -> Recognition:
        b64 = base64.b64encode(region.image).decode()
        body = {
            "model": self.model,
            "temperature": 0,  # greedy: faithfulness gate forbids sampling
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": self.prompts[task]},
            ]}],
        }
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            choice = json.load(r)["choices"][0]
        lp = (choice.get("logprobs") or {}).get("content") or []
        return Recognition(text=choice["message"]["content"],
                           token_probs=[math.exp(t["logprob"]) for t in lp],
                           engine=self.name)
