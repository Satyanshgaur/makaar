"""Engine adapter for any OpenAI-compatible vision server (llama-server, vLLM, SGLang, Gemini API)."""
from __future__ import annotations

import base64
import json
import re
import math
import time
import urllib.error
import urllib.request

from .engine import Recognition, Region

# PaddleOCR-VL task prompts; TeleOCR's prompt is configurable per run.
PADDLE_PROMPTS = {
    "text": "OCR:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
    "handwriting": "OCR:",
}

RETRY_STATUS = {429, 500, 502, 503, 504}


def chat_url(base_url: str) -> str:
    """`http://host:8080` → `.../v1/chat/completions`; a full `.../chat/completions` URL is kept."""
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else base_url + "/v1/chat/completions"


def mime(image: bytes) -> str:
    return "image/jpeg" if image[:3] == b"\xff\xd8\xff" else "image/png"


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up; retrying today won't help."""


class OpenAIEngine:
    def __init__(self, name: str, base_url: str, model: str = "default",
                 prompts: dict[str, str] | None = None, max_tokens: int = 4096,
                 timeout: float = 600.0, api_key: str | None = None, logprobs: bool = True,
                 extra: dict | None = None, retries: int = 6):
        self.name = name
        self.url = chat_url(base_url)
        self.model = model
        self.prompts = prompts or PADDLE_PROMPTS
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key
        self.logprobs = logprobs
        self.extra = extra or {}
        self.retries = retries

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                detail = e.read()[:2000]
                if e.code == 429 and re.search(rb"per ?day|PerDay", detail, re.I):
                    raise QuotaExhausted(f"daily quota exhausted at {self.url}") from None
                if e.code not in RETRY_STATUS or attempt == self.retries:
                    raise RuntimeError(f"{e.code} from {self.url}: {detail[:500]!r}") from None
                wait = float(e.headers.get("Retry-After") or 0) or min(60.0, 5.0 * 2 ** attempt)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def recognize(self, region: Region, task: str, draft: str | None = None) -> Recognition:
        b64 = base64.b64encode(region.image).decode()
        body = {
            "model": self.model,
            "temperature": 0,  # greedy: faithfulness gate forbids sampling
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime(region.image)};base64,{b64}"}},
                {"type": "text", "text": self.prompts[task]},
            ]}],
            **self.extra,
        }
        if self.logprobs:
            body["logprobs"] = True
        choice = self._post(body)["choices"][0]
        lp = (choice.get("logprobs") or {}).get("content") or []
        return Recognition(text=choice["message"]["content"] or "",
                           token_probs=[math.exp(t["logprob"]) for t in lp],
                           engine=self.name)
