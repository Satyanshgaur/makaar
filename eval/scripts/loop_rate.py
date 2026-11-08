"""Loop rate on whole handwritten pages: how often greedy decoding runs into a repetition loop.

Each page is read whole with the text prompt (as the pipeline does for dense handwriting). A
reading loops if it hits the token limit or ends in one unit repeated 4+ times over 200+
characters (the engine's runaway rule). Pages are split by name hash into practice and held-out
halves, so edits can be chosen on one and scored on the other.

    uv run python scripts/loop_rate.py <pages-dir> <out.jsonl> --url http://127.0.0.1:8091
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from makaar_eval.openai_engine import chat_url, mime


def runaway(text: str, min_repeats: int = 4, min_chars: int = 200) -> bool:
    n = len(text)
    for unit in range(1, min(200, n // min_repeats) + 1):
        run = 0
        while run + unit < n and text[n - 1 - run] == text[n - 1 - run - unit]:
            run += 1
        if run + unit >= max(min_chars, min_repeats * unit):
            return True
    return False


def split(name: str) -> str:
    return "practice" if int(hashlib.sha1(name.encode()).hexdigest(), 16) % 2 else "heldout"


def read(url: str, image: bytes, max_tokens: int, model: str) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime(image)};base64,{base64.b64encode(image).decode()}"}},
        {"type": "text", "text": "OCR:"}]}]}
    req = urllib.request.Request(chat_url(url), json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    c = out["choices"][0]
    text = c["message"]["content"] or ""
    return {"text": text, "finish": c.get("finish_reason"), "tokens": out.get("usage", {}).get("completion_tokens"),
            "seconds": time.time() - t, "loop": c.get("finish_reason") == "length" or runaway(text)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--model", default="default")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["practice", "heldout"])
    a = ap.parse_args(argv)
    done = {}
    if a.out.exists():
        done = {r["page"]: r for r in map(json.loads, a.out.read_text().splitlines())}
    pages = sorted(p for p in a.pages.rglob("*.png"))
    pages = [p for p in pages if not a.only or split(p.stem) == a.only]
    todo = [p for p in pages if str(p.relative_to(a.pages)) not in done]

    def work(p: Path):
        return {"page": str(p.relative_to(a.pages)), "split": split(p.stem), **read(a.url, p.read_bytes(), a.max_tokens, a.model)}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as f, ThreadPoolExecutor(a.workers) as pool:
        for r in pool.map(work, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["page"]] = r
    rows = [done[str(p.relative_to(a.pages))] for p in pages]
    for s in ("practice", "heldout"):
        rs = [r for r in rows if r["split"] == s]
        if rs:
            print(f"{s}: {len(rs)} pages, loops {sum(r['loop'] for r in rs)} ({100 * statistics.mean(r['loop'] for r in rs):.0f}%),"
                  f" median tokens {statistics.median(r['tokens'] or 0 for r in rs):.0f}")


if __name__ == "__main__":
    main()
