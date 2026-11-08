"""How much handwriting ability is latent in the model: sample many readings per item and compare
the greedy answer with the best reading among the samples (an upper bound no selector can beat)
and with a consensus pick that needs no ground truth (the reading closest to all the others).

If the best sample is much better than greedy, the model can read the item but doesn't by
default: weight edits, better decoding or selection can recover it. If not, only training can.

    uv run python scripts/latent_ability.py ../datasets/eval-hw out/latent --url http://127.0.0.1:8091 -k 6

Readings are cached in <out>/readings.jsonl, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from makaar_eval.metrics import clean_text, normalized_edit_distance
from makaar_eval.openai_engine import PADDLE_PROMPTS, chat_url, mime

IMG_EXT = {".png", ".jpg", ".jpeg"}


def compact(s: str) -> str:
    return "".join(clean_text(s).split())


def ned(pred: str, truth: str) -> float:
    return normalized_edit_distance(compact(pred), compact(truth))


def read(url: str, image: bytes, prompt: str, temperature: float, seed: int, max_tokens: int) -> dict:
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime(image)};base64,{base64.b64encode(image).decode()}"}},
            {"type": "text", "text": prompt}]}],
        "temperature": temperature, "top_p": 0.95, "seed": seed, "max_tokens": max_tokens,
        "logprobs": True,
    }
    req = urllib.request.Request(chat_url(url), json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)["choices"][0]
    lps = [t["logprob"] for t in (out.get("logprobs") or {}).get("content") or []]
    return {"text": out["message"]["content"] or "", "mean_logprob": statistics.mean(lps) if lps else None,
            "truncated": out.get("finish_reason") == "length"}


def items(root: Path):
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT):
        truth = img.with_suffix(".md")
        if truth.exists():
            yield img.parent.name, img, truth.read_text()


def consensus(texts: list[str]) -> int:
    """Index of the reading with the lowest mean distance to the others (a medoid)."""
    if len(texts) == 1:
        return 0
    c = [compact(t) for t in texts]
    return min(range(len(c)), key=lambda i: sum(normalized_edit_distance(c[i], c[j]) for j in range(len(c)) if j != i))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("-k", type=int, default=6, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "readings.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [(b, img, t) for b, img, t in items(a.dataset) if f"{b}/{img.stem}" not in done]
    print(f"{len(done)} cached, {len(todo)} to read", flush=True)

    def work(entry):
        bucket, img, truth = entry
        image = img.read_bytes()
        readings = []
        for task in ("text", "formula"):
            prompt = PADDLE_PROMPTS[task]
            readings.append({"task": task, "greedy": True, **read(a.url, image, prompt, 0.0, 0, a.max_tokens)})
            for s in range(a.k):
                readings.append({"task": task, "greedy": False,
                                 **read(a.url, image, prompt, a.temperature, 1 + s, a.max_tokens)})
        return {"id": f"{bucket}/{img.stem}", "bucket": bucket, "truth": truth, "readings": readings}

    with cache.open("a") as f, ThreadPoolExecutor(a.workers) as pool:
        for n, r in enumerate(pool.map(work, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["id"]] = r
            if n % 25 == 0:
                print(f"{n}/{len(todo)}", flush=True)

    report(list(done.values()), a.out)


def report(rows: list[dict], out: Path):
    by: dict[str, list] = {}
    for r in rows:
        rs, truth = r["readings"], r["truth"]
        d = [ned(x["text"], truth) for x in rs]
        g_text = next(i for i, x in enumerate(rs) if x["greedy"] and x["task"] == "text")
        g_form = next(i for i, x in enumerate(rs) if x["greedy"] and x["task"] == "formula")
        text_only = [i for i, x in enumerate(rs) if x["task"] == "text"]
        form_only = [i for i, x in enumerate(rs) if x["task"] == "formula"]
        # the task a well-routed pipeline would use: formula for maths sets, text for text lines
        task_idx = form_only if r["bucket"] != "iam_line" else text_only
        greedy_task = g_form if r["bucket"] != "iam_line" else g_text
        by.setdefault(r["bucket"], []).append({
            "greedy_task": d[greedy_task],
            "best_greedy_prompt": min(d[g_text], d[g_form]),
            "best_k_task": min(d[i] for i in task_idx),
            "best_all": min(d),
            "consensus_task": d[task_idx[consensus([rs[i]["text"] for i in task_idx])]],
            "consensus_all": d[consensus([x["text"] for x in rs])],
            "exact_greedy": d[greedy_task] == 0,
            "exact_any": min(d) == 0,
        })
    keys = ["greedy_task", "best_greedy_prompt", "consensus_task", "consensus_all", "best_k_task", "best_all"]
    summary = {}
    print(f"\n{'bucket':<14}{'n':>4} " + " ".join(f"{k:>18}" for k in keys) + f"{'exact greedy':>14}{'exact any':>11}")
    for b, v in sorted(by.items()):
        summary[b] = {k: statistics.mean(x[k] for x in v) for k in keys}
        summary[b]["exact_greedy"] = statistics.mean(x["exact_greedy"] for x in v)
        summary[b]["exact_any"] = statistics.mean(x["exact_any"] for x in v)
        summary[b]["n"] = len(v)
        print(f"{b:<14}{len(v):>4} " + " ".join(f"{summary[b][k]:>18.3f}" for k in keys)
              + f"{summary[b]['exact_greedy']:>14.2f}{summary[b]['exact_any']:>11.2f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
