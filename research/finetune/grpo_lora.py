"""Kaggle script (2x T4, fp16): GRPO on handwritten maths, starting from the run-3 adapter.

Run 3 (train_lora.py) learned handwriting by imitation. This step samples several readings of each
handwritten formula from the adapter itself and pushes it towards the readings that score better on
makaar's own metric (1 - compact NED against the truth), with a KL penalty that keeps it close to
run 3. The prompts are MathWriting's validation split, which run 3 never saw, plus CROHME train.
Evaluation items (CROHME 2019, MathWriting test, IAM lines) are never trained on.

GPU 0 holds the policy (run 3, trainable); GPU 1 holds the frozen reference (run 3 merged, fp16)
for the KL term. The adapter stays CC BY-NC-SA 4.0 (MathWriting; DATASETS.md).

Steps: load policy and reference -> sanity-check sampling -> GRPO for TRAIN_SECONDS -> score the
policy and the reference on the 450 eval items, and on 150 printed DocLayNet blocks against the
reference's own reading (forgetting check) -> save adapter/.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft"], check=False)

import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import peft  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

TRAIN_SECONDS = float(os.environ.get("TRAIN_SECONDS", 4 * 3600))
G = int(os.environ.get("GROUP", 6))  # samples per prompt
ITEMS_PER_STEP = 4
LR, BETA, TEMP, MAX_NEW = 2e-5, 0.04, 1.0, 200
SEED = 11
OUT = "/kaggle/working"
REPO = "PaddlePaddle/PaddleOCR-VL-1.6"
ADAPTER_REPO, ADAPTER_REV = "satyansh0/makaar-handwriting-lora", "b36979f5ae7a8cb1955dd18213d7b248321258eb"
P, R = "cuda:0", "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
random.seed(SEED)
torch.manual_seed(SEED)
t0 = time.time()


def log(*a):
    print(f"[{(time.time() - t0) / 60:6.1f} min]", *a, flush=True)


# ---------------------------------------------------------------- eval items + metric
metrics_py = glob.glob("/kaggle/input/**/makaar_metrics.py", recursive=True)[0]
sys.path.insert(0, os.path.dirname(metrics_py))
from makaar_metrics import clean_text, normalized_edit_distance  # noqa: E402

EVAL_ROOT = os.path.dirname(metrics_py)
for z in glob.glob(f"{EVAL_ROOT}/*.zip"):
    zipfile.ZipFile(z).extractall(EVAL_ROOT if os.access(EVAL_ROOT, os.W_OK) else "/kaggle/working/eval")
EVAL = []
for root in [EVAL_ROOT, "/kaggle/working/eval"]:
    for md in sorted(glob.glob(f"{root}/**/*.md", recursive=True)):
        img = next((md[:-3] + e for e in (".png", ".jpg") if os.path.exists(md[:-3] + e)), None)
        if img:
            EVAL.append((os.path.basename(os.path.dirname(md)), img, open(md).read()))
log("eval items", len(EVAL))


def compact_ned(pred, truth):
    a, b = "".join(clean_text(pred).split()), "".join(clean_text(truth).split())
    return normalized_edit_distance(a, b)


# ---------------------------------------------------------------- models
proc = AutoProcessor.from_pretrained(REPO)
peft_dir = os.path.join(snapshot_download(ADAPTER_REPO, revision=ADAPTER_REV, allow_patterns=["peft/*"]), "peft")
policy = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.float32)
policy = peft.PeftModel.from_pretrained(policy, peft_dir, is_trainable=True).to(P)
policy.eval()  # no dropout: samples and gradients come from the same policy
ref = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.float32)
ref = peft.PeftModel.from_pretrained(ref, peft_dir).merge_and_unload().half().to(R).eval()
for p in ref.parameters():
    p.requires_grad_(False)
eos = policy.generation_config.eos_token_id
EOS = set(eos if isinstance(eos, list) else [eos])
log("policy on", P, "reference on", R)


def chat(img, prompt):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")


def prompt_for(bucket):
    return "OCR:" if bucket in ("iam_line", "ocr") else "Formula Recognition:"


@torch.no_grad()
def read(m, dev, img, prompt, max_new=384):
    inp = chat(img, prompt).to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate(m, dev, items, tag):
    rows = []
    for b, path, truth in items:
        pred = read(m, dev, Image.open(path).convert("RGB"), prompt_for(b))
        rows.append({"bucket": b, "id": os.path.basename(path), "pred": pred, "ned": compact_ned(pred, truth)})
    summary = {b: round(statistics.mean(r["ned"] for r in rows if r["bucket"] == b), 4) for b in sorted({r["bucket"] for r in rows})}
    log(tag, summary)
    json.dump(rows, open(f"{OUT}/eval_{tag}.json", "w"))
    return summary


# ---------------------------------------------------------------- prompts
CMD = re.compile(r"\\[A-Za-z]+$")


def despace(latex):
    """CROHME's `x ^ { 2 }` -> `x^{2}`, keeping the space after a command before a letter."""
    out = ""
    for tok in latex.split():
        if out and CMD.search(out) and tok[0].isalpha():
            out += " "
        out += tok
    return out


def from_parquet(repo, files, image_col, label_col, fix=lambda s: s):
    rows = []
    for f in files:
        t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=[image_col, label_col])
        rows += [(i["bytes"], fix(l)) for i, l in zip(t.column(image_col).to_pylist(), t.column(label_col).to_pylist()) if i and l]
    return rows


mw = from_parquet("deepcopy/MathWriting-human", ["data/val-00000-of-00001-184984e66f80ed7a.parquet"], "image", "latex")
cr = from_parquet("Neeze/CROHME-full", ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
                  "image", "label", fix=despace)
random.shuffle(cr)
prompts = [(b, f"\\[{l}\\]") for b, l in mw] + [(b, f"\\[{l}\\]") for b, l in cr[: len(mw) // 4]]
random.shuffle(prompts)
log("prompts: mathwriting val", len(mw), "crohme train", len(mw) // 4)


# ---------------------------------------------------------------- sampling and log-probs
def trim(seq):
    """Tokens up to and including the first end-of-sequence token."""
    out = []
    for t in seq.tolist():
        out.append(t)
        if t in EOS:
            break
    return out


@torch.no_grad()
def sample(inp, n, batched):
    n_pre = inp["input_ids"].shape[1]
    kw = dict(max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=1.0, top_k=0)
    with torch.autocast("cuda", dtype=torch.float16):
        if batched:
            outs = policy.generate(**inp, num_return_sequences=n, **kw)
            return [trim(o[n_pre:]) for o in outs]
        return [trim(policy.generate(**inp, **kw)[0][n_pre:]) for _ in range(n)]


def token_logps(m, inp, completion, dev):
    """Log-probabilities of `completion` given the prompt in `inp`, one per completion token."""
    ids = torch.cat([inp["input_ids"], torch.tensor([completion], device=inp["input_ids"].device)], 1).to(dev)
    n_new, extra = len(completion), {}
    for k, v in inp.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if v.dim() == 2 and v.shape == inp["input_ids"].shape:  # per-token, e.g. mm_token_type_ids: text
            v = torch.cat([v, v.new_zeros(1, n_new)], 1)
        extra[k] = v.to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = m(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits
    n_pre = inp["input_ids"].shape[1]
    lp = torch.log_softmax(logits[0, n_pre - 1:-1].float(), -1)
    return lp.gather(1, ids[0, n_pre:, None]).squeeze(1)


def reward(completion, truth):
    return 1.0 - compact_ned(proc.decode(completion, skip_special_tokens=True), truth)


# sanity: does num_return_sequences expand the image correctly? Compare against one-at-a-time sampling.
BATCHED = True
check = prompts[:8]
rb, rl = [], []
try:
    for b, truth in check:
        inp = chat(Image.open(io.BytesIO(b)).convert("RGB"), "Formula Recognition:").to(P)
        rb += [reward(c, truth) for c in sample(inp, 4, True)]
        rl += [reward(c, truth) for c in sample(inp, 4, False)]
    log(f"sampling check: batched reward {statistics.mean(rb):.3f}, one at a time {statistics.mean(rl):.3f}")
    BATCHED = statistics.mean(rb) > statistics.mean(rl) - 0.15
except Exception as e:  # noqa: BLE001
    log("batched sampling failed:", type(e).__name__, str(e)[:200])
    BATCHED = False
log("batched sampling", BATCHED)

by_bucket = {}
for x in EVAL:
    by_bucket.setdefault(x[0], []).append(x)
quick = [x for b in sorted(by_bucket) for x in by_bucket[b][:20]]
evaluate(policy, P, quick, "start_quick")

# ---------------------------------------------------------------- GRPO
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
stats = {"items": 0, "skipped_flat": 0, "steps": 0}
hist = []  # per item: mean reward, KL
step_items, train_start = 0, time.time()
for b, truth in prompts:
    if time.time() - train_start > TRAIN_SECONDS:
        break
    inp = chat(Image.open(io.BytesIO(b)).convert("RGB"), "Formula Recognition:").to(P)
    comps = [c for c in sample(inp, G, BATCHED) if c]
    rs = [reward(c, truth) for c in comps]
    stats["items"] += 1
    if len(rs) < 2 or statistics.pstdev(rs) < 1e-3:
        stats["skipped_flat"] += 1  # every sample scored the same: no signal
        hist.append((statistics.mean(rs) if rs else 0.0, None))
        continue
    mu, sd = statistics.mean(rs), statistics.pstdev(rs)
    kls = []
    for c, r in zip(comps, rs):
        adv = (r - mu) / (sd + 1e-4)
        lp = token_logps(policy, inp, c, P)
        with torch.no_grad():
            ref_lp = token_logps(ref, inp, c, R).to(P)
        d = ref_lp - lp
        kl = (torch.exp(d) - d - 1).mean()  # k3 estimator, per token
        loss = (-adv * lp.mean() + BETA * kl) / (len(comps) * ITEMS_PER_STEP)
        scaler.scale(loss).backward()
        kls.append(kl.item())
    hist.append((mu, statistics.mean(kls)))
    step_items += 1
    if step_items == ITEMS_PER_STEP:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step_items = 0
        stats["steps"] += 1
        s = stats["steps"]
        if s % 25 == 0:
            recent = hist[-200:]
            kl_recent = [k for _, k in recent if k is not None]
            log(f"step {s} items {stats['items']} flat {stats['skipped_flat']} reward {statistics.mean(r for r, _ in recent):.4f}"
                f" kl {statistics.mean(kl_recent) if kl_recent else 0:.4f} {(time.time() - train_start) / stats['items']:.2f}s/item")
        if s % 250 == 0:
            policy.save_pretrained(f"{OUT}/adapter")
            evaluate(policy, P, quick, f"quick_{s}")
policy.save_pretrained(f"{OUT}/adapter")
log("trained", stats)

# ---------------------------------------------------------------- evaluation
tuned = evaluate(policy, P, EVAL, "grpo")
start = evaluate(ref, R, EVAL, "run3")

# forgetting check on printed blocks: the policy's reading against run 3's own reading
DLN = "docling-project/DocLayNet-v1.2"
KEEP = {1: "OCR:", 2: "OCR:", 3: "Formula Recognition:", 4: "OCR:", 8: "OCR:", 10: "OCR:", 11: "OCR:"}
f = pq.ParquetFile(hf_hub_download(DLN, "data/validation-00000-of-00007.parquet", repo_type="dataset"))
blocks = []
for r in f.read_row_group(0, columns=["image", "bboxes", "category_id"]).to_pylist()[:60]:
    page = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
    for (x, y, w, h), c in zip(r["bboxes"], r["category_id"]):
        if c in KEEP and w >= 40 and 20 <= h <= 400:
            crop = page.crop((max(0, x - 3), max(0, y - 3), min(page.width, x + w + 3), min(page.height, y + h + 3)))
            blocks.append((crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS), KEEP[c]))
random.shuffle(blocks)
rows = []
for img, prompt in blocks[:150]:
    pred, base_read = read(policy, P, img, prompt, 512), read(ref, R, img, prompt, 512)
    rows.append({"pred": pred, "ref": base_read, "ned": normalized_edit_distance(clean_text(pred), clean_text(base_read)),
                 "short": len(pred) < 0.7 * len(base_read)})
printed = {"blocks": len(rows), "ned": round(statistics.mean(r["ned"] for r in rows), 4), "stopped_short": sum(r["short"] for r in rows)}
json.dump(rows, open(f"{OUT}/printed_vs_run3.json", "w"))
log("printed vs run 3", printed)

json.dump({"grpo": tuned, "run3": start, "printed_vs_run3": printed, **stats, "batched_sampling": BATCHED,
           "group": G, "lr": LR, "beta": BETA, "temperature": TEMP,
           "reward_curve": [round(statistics.mean(r for r, _ in hist[i:i + 200]), 4) for i in range(0, len(hist), 200)]},
          open(f"{OUT}/summary.json", "w"), indent=1)
log("done", {"grpo": tuned, "run3": start, "printed_vs_run3": printed})
