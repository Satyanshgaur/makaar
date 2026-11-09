"""Kaggle notebook (T4, fp16): LoRA fine-tune of PaddleOCR-VL-1.6's language model on handwriting.

Free-tier research adapter: trains on non-commercial data (MathWriting train, CROHME train, GNHK),
so the adapter is CC BY-NC-SA 4.0 and must be replaced before any paid tier (DATASETS.md).
Evaluation items (CROHME 2019, MathWriting test, IAM lines) are never trained on.

Steps: cut printed replay blocks and start labelling them with the base model on the second GPU ->
sanity-check the base model on 60 eval items -> build the training mix -> train LoRA for a time budget,
mixing in labelled replay blocks -> evaluate base and adapter on all 450 eval items with makaar's
metric, and on 150 held-out printed blocks against the base model's own reading -> save adapter.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft"], check=False)

import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import peft  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

TRAIN_SECONDS = float(os.environ.get("TRAIN_SECONDS", 5 * 3600))
REPLAY_P = float(os.environ.get("REPLAY_P", 0.35))  # share of training examples that are printed replay
N_MATHWRITING = 30000
SEED = 7
OUT = "/kaggle/working"
REPO = "PaddlePaddle/PaddleOCR-VL-1.6"
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
for z in glob.glob(f"{EVAL_ROOT}/*.zip"):  # directories uploaded zipped
    zipfile.ZipFile(z).extractall(EVAL_ROOT if os.access(EVAL_ROOT, os.W_OK) else "/kaggle/working/eval")
eval_dirs = [EVAL_ROOT, "/kaggle/working/eval"]
EVAL = []
for root in eval_dirs:
    for md in sorted(glob.glob(f"{root}/**/*.md", recursive=True)):
        img = next((md[:-3] + e for e in (".png", ".jpg") if os.path.exists(md[:-3] + e)), None)
        if img:
            bucket = os.path.basename(os.path.dirname(md))
            EVAL.append((bucket, img, open(md).read()))
log("eval items", len(EVAL), {b: sum(1 for x in EVAL if x[0] == b) for b in {x[0] for x in EVAL}})


def compact_ned(pred, truth):
    a, b = "".join(clean_text(pred).split()), "".join(clean_text(truth).split())
    return normalized_edit_distance(a, b)


# ---------------------------------------------------------------- printed replay (self-distillation)
# Without it the adapter forgot printed text: always on, it stopped after the first line of a
# paragraph (DECISIONS 54). Replay printed blocks from DocLayNet (CDLA-Permissive 1.0), labelled by
# the base model itself, so the adapter learns to keep the base reading where it was already right.
DLN = "docling-project/DocLayNet-v1.2"
REPLAY_DIR, LABELS = f"{OUT}/replay", f"{OUT}/replay_labels.jsonl"
# DocLayNet classes: caption, footnote, formula, list item, section header, text, title
KEEP = {1: "OCR:", 2: "OCR:", 3: "Formula Recognition:", 4: "OCR:", 8: "OCR:", 10: "OCR:", 11: "OCR:"}
os.makedirs(REPLAY_DIR, exist_ok=True)


def dln_blocks(shard, n_pages, tag):
    f = pq.ParquetFile(hf_hub_download(DLN, shard, repo_type="dataset"))
    out, pages = [], 0
    for g in range(f.num_row_groups):
        for r in f.read_row_group(g, columns=["image", "bboxes", "category_id"]).to_pylist():
            page = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            for k, ((x, y, w, h), c) in enumerate(zip(r["bboxes"], r["category_id"])):
                if c not in KEEP or w < 40 or h < 8 or h > 400:
                    continue
                if h < 20 and random.random() > 0.4:
                    continue  # favour multi-line blocks: losing every line after the first was the fault
                crop = page.crop((max(0, x - 3), max(0, y - 3), min(page.width, x + w + 3), min(page.height, y + h + 3)))
                # DocLayNet pages are 1025 px; makaar renders about twice that
                crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
                path = f"{REPLAY_DIR}/{tag}{pages:05d}_{k:02d}.png"
                crop.save(path)
                out.append({"id": os.path.basename(path)[:-4], "path": path, "prompt": KEEP[c]})
            pages += 1
            if pages >= n_pages:
                return out
    return out


replay_val = dln_blocks("data/validation-00000-of-00007.parquet", 60, "val")
random.shuffle(replay_val)
replay_val = replay_val[:150]
replay_train = dln_blocks("data/train-00000-of-00072.parquet", 10**6, "train")
random.shuffle(replay_train)
replay_by_id = {x["id"]: x for x in replay_val + replay_train}
log("replay blocks: held out", len(replay_val), "train", len(replay_train))

LABELER = r'''
import json, sys, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
repo, items, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
proc = AutoProcessor.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.float32).cuda().eval()
with open(out, "a") as f:
    for it in items:
        img = Image.open(it["path"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": it["prompt"]}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o = model.generate(**inp, max_new_tokens=512, do_sample=False)
        new = o[0][inp["input_ids"].shape[1]:]
        f.write(json.dumps({"id": it["id"], "text": proc.decode(new, skip_special_tokens=True), "finished": len(new) < 512}) + "\n")
        f.flush()
'''
open(f"{OUT}/labeler.py", "w").write(LABELER)
json.dump(replay_val + replay_train, open(f"{OUT}/replay_items.json", "w"))
# label on the second T4 while the first trains (same fp16 greedy decoding as `read` below)
labeler = subprocess.Popen(
    [sys.executable, f"{OUT}/labeler.py", REPO, f"{OUT}/replay_items.json", LABELS],
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "1" if torch.cuda.device_count() > 1 else "0"},
    stdout=open(f"{OUT}/labeler.log", "w"), stderr=subprocess.STDOUT)
log("labeler started on", "GPU 1" if torch.cuda.device_count() > 1 else "GPU 0 (shared)")


def read_labels():
    labels = {}
    if os.path.exists(LABELS):
        for line in open(LABELS, "rb").read().split(b"\n"):
            try:
                r = json.loads(line)
            except ValueError:
                continue  # empty or half-written last line
            if r["finished"] and r["text"].strip():
                labels[r["id"]] = r["text"]
    return labels


# ---------------------------------------------------------------- model
proc = model = None
for trust in (False, True):
    try:
        proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=trust)
        model = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.float32, trust_remote_code=trust).cuda()
        log("loaded", type(model).__name__, "trust_remote_code", trust)
        break
    except Exception as e:  # noqa: BLE001
        log("load failed with trust", trust, type(e).__name__, str(e)[:200])
model.eval()


def chat(img, prompt, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    if answer is None:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    full = msgs + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    return proc.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt")


def prompt_for(bucket):
    return "OCR:" if bucket in ("iam_line", "gnhk_line", "gnhk_page", "replay_ocr") else "Formula Recognition:"


@torch.no_grad()
def read(m, img, prompt, max_new=384):
    inp = chat(img, prompt).to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate(m, items, tag):
    rows = []
    for b, path, truth in items:
        pred = read(m, Image.open(path).convert("RGB"), prompt_for(b))
        rows.append({"bucket": b, "id": os.path.basename(path), "pred": pred, "ned": compact_ned(pred, truth)})
    summary = {b: round(statistics.mean(r["ned"] for r in rows if r["bucket"] == b), 4) for b in sorted({r["bucket"] for r in rows})}
    log(tag, summary)
    json.dump(rows, open(f"{OUT}/eval_{tag}.json", "w"))
    return summary


# sanity: does the base model read like it does in llama.cpp? (expect ~0.19 / ~0.17 / ~0.05)
by_bucket = {}
for x in EVAL:
    by_bucket.setdefault(x[0], []).append(x)
quick = [x for b in sorted(by_bucket) for x in by_bucket[b][:20]]
base_quick = evaluate(model, quick, "base_quick")
if max(base_quick.values()) > 0.6:
    log("base model reads badly; stopping before training")
    sys.exit(0)

# ---------------------------------------------------------------- training mix
CMD = re.compile(r"\\[A-Za-z]+$")


def despace(latex):
    """CROHME's `x ^ { 2 }` -> `x^{2}`, keeping the space after a command before a letter."""
    out = ""
    for tok in latex.split():
        if out and CMD.search(out) and tok[0].isalpha():
            out += " "
        out += tok
    return out


def from_parquet(repo, files, image_col, label_col, n=None, fix=lambda s: s):
    rows = []
    for f in files:
        t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=[image_col, label_col])
        imgs, labs = t.column(image_col).to_pylist(), t.column(label_col).to_pylist()
        rows += [(i["bytes"], fix(l)) for i, l in zip(imgs, labs) if i and l]
    random.shuffle(rows)
    return rows[:n] if n else rows


mw = from_parquet("deepcopy/MathWriting-human", [f"data/train-0000{i}-of-00003-{h}.parquet" for i, h in
                  enumerate(["ab0ae6b9fa4a3f06", "589d2b65116e09aa", "42472859069c07bc"])], "image", "latex", N_MATHWRITING)
log("mathwriting train", len(mw))
cr = from_parquet("Neeze/CROHME-full", ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
                  "image", "label", fix=despace)
log("crohme train", len(cr))

gnhk_lines, gnhk_pages = [], []
outer = zipfile.ZipFile(hf_hub_download("staghado/GNHK-Dataset", "GNHK-dataset.zip", repo_type="dataset"))
inner = zipfile.ZipFile(io.BytesIO(outer.read("train_data.zip")))
for js in [n for n in inner.namelist() if n.endswith(".json")]:
    imgname = next((js[:-5] + e for e in (".jpg", ".png", ".jpeg") if js[:-5] + e in inner.namelist()), None)
    if not imgname:
        continue
    words = json.loads(inner.read(js))
    page = Image.open(io.BytesIO(inner.read(imgname))).convert("RGB")
    lines = {}
    for w in words:
        lines.setdefault(w["line_idx"], []).append(w)
    page_ok, page_text = True, []
    for idx in sorted(lines):
        # the annotation order within a line is not reading order: sort words left to right
        ws = sorted(lines[idx], key=lambda w: min(w["polygon"][f"x{i}"] for i in range(4)))
        text = " ".join(w["text"] for w in ws)
        if "%" in text:  # %math%, %SC%, %NA%: unreadable or untranscribed
            page_ok = False
            continue
        page_text.append(text)
        xs = [w["polygon"][f"x{i}"] for w in ws for i in range(4)]
        ys = [w["polygon"][f"y{i}"] for w in ws for i in range(4)]
        box = (max(0, min(xs) - 8), max(0, min(ys) - 8), min(page.width, max(xs) + 8), min(page.height, max(ys) + 8))
        if box[2] - box[0] > 20 and box[3] - box[1] > 10:
            buf = io.BytesIO()
            page.crop(box).save(buf, "PNG")
            gnhk_lines.append((buf.getvalue(), text))
    if page_ok and page_text:
        buf = io.BytesIO()
        page.save(buf, "JPEG", quality=90)
        gnhk_pages.append((buf.getvalue(), "\n".join(page_text)))
log("gnhk lines", len(gnhk_lines), "pages", len(gnhk_pages))

train = ([("mathwriting", b, f"\\[{l}\\]") for b, l in mw] + [("crohme", b, f"\\[{l}\\]") for b, l in cr]
         + [("gnhk_line", b, t) for b, t in gnhk_lines] * 2 + [("gnhk_page", b, t) for b, t in gnhk_pages] * 2)
random.shuffle(train)
log("training examples", len(train))

# ---------------------------------------------------------------- LoRA
TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
cfg = peft.LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, target_modules=TARGETS)
model = peft.get_peft_model(model, cfg)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
ACCUM, WARMUP = 8, 50
est_steps = 4000
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / est_steps))))

replay_uses, replay_pool = {}, []


def with_replay():
    """The handwriting mix with base-labelled printed blocks mixed in at REPLAY_P, each used at most 3 times."""
    for k, ex in enumerate(train):
        yield ex
        if k % 50 == 0:
            labels = read_labels()
            replay_pool[:] = [(i, t) for i, t in labels.items() if i.startswith("train") and replay_uses.get(i, 0) < 3]
        if replay_pool and random.random() < REPLAY_P / (1 - REPLAY_P):
            i, text = replay_pool.pop(random.randrange(len(replay_pool)))
            replay_uses[i] = replay_uses.get(i, 0) + 1
            if replay_uses[i] < 3:
                replay_pool.append((i, text))
            it = replay_by_id[i]
            src = "replay_ocr" if it["prompt"] == "OCR:" else "replay_formula"
            yield src, open(it["path"], "rb").read(), text


step, seen, losses, train_start = 0, 0, [], time.time()
for i, (src, img_bytes, answer) in enumerate(with_replay()):
    if time.time() - train_start > TRAIN_SECONDS:
        break
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    prompt = prompt_for(src)
    full, pre = chat(img, prompt, answer), chat(img, prompt)  # the template appends </s> after the answer
    n_pre = pre["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n_pre], pre["input_ids"][0]):
        continue  # template mismatch: skip rather than train on a misaligned mask
    labels = full["input_ids"].clone()
    labels[:, :n_pre] = -100
    full["labels"] = labels
    full = full.to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**full).loss / ACCUM
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        continue
    scaler.scale(loss).backward()
    losses.append(loss.item() * ACCUM)
    seen += 1
    if seen % ACCUM == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % 25 == 0:
            log(f"step {step} examples {seen} replay {sum(replay_uses.values())} loss {statistics.mean(losses[-200:]):.4f} lr {sched.get_last_lr()[0]:.2e}"
                f" {(time.time() - train_start) / seen:.2f}s/example")
        if step % 500 == 0:
            model.save_pretrained(f"{OUT}/adapter")
model.save_pretrained(f"{OUT}/adapter")
log("trained", step, "steps on", seen, "examples, of which replay", sum(replay_uses.values()))
labeler.terminate()

# ---------------------------------------------------------------- evaluation
model.eval()
tuned = evaluate(model, EVAL, "tuned")
with model.disable_adapter():
    base = evaluate(model, EVAL, "base")


def replay_eval(m, tag):
    """Forgetting check: NED of a reading against the base model's own reading of held-out printed blocks."""
    labels, rows = read_labels(), []
    for it in replay_val:
        if it["id"] not in labels:
            continue
        pred, ref = read(m, Image.open(it["path"]).convert("RGB"), it["prompt"], max_new=512), labels[it["id"]]
        rows.append({"id": it["id"], "pred": pred, "ref": ref, "ned": normalized_edit_distance(clean_text(pred), clean_text(ref)),
                     "short": len(pred) < 0.7 * len(ref)})
    s = {"blocks": len(rows), "ned": round(statistics.mean(r["ned"] for r in rows), 4) if rows else None,
         "stopped_short": sum(r["short"] for r in rows)}
    log("printed replay", tag, s)
    json.dump(rows, open(f"{OUT}/replay_eval_{tag}.json", "w"))
    return s


printed = {"tuned": replay_eval(model, "tuned")}
with model.disable_adapter():
    printed["base"] = replay_eval(model, "base")  # should be ~0: the labels are this model's greedy readings
json.dump({"base": base, "tuned": tuned, "printed_vs_base": printed, "steps": step, "examples": seen,
           "replay_examples": sum(replay_uses.values()), "replay_labelled": len(read_labels()),
           "mix": {"mathwriting": len(mw), "crohme": len(cr), "gnhk_lines": len(gnhk_lines), "gnhk_pages": len(gnhk_pages)}},
          open(f"{OUT}/summary.json", "w"), indent=1)
log("done", {"base": base, "tuned": tuned, "printed_vs_base": printed})
