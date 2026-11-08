import sys, time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
M = "/home/satyansh/makaar/models/paddleocr-vl-1.6-hf"
for trust in (False, True):
    try:
        proc = AutoProcessor.from_pretrained(M, trust_remote_code=trust)
        model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, trust_remote_code=trust).cuda().eval()
        print("loaded, trust_remote_code =", trust, type(model).__name__); break
    except Exception as e:
        print("trust", trust, "failed:", type(e).__name__, str(e)[:300])
img = Image.open(sys.argv[1]).convert("RGB")
msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "OCR:"}]}]
inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
print("prompt tokens", inp["input_ids"].shape[1], "gpu GB %.1f" % (torch.cuda.memory_allocated() / 1e9))
t = time.time()
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=int(sys.argv[2]) if len(sys.argv) > 2 else 1500, do_sample=False)
txt = proc.decode(out[0][inp["input_ids"].shape[1]:])
print("new tokens", out.shape[1] - inp["input_ids"].shape[1], "%.1fs" % (time.time() - t), "peak GB %.1f" % (torch.cuda.max_memory_allocated() / 1e9))
print(txt[:300]); print("...TAIL:", txt[-300:])
