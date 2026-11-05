#!/usr/bin/env bash
# Gemini Flash baseline for gate 1, through the Gemini API's OpenAI-compatible endpoint.
# Key: create one at https://aistudio.google.com/apikey and save it, readable only by you:
#   mkdir -p ~/.config/makaar && (umask 077; cat > ~/.config/makaar/gemini_key)   # paste, Enter, Ctrl-D
# Free-tier inputs may be used by Google to improve its models; don't send pages you can't share.
# Usage: scripts/gemini_baseline.sh [model] [dataset]   (model defaults to the newest stable Flash)
set -euo pipefail
KEY=${GEMINI_KEY_FILE:-$HOME/.config/makaar/gemini_key}
BASE=https://generativelanguage.googleapis.com/v1beta/openai
[ -s "$KEY" ] || { echo "No key at $KEY (see the header of this script)" >&2; exit 1; }
MODEL=${1:-$(python3 - "$KEY" "$BASE" <<'PY'
import json, re, sys, urllib.request
key = open(sys.argv[1]).read().strip()
req = urllib.request.Request(sys.argv[2] + "/models", headers={"Authorization": f"Bearer {key}"})
ids = [m["id"].removeprefix("models/") for m in json.load(urllib.request.urlopen(req))["data"]]
flash = [i for i in ids if re.fullmatch(r"gemini-[\d.]+-flash", i)]
if not flash:
    sys.exit("no stable gemini-*-flash model; pick one of: " + ", ".join(i for i in ids if "flash" in i))
print(max(flash, key=lambda i: [int(x) for x in re.findall(r"\d+", i)]))
PY
)}
DATA=${2:-data}
echo "model: $MODEL" >&2
cd "$(dirname "$0")/../eval"
uv run python -m makaar_eval.runner "$DATA" --name gemini-api --url "$BASE/chat/completions" \
  --model "$MODEL" --api-key-file "$KEY" --no-logprobs --reasoning-effort low --max-tokens 8192 \
  --prompt-file prompts/page_markdown.txt
