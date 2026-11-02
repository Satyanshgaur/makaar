#!/usr/bin/env bash
# Installs Tectonic and prints model-download commands. Models are large; run those yourself.
set -euo pipefail
command -v tectonic >/dev/null || {
  echo "Tectonic missing. Fedora: sudo dnf install tectonic  |  or: cargo install tectonic" >&2; exit 1; }
tectonic --version
cat <<'MSG'
Models (not downloaded automatically):
  hf download PaddlePaddle/PaddleOCR-VL-1.6-GGUF --local-dir models/paddleocr-vl-1.6   # needs mmproj too
  hf download XingChen-AGI/TeleOCR --local-dir models/teleocr                          # serve with vLLM
Serve Paddle:  llama-server -m models/paddleocr-vl-1.6/<Q8_0>.gguf --mmproj models/paddleocr-vl-1.6/<mmproj>.gguf --port 8080
Run eval:      cd eval && uv run python -m makaar_eval.runner <dataset> --name paddle-q8
MSG
