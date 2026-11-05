#!/usr/bin/env bash
# Installs Tectonic and prints model-download commands. Models are large; run those yourself.
set -euo pipefail
command -v tectonic >/dev/null || {
  echo "Tectonic missing. cargo install needs libpng/harfbuzz/fontconfig dev libs; easier: gh release download tectonic@0.17.0 -R tectonic-typesetting/tectonic -p "*x86_64-unknown-linux-musl.tar.gz"" >&2; exit 1; }
tectonic --version
cat <<'MSG'
Models (not downloaded automatically):
  hf download PaddlePaddle/PaddleOCR-VL-1.6-GGUF --local-dir models/paddleocr-vl-1.6   # needs mmproj too
  hf download PaddlePaddle/PP-DocLayoutV3_onnx --local-dir models/pp-doclayoutv3       # layout, ONNX
  hf download XingChen-AGI/TeleOCR --local-dir models/teleocr                         # serve with vLLM
Serve Paddle:  llama-server -m models/paddleocr-vl-1.6/<Q8_0>.gguf --mmproj models/paddleocr-vl-1.6/<mmproj>.gguf --port 8080
Layout demo:   cargo run --release --manifest-path core/Cargo.toml --example layout -- models/pp-doclayoutv3/inference.onnx eval/data/printed
Run eval:      cd eval && uv run python -m makaar_eval.runner <dataset> --name paddle-q8
MSG
