# Decisions and corrections to docs/research.md

Verified against primary sources on 2026-11-02.

1. **TeleOCR runtimes.** teleocr-rs GGUF is explicitly "Not for llama.cpp". The model card also lists vLLM, SGLang, Transformers, and community GGUFs for Ollama/LM Studio; re-check those in week 1.
2. **IndicOCR** ships as safetensors with a Python/CUDA installer, tested on vLLM. "In-process on ONNX Runtime" is unproven; needs a conversion spike, else sidecar/fast-mode only. It scores 92.76 vs PaddleOCR-VL-1.6's 96.36 on English OmniDocBench v1.6, so it is an Indic-only specialist.
3. **Texo weights** (alephpi/FormulaNet) are tagged AGPL-3.0, not just the web app. Default formula specialist: PP-FormulaNet_plus-S (Apache-2.0). Texo only as an optional download.
4. **PP-DocLayoutV3** has community ONNX exports (Apache-2.0, ~480 ms/page CPU on Apple M). Layout module does not need to fall back to TeleOCR's built-in layout.
5. **PaddleOCR-VL-1.6 GGUF**: an official PaddlePaddle repo exists; the mmproj file is required alongside.
6. **Tectonic** is maintained (0.17.0, MIT).
7. Still unverified: speed estimates, speculative-decoding speedup, GPU prices.
8. Code licence: Apache-2.0 (provisional, doc left MIT vs Apache open).
