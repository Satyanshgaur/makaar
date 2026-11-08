# Weight-space editing experiments

Reading and editing PaddleOCR-VL-1.6's weights directly, without training (DECISIONS 52).

- `lm.py`: the language model (ERNIE-4.5-0.3B) written from scratch over `model.safetensors`, so no repository code runs. Text only. Matches llama.cpp token for token on greedy decoding.
- `find_direction.py`: difference-of-means "loop" direction per layer from teacher-forced runtime readings, and which blocks write it.
- `find_heads.py`: induction-style copying per head, inside loops vs in normal text.
- `make_edit.py`: writes an edited BF16 GGUF by projecting a direction out of chosen output matrices.
- `load_check.py`: one-page generation with transformers (needs the repository's own code, `trust_remote_code`).

Setup (big files stay out of /tmp, which is RAM on the dev machine):

    uv venv -p 3.12 ~/.cache/makaar/wenv
    VIRTUAL_ENV=~/.cache/makaar/wenv uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    VIRTUAL_ENV=~/.cache/makaar/wenv uv pip install "transformers>=5" safetensors tokenizers numpy pillow
    hf download PaddlePaddle/PaddleOCR-VL-1.6 --local-dir models/paddleocr-vl-1.6-hf

Loop rates come from `eval/scripts/loop_rate.py` against llama-server serving each GGUF; pages are rendered with `cargo run --release --example render_pages`.
