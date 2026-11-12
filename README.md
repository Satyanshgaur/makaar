# makaar

Turn scanned or photographed notes, handwritten or printed, into Markdown, LaTeX and Word, entirely on your own computer. Nothing is uploaded. Maths comes out as real equations: LaTeX you can compile, equations you can edit in Word. Blocks the model is unsure of are flagged for a quick review.

**Download:** [latest release](https://github.com/Satyanshgaur/makaar/releases/latest). Get the desktop app for Windows, macOS or Linux; the first run downloads the reading models (about 1.9 GB, checked).

**How good is it?** On 126 printed OmniDocBench pages, makaar's error (clean NED, lower is better) is 0.130, against 0.207 for Gemini 3.5 Flash-Lite. On handwritten maths (CROHME, 30 items) it is level with Gemini 3.5 Flash; on handwritten prose Gemini is still clearly better. Details are in [`docs/RESULTS.md`](docs/RESULTS.md).

**How fast?** About a minute a page on a recent laptop CPU, and about 2 minutes on an older dual-core one. A GPU is much faster.

The installers are not code-signed yet. On Windows, click *More info → Run anyway*. On macOS, right-click the app and choose *Open*, or allow it under *System Settings → Privacy & Security*.

For the research and build plan see `docs/research.md`; `docs/DECISIONS.md` records where we diverged from it.

## Try it

```
cargo build --release --manifest-path core/Cargo.toml   # or a build from the release workflow
core/target/release/makaar setup                     # models (~1.9 GB) + llama.cpp's llama-server, SHA-256 checked
core/target/release/makaar convert scans/ -o out/    # out/notes.{md,tex,pdf,docx,json}
core/target/release/makaar review out/notes.json      # fix flagged blocks in the browser; saving re-exports
```

`setup` installs llama.cpp's official CPU build for your platform (`--backend vulkan` uses most laptop GPUs); an existing `llama-server` on PATH or `$MAKAAR_LLAMA_SERVER` also works. `tectonic` is optional and enables the LaTeX compile check. Use `--gpu-layers 0` for CPU only. Rerun the same command to resume an interrupted job.

`setup --handwriting-adapter` also fetches the optional handwriting adapter ([satyansh0/makaar-handwriting-lora](https://huggingface.co/satyansh0/makaar-handwriting-lora), 48 MB). It reads handwritten maths better, but it is CC BY-NC-SA 4.0 (free, non-commercial use only; see `DATASETS.md`), and on a CPU it roughly doubles the time spent on handwritten blocks. For handwritten notes, `convert --handwritten` reads each block once with the adapter: better handwritten maths at normal speed, but printed maths can come out restyled.

Beta testers: see `docs/BETA.md`. `convert` ends with a `speed:` line (median s/page, CPU, RAM, mode) to paste into a report.

## Desktop app

`desktop/` is a Tauri window over the same core: pick pages (images or PDFs), convert, review. Build installers with `cd desktop && cargo tauri build` (Linux needs `webkit2gtk4.1-devel`), or take them from the release workflow.

## Layout
- `schema/document.schema.json` – the job document (pages → blocks), single source of truth for review UI and exports
- `eval/` – evaluation harness (metrics, engine adapters, runner)
- `desktop/` – Tauri desktop app (home screen + review window)
- `core/` – Rust core and the `makaar` CLI: document model, resumable queue, ingest, layout (PP-DocLayoutV3 on ONNX Runtime), recognition via a llama-server sidecar, exports, model download

## Dev
    cd eval && uv run pytest

## License

makaar's code is Apache 2.0 (`LICENSE`). The reading models it downloads, PaddleOCR-VL 1.6, PP-DocLayoutV3 and PP-OCRv6, are Apache 2.0. The optional handwriting add-on ([satyansh0/makaar-handwriting-lora](https://huggingface.co/satyansh0/makaar-handwriting-lora)) is CC BY-NC-SA 4.0, for non-commercial use only, because of its training data (see `DATASETS.md`).
