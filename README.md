# makaar

Open-source, offline notes-to-LaTeX/Markdown/Word converter. See `docs/research.md` for the research and build plan, `docs/DECISIONS.md` for where we diverge from it, and `docs/RESULTS.md` for measured accuracy and speed.

Status: Phase 3. The `makaar` CLI converts page images end to end, offline: layout, per-region recognition, then Markdown, compile-checked LaTeX and Word. Gate 1 is still open: only 10 eval pages so far.

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
