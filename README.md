# makaar

Open-source, offline notes-to-LaTeX/Markdown/Word converter. See `docs/research.md` for the research and build plan, and `docs/DECISIONS.md` for where we diverge from it.

Status: Phase 3. The `makaar` CLI converts page images end to end, offline: layout, per-region recognition, then Markdown, compile-checked LaTeX and Word. Gate 1 is still open: only 10 eval pages so far.

## Try it

```
cargo build --release --manifest-path core/Cargo.toml   # or a build from the release workflow
core/target/release/makaar setup                     # models (~1.9 GB) + llama.cpp's llama-server, SHA-256 checked
core/target/release/makaar convert scans/ -o out/    # out/notes.{md,tex,pdf,docx,json}
core/target/release/makaar review out/notes.json      # fix flagged blocks in the browser; saving re-exports
```

`setup` installs llama.cpp's official CPU build for your platform (`--backend vulkan` uses most laptop GPUs); an existing `llama-server` on PATH or `$MAKAAR_LLAMA_SERVER` also works. `tectonic` is optional and enables the LaTeX compile check. Use `--gpu-layers 0` for CPU only. Rerun the same command to resume an interrupted job.

## Layout
- `schema/document.schema.json` – the job document (pages → blocks), single source of truth for review UI and exports
- `eval/` – evaluation harness (metrics, engine adapters, runner)
- `core/` – Rust core and the `makaar` CLI: document model, resumable queue, ingest, layout (PP-DocLayoutV3 on ONNX Runtime), recognition via a llama-server sidecar, exports, model download

## Dev
    cd eval && uv run pytest
