# makaar

Open-source, offline notes-to-LaTeX/Markdown/Word converter. See `docs/research.md` for the research and build plan, and `docs/DECISIONS.md` for where we diverge from it.

Status: Phase 2–3 (ingest, layout, per-region recognition, Markdown, compile-checked LaTeX and Word export in `core/`). Gate 1 is still open: only 10 eval pages so far.

## Layout
- `schema/document.schema.json` – the job document (pages → blocks), single source of truth for review UI and exports
- `eval/` – evaluation harness (metrics, engine adapters, runner)
- `core/` – Rust core: document model, resumable queue, ingest, layout (PP-DocLayoutV3 on ONNX Runtime)

## Dev
    cd eval && uv run pytest
