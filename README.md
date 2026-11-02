# makaar

Open-source, offline notes-to-LaTeX/Markdown/Word converter. See `docs/research.md` for the research and build plan, and `docs/DECISIONS.md` for where we diverge from it.

Status: Phase 0 (schema, engine interface, eval harness). Gates 1–2 decide whether product code follows.

## Layout
- `schema/document.schema.json` – the job document (pages → blocks), single source of truth for review UI and exports
- `eval/` – evaluation harness (metrics, engine adapters, runner)

## Dev
    cd eval && uv run pytest
