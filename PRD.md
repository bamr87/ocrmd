# PRD — ocrmd

> What ocrmd is for and whom it serves: the problem, users, goals and non-goals, current features, requirements, proposed success metrics and open questions.

Product requirements for [`bamr87/ocrmd`](https://github.com/bamr87/ocrmd), derived from the README, agent docs, config and code in this repo. Anything not stated in those sources is marked **TBD**; success metrics are proposals.

## 1. Overview / problem

ocrmd turns physical books and loose pages into cross-checked Markdown, plain text and JSON from a webcam, a folder of photos or a PDF, and keeps an inventory of everything scanned. Every page is read twice (local OCR engines and a vision model, Claude by default) and the readings are diffed, flagging disagreements rather than silently picking one.

## 2. Target users

- People digitizing books and documents (e.g. study material) on a Mac
- Developers embedding the pipeline via `ocrmd.api.Library`
- Downstream systems receiving pages via the destination HTTP contract (e.g. law-ai)

## 3. Goals and non-goals

**Goals**

- Accurate page transcriptions with visible disagreement markers
- A searchable SQLite inventory across all scanned pages
- Work offline (OCR-only) with the LLM pass deferrable
- Only page images leave the machine, and only to the configured provider

**Non-goals**

- Cloud hosting / multi-user service: TBD (not described)
- Silent auto-correction of disagreements

## 4. Core features (current)

- CLI: scan, ingest (incl. `--watch`), verify (incl. Batches API), review, status, reprocess, delete, build, read, library, search, export, send, etl, ocr-bench, doctor
- Local OCR: Apple Vision + RapidOCR voting (Tesseract fallback)
- Vision-model providers: Anthropic (default) and xAI
- Corpus per document (`documents/<slug>/pages/NNNN.md`) with rich front matter
- Exports: md, txt, json, jsonl; local web UI
- Structured-data ETL pipelines and destinations

## 5. Requirements

**Functional**

- Each page stored as Markdown with front matter and both readings reconciled
- `ocrmd doctor` reports engines, cameras, credentials and DB health
- Config via `ocrmd.toml` / user config / `OCRMD_*` env / CLI flags (in increasing precedence)

**Non-functional**

- Python ≥ 3.12; macOS primary (Linux/Windows via RapidOCR)
- Test suite (~306 checks)
- Cost awareness: `doctor --costs`, batch mode at reduced price

## 6. Success metrics (proposed — not defined in sources)

- OCR vs LLM disagreement rate per page
- Pages requiring re-shoot/review
- Cost per page
- Targets: TBD

## 7. Roadmap / open questions

- Supported non-macOS camera capture: TBD
- xAI provider not exercised live (per README)

## 8. Links

- [README.md](README.md) · [AGENTS.md](AGENTS.md) · [CHANGELOG.md](CHANGELOG.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Backlog of record: [GitHub Issues](https://github.com/bamr87/ocrmd/issues)
