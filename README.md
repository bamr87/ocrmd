# ocrmd

> Scan books and pages into cross-checked Markdown: local OCR diffed against a Claude vision read, with full-text search.

[![CI](https://github.com/bamr87/ocrmd/actions/workflows/ci.yml/badge.svg)](https://github.com/bamr87/ocrmd/actions/workflows/ci.yml) [![License](https://img.shields.io/github/license/bamr87/ocrmd)](LICENSE)

## Overview

Turn physical books and loose pages into cross-checked Markdown, plain text and JSON — from a webcam pointed at the book, a folder of photographs, or a PDF — and keep an inventory of everything you have scanned.

Every page is read **twice**: once by local OCR engines (Apple Vision + RapidOCR voting), once by a vision model (Claude by default). The two readings are diffed into a Markdown page file with rich frontmatter; where they disagree the page says so (`==[word]==[^ocrmd-n]`) rather than quietly picking one, and digits are never auto-resolved. The cross-check is the product: an early version fabricated a table of contents for a page it could not read, and only the diff caught it.

What you get, per root directory:

- **A corpus** — `documents/<slug>/pages/0001.md` … one file per page, and a built `<slug>.md` with a TOC; `document.yaml` for what only you know about the book (title, author, ISBN, tags).
- **An inventory** — every document, its catalogue entry, its progress, in SQLite (`ocrmd library`) and **full-text search across every page ever scanned** (`ocrmd search`).
- **Exports** — the same pages as Markdown, plain text, one JSON document or JSON-lines (`ocrmd export`).
- **Destinations** — hand a page's reading to another system that speaks a small HTTP contract (`ocrmd send`); [law-ai](https://github.com/bamr87/law-ai) is the worked example.
- **A library** — `from ocrmd.api import Library` embeds all of it in another program.

Built and tuned on a fixed camera over a textbook. Page images are sent to the provider you configure; nothing else leaves the machine.

## Features

- Reads every page twice: local OCR (Apple Vision and RapidOCR voting) and a vision model (Claude by default, xAI as an alternate)
- Diffs the two readings and flags disagreements instead of silently picking one; refuses to invent text it cannot read
- Ingests from a webcam (an auto-shutter that waits for a page turn to finish), a folder of photos or a PDF
- A SQLite inventory with full-text search across every scanned page
- Exports to Markdown, plain text and JSON, plus a local web UI for scanning, review and editing
- Structured-data extraction (`ocrmd etl`) and hand-off to configured destinations (`ocrmd send`)
- Rebuilds pages from their Markdown and scores them against the photograph (`ocrmd reconstruct`); folds pen and highlighter marks back into the Markdown (`ocrmd annotate`)
- Works offline: without credentials it runs OCR-only, and `ocrmd verify` fills in the model pass later

## Quick start

macOS is the primary platform (Apple Vision and the camera controls are macOS-only); RapidOCR is the engine elsewhere.

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'   # add ,pdf for PDF ingest; ,tesseract as a last resort
.venv/bin/ocrmd doctor --costs                    # engines, cameras, credentials, database health, token spend
.venv/bin/ruff check ocrmd tests                  # lint (the CI gate)
.venv/bin/python -m pytest                        # the full suite; tests that need a local OCR engine skip without one
.venv/bin/ocrmd ingest examples/captures/*.jpg --doc example --no-llm   # the bundled sample captures, local OCR only
.venv/bin/ocrmd serve                             # the web UI at http://127.0.0.1:8765
```

Put `ANTHROPIC_API_KEY` (or a Claude Code `CLAUDE_CODE_OAUTH_TOKEN`, which only reaches `claude-haiku-5-5`) in `.env`, and on macOS allow your terminal to use the camera. [docs/setup.md](docs/setup.md) has the details: the credentials and the models each reaches, camera permission, and the local OCR engines.

## Usage

```sh
ocrmd scan --doc lsat --camera c525 --focus settle --title "LSAT Prep" --tag study
ocrmd scan --doc lsat --camera 0 --no-llm      # offline: capture + OCR only; `verify` later
ocrmd scan --doc lsat --duration 600 --max-pages 40 --no-preview   # unattended, bounded run
ocrmd ingest ./photos/ --doc receipts-aug      # image files or PDFs instead of a webcam
ocrmd ingest ./inbox/ --doc lsat --watch       # keep ingesting whatever lands in the folder
ocrmd verify --doc lsat                        # run missing stages (e.g. the LLM pass)
ocrmd verify --doc lsat --batch                # same, through the Batches API at half price
ocrmd verify --refresh                         # re-run pages made by an older prompt or model (costs tokens)
ocrmd review --doc lsat                        # walk the disagreements, sign pages off
ocrmd status --doc lsat                        # what needs a re-shoot, a review, an LLM pass — and why
ocrmd reprocess --doc lsat                     # re-derive pages after upgrading the preprocessing
ocrmd delete --doc lsat --page 12 --page 13    # remove pages (+ their raw capture if nothing else uses it)
ocrmd delete --doc live-test --all             # remove a whole document (asks first; -y skips)
ocrmd build                                    # regenerate document .md + TOC + nav links
ocrmd read --doc lsat --page 12 --region 0.1,0.42,0.6,0.05   # re-read one region of a page, harder
ocrmd library                                  # the inventory: every document, catalogue entry, progress
ocrmd search "ridership will rise"             # full-text search across every page ever scanned
ocrmd export --doc lsat --format jsonl         # md | txt | json | jsonl -> export/lsat/lsat.jsonl
ocrmd send --doc lsat --page 15 --to law-ai -o hint="question 15"   # hand a page to a destination
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --dry-run   # extract structured data
ocrmd ocr-bench --doc lsat --compare           # score every installed OCR engine (and pair) against the LLM transcriptions
ocrmd doctor --costs
ocrmd --debug verify                           # full tracebacks and debug logs
```

Every command and option, with its default, is in the generated [CLI reference](docs/reference/cli.md).

## Configuration

Every setting has a default, can be set in `ocrmd.toml` (project root) or `~/.config/ocrmd/config.toml`, and can be overridden by an `OCRMD_<KEY>` environment variable; CLI flags win over all of them. Invalid values fail fast with the offending key named, and `ocrmd doctor` prints the effective model/provider/focus/workers. The generated [settings reference](docs/reference/settings.md) lists every key with its variable, default and the measurement behind it; [ocrmd.example.toml](ocrmd.example.toml) is a commented file to start from.

## Documentation

[docs/README.md](docs/README.md) indexes every document. The guides:

- [Setup](docs/setup.md) — credentials, camera permission, OCR engines, configuration.
- [Scanning with a webcam](docs/scanning.md) — the rig, the scan window and its keys, getting captures the OCR can use, what the shutter refuses, tuning it.
- [The web UI](docs/web-ui.md) — library, scan, documents, page review and editing, jobs, settings.
- [Output](docs/output.md) — the files on disk, each page's verification status, and which edits survive a re-run.
- [Inventory, search, exports and extraction](docs/corpus.md) — `library`, `search`, `export` and `etl`.
- [Destinations and the library API](docs/integrations.md) — the sink contract and `ocrmd.api.Library`.
- [Rebuilding and annotating pages](docs/round-trip.md) — `reconstruct` and `annotate`.
- [Operations](docs/operations.md) — degraded mode, costs, health, upgrades, logs, debugging in VS Code.
- [Architecture](docs/ARCHITECTURE.md) — the design and the measurement behind each default.

## Project structure

| Path | What lives there |
| --- | --- |
| `ocrmd/` | The package (flat, no sub-packages): CLI, pipeline stages, capture, OCR engines, LLM providers, store, exports, sinks, and `web/` (FastAPI + a dependency-free SPA) |
| `tests/` | pytest suite: synthetic PIL-rendered pages, a fake provider, a temporary root per test |
| `docs/` | Guides, [architecture](docs/ARCHITECTURE.md), generated `reference/`, and [ADRs](docs/adr/README.md) |
| `tools/` | `gen_reference.py` (writes `docs/reference/`) and `docs_index.py` (writes the docs index) |
| `examples/` | A sample ETL pipeline and captures |
| `features/`, `verify/` | The fleet feature index and Playwright user scenarios |
| `.claude/` | The `run-ocrmd` skill (drives the real app), the `verify-feature` skill and the `verifier` agent |
| `.vscode/` | Debug configurations and tasks for the common commands |

Scan data (`raw/`, `processed/`, `ocr/`, `llm/`, `documents/`, `ocrmd.db`) lands in the project directory unless `OCRMD_ROOT` points elsewhere, and is gitignored.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'
.venv/bin/ruff check ocrmd tests
.venv/bin/python -m pytest
.venv/bin/python tools/gen_reference.py --write   # after changing a command, an option or a setting
```

Full commands, conventions and invariants for humans and AI agents live in [AGENTS.md](AGENTS.md).

## Contributing

Contributions are welcome via pull request. Use [Conventional Commits](https://www.conventionalcommits.org/) and keep the CI checks green. The fleet-wide workflow is described in the hub's [CONTRIBUTING.md](https://github.com/bamr87/bamr87/blob/main/CONTRIBUTING.md). AI agents should start with [AGENTS.md](AGENTS.md).

## Changelog

Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

Released under the [MIT License](LICENSE).

---

<!-- bamr87-portfolio-footer -->
Part of the [bamr87 portfolio](https://github.com/bamr87) — browse every project on [Amr's GitHub profile](https://github.com/bamr87).
