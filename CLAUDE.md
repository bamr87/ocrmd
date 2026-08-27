# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ocrmd` turns a webcam pointed at a book (or a folder of images/PDFs) into cross-checked Markdown. Every page is read **twice** — once by local OCR engines, once by a vision LLM — and the two readings are diffed. Disagreements become `==[word]==[^ocrmd-n]` marks in the page file rather than a silent choice; digits are never auto-resolved. That cross-check is the product, not a nicety: an early version fabricated a table of contents for a page it could not read, and only the diff caught it.

Long-form design lives in [ARCHITECTURE.md](ARCHITECTURE.md) (module-by-module, with the measurements that produced each default) and [README.md](README.md) (operator-facing). [CHANGELOG.md](CHANGELOG.md) records what each version changed and why. Read the relevant ARCHITECTURE section before changing a subsystem — most non-obvious constants there are the result of a measurement, not a guess.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'   # ,pdf for PDF ingest; ,tesseract last resort
.venv/bin/ruff check ocrmd tests      # lint (CI gate)
.venv/bin/python -m pytest            # full suite, ~1 min
.venv/bin/python -m pytest tests/test_pipeline.py::test_name -q   # one test
.venv/bin/ocrmd doctor --costs        # engines, cameras, credentials, DB health, token spend
```

To drive the actual app — synthetic pages through the real pipeline, the JSON API, screenshots of every web view, an optional live-webcam check — use the run skill's driver, which does all of it against a throwaway `OCRMD_ROOT` so the repo's own store is untouched:

```sh
.venv/bin/python .claude/skills/run-ocrmd/driver.py all   # see .claude/skills/run-ocrmd/SKILL.md
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs `ruff check ocrmd tests` then `pytest -q` on **macos-latest** — Apple Vision and AVFoundation are macOS-only and the pipeline tests need a local OCR engine. [.vscode/tasks.json](.vscode/tasks.json) and [.vscode/launch.json](.vscode/launch.json) wrap these (lint+test is the default build task; every debug config runs `ocrmd.cli` in-process so breakpoints work anywhere, including under uvicorn).

Common runtime entry points: `ocrmd scan --doc X --camera c525` (webcam), `ocrmd ingest ./dir --doc X`, `ocrmd verify --doc X` (finish missing stages), `ocrmd serve` (web UI on 127.0.0.1:8765), `ocrmd ocr-bench --doc X --compare` (score OCR engines against the store's LLM transcriptions).

## Architecture

Flat package, no sub-packages — the modules are small enough that packaging adds ceremony.

```
Capture ──▶ Preprocess ──▶ Local OCR ──▶ LLM pass ──▶ Reconcile ──▶ Render
                     SQLite state + on-disk artifacts (append-only)
```

The load-bearing idea: **each stage writes its artifact to disk and records `(page_id, stage, version)` in SQLite; a run is "for each page, run the stages whose output is missing."** That is what makes everything incremental, resumable, and interruptible mid-scan. Staleness is *detectable* from the recorded versions but is never acted on automatically, because acting on it costs tokens.

Three stage version strings gate re-work, and each has exactly one command that acts on it:

| Version string | Where | Bumped when | Re-run by |
|---|---|---|---|
| `preprocess@N` | [preprocess.py:7](ocrmd/preprocess.py#L7) | crops/rotation/spread-splitting change | `ocrmd reprocess` |
| `ocr@N:<engines>` | [ocr.py:31](ocrmd/ocr.py#L31) | preparation or merge changes; also flips when the engine set changes | `ocrmd verify` (local, free) |
| `transcribe@N:<model>` | `prompt_version` in [config.py](ocrmd/config.py) | the prompt changes; also flips on a different model | `ocrmd verify --refresh` (costs tokens) |

**A page's id is the sha256 of its *processed* image.** Bumping `preprocess@N` therefore changes the page identities themselves — that is why `reprocess` exists and why it discards the OCR/LLM results and re-pays for the LLM pass. Raw captures in `raw/` are immutable and never modified.

### Module map

`cli.py` (typer app) and `web/app.py` (FastAPI) are two drivers over one core. `session.ScanSession` owns a scanning run — capture generator, OCR worker thread, LLM `ThreadPoolExecutor`, health, event buffer. The CLI calls `run()` on the main thread with an OpenCV window; the web server calls `start()` on a thread with `preview=False` and receives HUD frames as JPEG via `auto_shutter(on_frame=…)`, feeding key presses through a command queue. **Gating and processing must stay identical between the two** — put new behaviour in `session.py`/`pipeline.py`, never in a driver.

- [capture.py](ocrmd/capture.py) — auto-shutter generator and its gates; [occlusion.py](ocrmd/occlusion.py) hand gate (reference paper-mask tracker, *not* skin detection); [hud.py](ocrmd/hud.py) overlay; [camview.py](ocrmd/camview.py) rotate/zoom/pan persisted per camera; [camctl.py](ocrmd/camctl.py) AVFoundation focus/exposure/format **plus the OpenCV↔AVFoundation camera-index map** (they number cameras differently; OpenCV also reports `-1` for every property on macOS).
- [preprocess.py](ocrmd/preprocess.py) — page rect, warp, three-test orientation cascade, spread split, deskew, binarise, framing advice.
- [ocr.py](ocrmd/ocr.py) — engine registry (`apple_vision`, `rapidocr`, `tesseract`), the cross-engine merge (`ocr.merge`), `Recognizer`, `OcrResult`. [bench.py](ocrmd/bench.py) scores engines against the store.
- [llm.py](ocrmd/llm.py) — `Provider` protocol, `AnthropicProvider` (streaming, structured output, Batches API), `XAIProvider`, `PageHint`, and the error taxonomy that distinguishes service failures from page-level ones.
- [pipeline.py](ocrmd/pipeline.py) — the stage runner: ingest/dedup, the four stages, continuity, batches, edit preservation, deletion, housekeeping.
- [reconcile.py](ocrmd/reconcile.py) — alignment, flags, the two scores, status policy. [review.py](ocrmd/review.py) — flag parsing/resolution and the human-edit fingerprint. [render.py](ocrmd/render.py) — page/document writers, `document.yaml` inheritance, TOC, gaps.
- [store.py](ocrmd/store.py) — SQLite (WAL), migrations, locking. [config.py](ocrmd/config.py) — `Settings` and logging.
- [web/](ocrmd/web/) — `app.py` API + MJPEG stream, `jobs.py` single worker thread, `health.py` doctor-as-data (shared with `ocrmd doctor`), `static/` a dependency-free hash-routed SPA served as package data.

## Invariants worth knowing before you edit

- **Two agreement scores, both required.** `agreement` is order-sensitive, `overlap` is order-insensitive
vocabulary containment. A sequence diff alone scored 0% on two *correct* transcriptions of the same contents page, because OCR reads multi-column and contents layouts out of order. Don't collapse them.
- **Human edits survive re-runs.** The pipeline fingerprints the human-editable frontmatter and body
every time it writes ([review.fingerprint](ocrmd/review.py#L60)); if the file's fingerprint doesn't match, the user's `context`, `page_label`, `references.related`, `reviewed_by` and (for signed-off pages) the body are merged forward. The web UI's `PUT /pages/{seq}` and `/resolve` deliberately write *without* updating the fingerprint so browser edits count as human edits. Any new write path must preserve this.
- **Provenance is the pipeline's, context is the human's.** `source`, `processing`, `ocr`, `llm`,
  `history` and `nav` are always overwritten; the fields above are never clobbered.
- **Two-and-a-half concurrency lanes during a scan.** The capture loop owns the window and only
enqueues; one worker thread does ingest/preprocess/OCR (Vision is CPU-bound and serial) and writes an **OCR-only page file immediately**, upgrading it in place when the LLM answers. Processing inline froze the preview for ~25 s per capture — don't move work back onto the capture thread.
- **Store access is serialized** by an `RLock` around every `Store` method on one
  `check_same_thread=False` connection. Page files are written atomically (temp + rename).
- **Degraded mode:** `LLMHealth` counts *consecutive service* failures (network/auth/quota); past
`llm_failure_limit` the pool stops being fed and the scan continues OCR-only. Page-level refusals must not count toward it.
- **Anti-fabrication is a hard requirement.** The prompt forbids inventing unreadable text and requires
`legible: false` with a reason; `suspect`/`illegible` derived from the *numbers* are the backstop, because `illegible_reason` is the model's opinion and has been observed to be wrong.
- **Nothing is deleted automatically.** A failed stage is recorded with its error, the page stays
  `unverified`, and the next `verify` retries it.
- **opencv is pinned `<5`**: 5.0's KleidiCV HAL segfaults in `cv2.resize` on some page crops (arm64 macOS, measured).

## Settings and credentials

Precedence: defaults → `~/.config/ocrmd/config.toml` → `<root>/ocrmd.toml` → `OCRMD_<FIELD>` env → CLI flags. Every field on `Settings` is automatically an `OCRMD_*` variable; invalid values fail fast naming the key. [ocrmd.example.toml](ocrmd.example.toml) documents the full set. `OCRMD_ROOT` moves all data (default: the project directory itself — `raw/`, `processed/`, `ocr/`, `llm/`, `documents/`, `ocrmd.db` live alongside the code, and all but `documents/` are gitignored).

Credentials come from `.env` (cwd → project root → `~/.config/ocrmd/.env`, first found wins, never overriding the existing environment). `CLAUDE_CODE_OAUTH_TOKEN` is re-exported as `ANTHROPIC_AUTH_TOKEN` for the SDK. **A subscription OAuth token only serves `claude-haiku-4-5`** — other models come back as a bare 429; set `OCRMD_MODEL=claude-haiku-4-5` alongside it. `ANTHROPIC_API_KEY` reaches every model and is required for `verify --batch`.

## Testing

`tests/` uses synthetic PIL-rendered pages, a temporary root per test, and `FakeProvider` instead of the API ([tests/conftest.py](tests/conftest.py) has `text_page`, `on_desk`, `with_hand`, `root`, `store`). Tests that need a real OCR engine skip when none is installed. When adding pipeline behaviour, prefer an end-to-end test through `run_page`/`verify` over unit-testing a helper — the suite's value is in covering stage reuse, refresh, edit preservation and the status policy together.

## Style

`ruff` with `line-length = 120`, `select = ["E","F","W","B"]`, and `E401/E501/E701/E702/E731/B008` ignored — **compact multi-statement lines and grouped imports are deliberate here**, and so is typer's `B008` default-in-signature idiom. Match the surrounding density; module docstrings explain the *why* and inline comments carry the measurement that justifies a constant. Keep that habit: when you change a threshold, say what you measured.
