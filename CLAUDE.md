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

Common runtime entry points: `ocrmd scan --doc X --camera c525` (webcam), `ocrmd ingest ./dir --doc X`, `ocrmd verify --doc X` (finish missing stages), `ocrmd serve` (web UI on 127.0.0.1:8765), `ocrmd read --doc X --page N --region x,y,w,h` (re-read one region), `ocrmd etl --doc X -p pipeline.yaml` (extract structured data), `ocrmd ocr-bench --doc X --compare` (score OCR engines against the store's LLM transcriptions).

## Architecture

Flat package, no sub-packages — the modules are small enough that packaging adds ceremony.

```
Capture ──▶ Admit? ──▶ Preprocess ──▶ Local OCR ──▶ LLM pass ──▶ Reconcile ──▶ Render
                     SQLite state + on-disk artifacts (append-only)
```

The load-bearing idea: **each stage writes its artifact to disk and records `(page_id, stage, version)` in SQLite; a run is "for each page, run the stages whose output is missing."** That is what makes everything incremental, resumable, and interruptible mid-scan. Staleness is *detectable* from the recorded versions but is never acted on automatically, because acting on it costs tokens.

Three stage version strings gate re-work, and each has exactly one command that acts on it:

| Version string | Where | Bumped when | Re-run by |
|---|---|---|---|
| `preprocess@N` | [preprocess.py:7](ocrmd/preprocess.py#L7) | crops/rotation/spread-splitting change | `ocrmd reprocess` |
| `ocr@N:<engines>` | [ocr.py:35](ocrmd/ocr.py#L35) | preparation or either merge changes; also flips when the engine set changes | `ocrmd verify` (local, free) |
| `transcribe@N:<model>` | `prompt_version` in [config.py](ocrmd/config.py) | the prompt changes; also flips on a different model | `ocrmd verify --refresh` (costs tokens) |

**A page's id is the sha256 of its *processed* image.** Bumping `preprocess@N` therefore changes the page identities themselves — that is why `reprocess` exists and why it discards the OCR/LLM results and re-pays for the LLM pass. Raw captures in `raw/` are immutable and never modified.

### Module map

`cli.py` (typer app) and `web/app.py` (FastAPI) are two drivers over one core. `session.ScanSession` owns a scanning run — capture generator, OCR worker thread, LLM `ThreadPoolExecutor`, health, event buffer. The CLI calls `run()` on the main thread with an OpenCV window; the web server calls `start()` on a thread with `preview=False` and receives HUD frames as JPEG via `auto_shutter(on_frame=…)`, feeding key presses through a command queue. **Gating and processing must stay identical between the two** — put new behaviour in `session.py`/`pipeline.py`, never in a driver.

- [capture.py](ocrmd/capture.py) — auto-shutter generator, its gates, and the shutter's burst + focus bracket (`focus_bracket`); [scene.py](ocrmd/scene.py) the settle decision (page-outline stability + what disturbed it); [legibility.py](ocrmd/legibility.py) the admission test; [occlusion.py](ocrmd/occlusion.py) hand gate (reference paper-mask tracker, *not* skin detection); [hud.py](ocrmd/hud.py) overlay; [camview.py](ocrmd/camview.py) rotate/zoom/pan persisted per camera; [camctl.py](ocrmd/camctl.py) AVFoundation focus/exposure/format **plus the OpenCV↔AVFoundation camera-index map** (they number cameras differently; OpenCV also reports `-1` for every property on macOS).
- [preprocess.py](ocrmd/preprocess.py) — page rect, warp, three-test orientation cascade, spread split, deskew, binarise, framing advice,
  and `legible()` — the flat rendering (`processed/<pid>.flat.png`, the `flat` view): illumination divided out, unsharp, Otsu tone curve. Derived, not staged — no version, no page id, made on first request by `pipeline.ensure_flat`.
- [ocr.py](ocrmd/ocr.py) — engine registry (`apple_vision`, `rapidocr`, `tesseract`), the cross-engine merge (`ocr.merge`), `Recognizer`, `OcrResult`. [bench.py](ocrmd/bench.py) scores engines against the store.
- [multifocus.py](ocrmd/multifocus.py) — the same vote one axis over: the focus bracket's exposures resolved line by line (`resolve`), and the unresolved regions handed to the model (`describe`).
- [llm.py](ocrmd/llm.py) — `Provider` protocol, `AnthropicProvider` (streaming, structured output, Batches API), `XAIProvider`, `PageHint`, and the error taxonomy that distinguishes service failures from page-level ones.
- [pipeline.py](ocrmd/pipeline.py) — the stage runner: ingest/dedup, the four stages, continuity, batches, edit preservation, deletion, housekeeping.
- [etl.py](ocrmd/etl.py) — extraction over a finished document: DocETL's operator model (map/filter/extract/reduce/resolve/split/unnest/sample)
  on ocrmd's own `Provider.complete_json`, YAML specs, per-operator caching versioned by the prompt, `--export-docetl` to hand the same spec to DocETL's runner.
- [reconcile.py](ocrmd/reconcile.py) — alignment, flags, the two scores, status policy. [review.py](ocrmd/review.py) — flag parsing/resolution and the human-edit fingerprint. [render.py](ocrmd/render.py) — page/document writers, `document.yaml` inheritance, TOC, gaps.
- [store.py](ocrmd/store.py) — SQLite (WAL), migrations, locking. [config.py](ocrmd/config.py) — `Settings` and logging.
- [web/](ocrmd/web/) — `app.py` API + MJPEG stream, `jobs.py` single worker thread, `health.py` doctor-as-data (shared with `ocrmd doctor`), `static/` a dependency-free hash-routed SPA served as package data.

## Invariants worth knowing before you edit

- **A still frame is not a settled scene.** Every per-frame test (motion, focus, hand, novelty) can
be true halfway through a page turn — measured: one real session took 21 captures with the page outline drifting 0.08-0.16 of the frame between consecutive ones. [scene.py](ocrmd/scene.py) adds page-outline stability and a memory of *what* disturbed the scene, so a hand or a moved page demands `settle_frames` rather than `stable_frames`. Don't reduce it back to a memoryless chain. Two traps it already hit: an outline the detector never finds must be treated as a stable observation (a page filling the frame is never a bright blob, and blocking there deadlocks that rig), and `Outline` must scale both sides of the sheet by the same number — `minAreaRect` names one sheet `(w,h,-90)` or `(h,w,0)`.
- **Refusing to add is not deleting.** [legibility.py](ocrmd/legibility.py) declines to create a page
for an automatic capture no OCR engine could read; the raw frame stays, the `captures` row records why, and `ocrmd ingest raw/<sha>.jpg` overrides it. Never extend this to `ingest` or to forced captures. Image-quality thresholds were measured against the model's verdict and do not separate legible from illegible (best AUC 0.64) — don't add one back on intuition.
- **The focus bracket adds evidence, never identity.** A capture's other exposures live in
`raw/<sha>.v<n>.jpg` and `processed/<pid>.v<n>.png`; the page id, the dedup decision and every stage still derive from the primary frame alone. Variant crops are made with the page's *forced* orientation and scaled onto its exact dimensions so boxes are comparable, and one that crops to a different aspect ratio is dropped rather than stretched. `capture.focus_bracket` must always put the lens back where it found it.
- **Confidence says nothing about a line an engine never produced.** `run_multifocus` reads every
variant, not only the ones for pages with doubtful lines: measured on a half-blurred page, the two biggest recoveries (55→165 and 101→146 correct words) were on captures the primary read at ≥ 0.98 mean confidence and had simply not emitted the unreadable lines. Don't reintroduce a confidence gate there.
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
- **An extracted row names its page.** `etl.py` attaches `_document`/`_page`/`_page_id` at the source
and merges them through every operator (`_pages` for a reduce). An extraction you cannot take back to the photograph is a rumour, and the whole repo exists because a model once invented a contents page. The ETL system prompt is the same refusal the transcription stage runs on: use the text, leave the field empty, never fill in what a document of this kind would probably say.
- **A region read writes nothing.** `pipeline.read_region` (the web UI's drag-a-box tool, `ocrmd read`,
`POST /pages/{seq}/region`) cuts the selection out of the flat rendering, enlarges it on its own terms (`REGION_MAX_SCALE`, larger than a page's cap — a region is a fraction of a page, so the same pixels buy several times the glyph height) and re-reads it. No stage is marked, no artifact is rewritten, and the model's answer is returned *beside* the OCR one, never instead of it: this path has no second reading to reconcile against, so the operator is the reconciler. The UI inserts into the editor; only Save writes.
- **opencv is pinned `<4.12`**: the KleidiCV HAL segfaults in `cv2.resize` on some page crops (arm64
macOS, measured). `<5` was not enough — KleidiCV ships in 4.12+ too, and 4.14 crashes on the wider crops the current spread logic produces. 4.11.0.86 is the last release with no KleidiCV in `getBuildInformation()`. A hard process crash, so it looks like a hang or a silent death, not an exception.
- **The orientation cascade is a coin toss on real captures** (57 % / 25 % / 53 % per stage, ~30 % end
to end, 10 % of stored crops sideways). `orientation_probe = true` swaps in an OCR probe over all four rotations that is measurably better, and is off by default because it costs 4-8 Vision passes a page. Don't "simplify" the geometry fallback away — it is what `false` means.
- **[layout.py](ocrmd/layout.py) / [reconstruct.py](ocrmd/reconstruct.py) / [annotate.py](ocrmd/annotate.py)**:
the round trip (page → Markdown → SVG/LaTeX → page, scored) and annotation sync (camera on an already-scanned page; underline/highlight cost nothing, only handwriting reaches a model).

## Settings and credentials

Precedence: defaults → `~/.config/ocrmd/config.toml` → `<root>/ocrmd.toml` → `OCRMD_<FIELD>` env → CLI flags. Every field on `Settings` is automatically an `OCRMD_*` variable; invalid values fail fast naming the key. [ocrmd.example.toml](ocrmd.example.toml) documents the full set. `OCRMD_ROOT` moves all data (default: the project directory itself — `raw/`, `processed/`, `ocr/`, `llm/`, `documents/`, `ocrmd.db` live alongside the code, and all but `documents/` are gitignored).

Credentials come from `.env` (cwd → project root → `~/.config/ocrmd/.env`, first found wins, never overriding the existing environment). `CLAUDE_CODE_OAUTH_TOKEN` is re-exported as `ANTHROPIC_AUTH_TOKEN` for the SDK. **A subscription OAuth token only serves `claude-haiku-4-5`** — other models come back as a bare 429; set `OCRMD_MODEL=claude-haiku-4-5` alongside it. `ANTHROPIC_API_KEY` reaches every model and is required for `verify --batch`.

## Testing

`tests/` uses synthetic PIL-rendered pages, a temporary root per test, and `FakeProvider` instead of the API ([tests/conftest.py](tests/conftest.py) has `text_page`, `on_desk`, `with_hand`, `root`, `store`). Tests that need a real OCR engine skip when none is installed. When adding pipeline behaviour, prefer an end-to-end test through `run_page`/`verify` over unit-testing a helper — the suite's value is in covering stage reuse, refresh, edit preservation and the status policy together.

## Style

`ruff` with `line-length = 120`, `select = ["E","F","W","B"]`, and `E401/E501/E701/E702/E731/B008` ignored — **compact multi-statement lines and grouped imports are deliberate here**, and so is typer's `B008` default-in-signature idiom. Match the surrounding density; module docstrings explain the *why* and inline comments carry the measurement that justifies a constant. Keep that habit: when you change a threshold, say what you measured.
