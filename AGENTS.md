<!-- kit: sdlc v0.1.0 · AGENTS.md is the CANONICAL agent instructions file (decision D4, UPS-AGENT-07/08). -->

# ocrmd

> A webcam, image and PDF book scanner that reads every page twice — local OCR and a vision LLM — and diffs the readings into cross-checked Markdown with a searchable inventory.

## What this repo is

`ocrmd` turns a webcam pointed at a book (or a folder of images/PDFs) into cross-checked Markdown, and keeps an inventory of everything scanned. Every page is read **twice** — by local OCR engines and by a vision LLM — and the readings are diffed. Disagreements become `==[word]==[^ocrmd-n]` marks in the page file rather than a silent choice; digits are never auto-resolved. That cross-check is the product: an early version fabricated a table of contents for a page it could not read, and only the diff caught it. The root is a searchable **library**; documents **export** as md/txt/json/jsonl, a page can be **sent** to any receiver speaking the contract in [sinks.py](ocrmd/sinks.py), and [api.py](ocrmd/api.py) exposes it all to other programs. One operator at a desk, macOS first.

A change is done when `ruff check ocrmd tests` and `pytest` pass, `docs/reference/` is regenerated if a command, option or `Settings` field changed, and the matching `docs/` guide and [ARCHITECTURE.md](docs/ARCHITECTURE.md) section describe what the code now does. Read that section before changing a subsystem: most non-obvious constants there were measured, not guessed.

## Stack & commands

Python 3.12+, typer, FastAPI, SQLite (WAL), OpenCV, Apple Vision (pyobjc) / RapidOCR, the Anthropic SDK.

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'   # add ,pdf for PDF ingest; ,tesseract as a last resort
.venv/bin/ocrmd doctor --costs                    # engines, cameras, credentials, database health, token spend
.venv/bin/ruff check ocrmd tests                  # lint (the CI gate)
.venv/bin/python -m pytest                        # the full suite; tests that need a local OCR engine skip without one
.venv/bin/ocrmd ingest examples/captures/*.jpg --doc example --no-llm   # the bundled sample captures, local OCR only
.venv/bin/ocrmd serve                             # the web UI at http://127.0.0.1:8765
```

```sh
.venv/bin/python -m pytest tests/test_pipeline.py::test_name -q   # one test
.venv/bin/python .claude/skills/run-ocrmd/driver.py all           # drive the real app: synthetic pages, JSON API, screenshots, optional webcam
.venv/bin/python tools/gen_reference.py --write                   # regenerate docs/reference/ (tests/test_reference_docs.py fails when stale)
python3 tools/docs_index.py --write                               # regenerate the docs/README.md index after adding or renaming a doc
```

The driver uses a throwaway `OCRMD_ROOT` ([run-ocrmd skill](.claude/skills/run-ocrmd/SKILL.md)). [CI](.github/workflows/ci.yml) runs `ruff` on Linux for pull requests and the full suite on **macos-latest** for `main` pushes and manual runs (Apple Vision and AVFoundation are macOS-only). `.vscode/` debug configs run `ocrmd.cli` in-process, so breakpoints work anywhere, including under uvicorn. Every command and option: [docs/reference/cli.md](docs/reference/cli.md).

## Layout

A flat package; module map in [ARCHITECTURE.md §4](docs/ARCHITECTURE.md#4-module-layout), the two drivers in §13a.

```
Capture ──▶ Admit? ──▶ Preprocess ──▶ Local OCR ──▶ LLM pass ──▶ Reconcile ──▶ Render
                     SQLite state + on-disk artifacts (append-only)
```

- **Drivers:** `cli.py` (typer) and `web/app.py` (FastAPI; `web/static/` is a dependency-free SPA served as package data) over one core — `session.ScanSession` owns a scan run (capture generator, OCR worker thread, LLM pool, health, events) and `pipeline.py` runs the stages.
- **Capture:** `capture.py`, `focus.py`, `scene.py`, `legibility.py`, `occlusion.py` (a reference paper-mask tracker, *not* skin detection), `camctl.py` (AVFoundation control plus the OpenCV↔AVFoundation camera-index map — they number cameras differently, and OpenCV reports `-1` for every property on macOS).
- **Stages:** `preprocess.py` (incl. `legible()`, the derived flat rendering), `ocr.py`, `multifocus.py`, `llm.py`, `reconcile.py`, `review.py`, `render.py`; state in `store.py`, settings in `config.py`.
- **After the scan:** `library.py`, `export/`, `etl.py`, `sinks.py`, `api.py`, `reconstruct.py` / `annotate.py`, `bench.py`.
- **Outside the package:** `tests/`, `docs/` (guides, generated `reference/`, `adr/`), `tools/`, `examples/`, `features/` + `verify/` (fleet verify kit), `.claude/` (skills, verifier agent).

## Conventions

- **Backlog of record:** GitHub Issues with the fleet labels (`.github/sdlc.yml` `backlog.mode: issues`). Pick up `agent:ready`; never touch `agent:hold` or `human-review`.
- **Definition of Done:** "done" above, plus the fleet default PR checklist (no PR template here, so the owner's `.github` default applies).
- **Decisions:** ADRs in `docs/adr/NNNN-slug.md`, indexed in [docs/adr/README.md](docs/adr/README.md). Add one for anything hard to reverse.
- **Commits:** Conventional Commits; the PR title is the squash subject. release-please writes `CHANGELOG.md` and the version (`release.yml`) — never edit either by hand.
- **Branches:** `feature/`, `fix/`, `docs/`, `refactor/`, `test/` or `chore/` off `main`.
- **Style:** `ruff` with `line-length = 120`, `select = ["E","F","W","B"]`, and `E401/E501/E701/E702/E731/B008` ignored — compact multi-statement lines and grouped imports are deliberate, and so is typer's `B008` default-in-signature idiom. Match the surrounding density. Module docstrings explain the *why*; inline comments carry the measurement that justifies a constant. When you change a threshold, say what you measured.
- **Tests:** synthetic PIL-rendered pages, a temporary root per test, `FakeProvider` instead of the API ([conftest.py](tests/conftest.py): `text_page`, `on_desk`, `with_hand`, `root`, `store`). Tests needing a real OCR engine skip without one. For pipeline behaviour prefer an end-to-end test through `run_page`/`verify` over unit-testing a helper.
- **Markdown:** one paragraph per line.
- **Never:** push to `main`, commit secrets or lockfiles, suppress type errors, leave empty exception handlers, hand-edit generated files (`docs/reference/`, the docs index span), or edit the vendored `tools/docs_index.py`.

## Stages and versions

- Each stage writes its artifact to disk and records `(page_id, stage, version)` in SQLite; a run is "for each page, run the stages whose output is missing". That makes everything incremental, resumable and interruptible mid-scan. Staleness is *detectable* from the versions but never acted on automatically: that costs tokens.
- Three version strings gate re-work, each with exactly one command that acts on it:

| Version string | Where | Bumped when | Re-run by |
|---|---|---|---|
| `preprocess@N` | [preprocess.py](ocrmd/preprocess.py) | crops/rotation/spread-splitting change | `ocrmd reprocess` |
| `ocr@N:<engines>` | [ocr.py](ocrmd/ocr.py) | preparation or either merge changes; also flips when the engine set changes | `ocrmd verify` (local, free) |
| `transcribe@N:<model>` | `prompt_version` in [config.py](ocrmd/config.py) | the prompt changes; also flips on a different model | `ocrmd verify --refresh` (costs tokens) |

- The search index, the catalogue mirror, the exports and the flat rendering are *derived*: no version, no page id, rebuilt from the files (`ocrmd index --force`, `ocrmd export`).
- **A page's id is the sha256 of its *processed* image**, so bumping `preprocess@N` changes the page identities themselves — that is why `reprocess` exists and why it discards the OCR/LLM results and re-pays for the LLM pass. Raw captures in `raw/` are immutable.
- **Gating and processing stay identical between the CLI and the web server**: new behaviour goes in `session.py`/`pipeline.py`, never in a driver.

## Invariants

Each was measured; the long form is in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

- **A still frame is not a settled scene.** Every per-frame test can be true halfway through a page turn (one session: 21 captures, the outline drifting 0.08–0.16 of the frame between them). [scene.py](ocrmd/scene.py) adds outline stability and a memory of *what* disturbed the scene (a hand or a moved page demands `settle_frames`, not `stable_frames`); don't reduce it to a memoryless chain. Two traps it already hit: an outline the detector never finds counts as a stable observation (a page filling the frame is never a bright blob; blocking deadlocks that rig), and `Outline` scales both sides of the sheet by one number (`minAreaRect` names a sheet `(w,h,-90)` or `(h,w,0)`).
- **Refusing to add is not deleting.** [legibility.py](ocrmd/legibility.py) declines to create a page for an automatic capture no engine could read; the raw frame stays, the `captures` row says why, and `ocrmd ingest raw/<sha>.jpg` overrides it. Never extend it to `ingest` or forced captures. Image-quality thresholds do not separate legible from illegible (best AUC 0.64); don't add one.
- **The focus bracket adds evidence, never identity.** Variants live in `raw/<sha>.v<n>.jpg` and `processed/<pid>.v<n>.png`; the page id, dedup and every stage derive from the primary frame. Variants are cropped with the page's *forced* orientation and scaled to its exact size; one that crops to another aspect ratio is dropped, not stretched. `capture.focus_bracket` puts the lens back where it found it.
- **Confidence says nothing about a line an engine never produced.** `run_multifocus` reads every variant: the two biggest recoveries (55→165 and 101→146 correct words) came from captures read at ≥ 0.98 mean confidence. No confidence gate.
- **Two agreement scores, both required.** `agreement` is order-sensitive, `overlap` order-insensitive containment; a sequence diff alone scored 0% on two *correct* readings of one contents page (OCR reads multi-column layouts out of order). Don't collapse them.
- **Human edits survive re-runs.** Every pipeline write fingerprints the human-editable frontmatter and body ([review.fingerprint](ocrmd/review.py)); on a mismatch the user's `context`, `page_label`, `references.related`, `reviewed_by` and (signed-off pages) body merge forward. The web UI's `PUT /pages/{seq}` and `/resolve` write *without* updating the fingerprint so browser edits count as human edits. Any new write path must preserve this. Provenance (`source`, `processing`, `ocr`, `llm`, `history`, `nav`) is the pipeline's and always overwritten.
- **Two-and-a-half concurrency lanes.** The capture loop owns the window and only enqueues; one worker thread does ingest/preprocess/OCR (Vision is CPU-bound and serial) and writes an OCR-only page file at once, upgraded in place when the LLM answers. Inline processing froze the preview ~25 s per capture — never move work back onto the capture thread.
- **Store access is serialized** by an `RLock` around every `Store` method on one `check_same_thread=False` connection; page files are written atomically (temp + rename).
- **Degraded mode:** `LLMHealth` counts *consecutive service* failures (network/auth/quota); past `llm_failure_limit` the pool stops being fed and the scan continues OCR-only. Page-level refusals do not count.
- **Anti-fabrication is a hard requirement.** The prompt forbids inventing unreadable text and requires `legible: false` with a reason; `suspect`/`illegible` derived from the *numbers* are the backstop, since `illegible_reason` is the model's opinion and has been wrong.
- **The camera will not tell you where the lens is.** On the C525 `isAdjustingFocus` was false on 599/599 frames and `lensPosition` pinned at 0.0. [focus.py](ocrmd/focus.py) judges focus from the sharpness trace (spread over ~0.4 s = moving; ratio to this rig's best in `.ocrmd-camera.json` = soft) and anchors the shutter's floor to that memory, or a session that starts out of focus lowers its own bar. A changed rig is believed after four failed recoveries.
- **Nothing is deleted automatically.** A failed stage records its error; the page stays `unverified` until the next `verify` retries it.
- **An extracted row names its page.** [etl.py](ocrmd/etl.py) attaches `_document`/`_page`/`_page_id` at the source and merges them through every operator (`_pages` for a reduce). Its system prompt is the transcription stage's refusal: use the text, leave the field empty, never fill in what such a document would probably say.
- **Sending is not a stage.** `sinks.send_page` writes no file, marks no stage, moves no version. A down receiver is a **502** from the web route (not a 500), never retried into a silent success. Options are checked against the receiver's descriptor before the wire (`coerce_options`); receiver vocabulary stays in the receiver. The *text* travels, never the photograph — a receiver with the image would re-read it single-pass and overrule the cross-check.
- **The search index is never written by a write path.** `library.reindex` compares page-file mtimes on every search and re-reads only what changed; `Store.delete_page` drops a deleted page's row. Don't hook `render.write_page` or the web routes.
- **A region read writes nothing.** `pipeline.read_region` (drag-a-box, `ocrmd read`) cuts the selection from the flat rendering, enlarges it on its own terms (`REGION_MAX_SCALE`) and returns the model's answer *beside* the OCR one. No stage is marked; the operator is the reconciler, and only Save writes.
- **opencv is pinned `<4.12`:** the KleidiCV HAL segfaults `cv2.resize` on some crops (arm64 macOS); 4.11.0.86 is the last release without it. The crash is hard: it looks like a hang, not an exception.
- **The orientation cascade is a coin toss on real captures** (57 % / 25 % / 53 % per stage, ~30 % end to end, 10 % of stored crops sideways). `orientation_probe = true` (OCR at all four rotations) is measurably better but off by default for its 4–8 Vision passes a page. Don't "simplify" the geometry fallback away — it is what `false` means.

## Settings and credentials

- Precedence: defaults → `~/.config/ocrmd/config.toml` → `<root>/ocrmd.toml` → `OCRMD_<FIELD>` env → CLI flags. Every `Settings` field is an `OCRMD_*` variable; invalid values fail fast naming the key ([generated list](docs/reference/settings.md), [commented example](ocrmd.example.toml)). `OCRMD_ROOT` moves all data; by default `raw/`, `processed/`, `ocr/`, `llm/`, `documents/` and `ocrmd.db` sit beside the code, all gitignored.
- Credentials come from `.env` (cwd → project root → `~/.config/ocrmd/.env`; first found wins, never overriding the environment); `CLAUDE_CODE_OAUTH_TOKEN` is re-exported as `ANTHROPIC_AUTH_TOKEN`. **A subscription OAuth token only serves `claude-haiku-5-5`** (other models return a bare 429), so set `OCRMD_MODEL=claude-haiku-5-5` with it. `ANTHROPIC_API_KEY` reaches every model and is required for `verify --batch`.

## Fleet context

Member of the bamr87 fleet: the hub [bamr87/bamr87](https://github.com/bamr87/bamr87) checks this repo out as `projects/ocrmd` and holds the standards (`specs/`, the Universal Project Standard); the SDLC declaration is `.github/sdlc.yml`. Commit and push here first; the hub bumps its pointer afterwards. Loops acting here: `@claude` mentions (`claude.yml`), advisory conformance (`conformance.yml`) and verify (`verify.yml`; a `verify` label adds the agent pass), release-please (`release.yml`), and the hub's fleet-pulse doctor and issue pipeline.

## Standard deviations

none
