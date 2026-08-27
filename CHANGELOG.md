# Changelog

## 0.4.0 — 2026-08-26

### OCR
- **Multi-engine OCR with voting.** `ocr.py` now hosts an engine registry (`apple_vision`, `rapidocr`,
  `tesseract`) and merges two engines' readings line by line: a word both read is *confirmed*
  (confidence ≥ 0.9, so it may overrule the LLM in a short disagreement), a word they read differently
  is *doubted* (×0.6, so it is flagged instead of trusted), lines the primary missed are filled in from
  a confident secondary reading, and a line the primary barely read is swapped for a confident one.
  Measured on 35 webcam pages against the LLM transcriptions: the text itself is unchanged (F1 0.735 →
  0.736) but Vision's near-binary confidence becomes a usable signal — words it marks ≥ 0.9 are right
  82 % of the time alone, 89 % once a second engine agrees, and the doubted band (70 %) no longer
  passes as confident. Costs ~1.5 s/page of CPU; `ocr_ensemble = false` turns it off.
- **Apple Vision is called directly** (pyobjc `VNRecognizeTextRequest`) instead of through `ocrmac`:
  25 % faster, and recognition languages (`ocr_languages`, default `en-US`) and language correction
  are explicit. Language correction is worth 6 F1 points on these pages and stays on.
- **RapidOCR** (`pip install 'ocrmd[rapidocr]'`): PP-OCR on ONNX Runtime, the primary engine off macOS.
  Measured F1 0.66 alone (Tesseract: 0.39, Apple Live Text: 0.72, Vision: 0.74).
- Preparation stays plain cubic upscaling to 1400 px: background flattening, CLAHE, unsharp masking and
  larger targets all measured at or below it for Vision (flattening helps Tesseract and is applied there).
- OCR results record the engine set, per-engine timings and the share of words the engines agreed on
  (`ocr.agreement` in page frontmatter and the UI). The OCR stage version is `ocr@2:<engines>`;
  `verify` re-reads pages made by an older engine set (local, no tokens) and re-renders them.
- `ocrmd ocr-bench [--compare] [--engines a+b]`: bag-of-words recall/precision/F1 of local OCR against
  the legible LLM transcriptions in the store, split by page width. This is the harness that chose the
  defaults above; run it after installing an engine or moving the camera.

### Capture
- **Burst capture**: when the gates open, the shutter reads `burst_frames` (default 3) consecutive
  frames and keeps the sharpest — consecutive webcam frames of a still page differ by 10-20 % in
  Laplacian variance, so this is free sharpness.

### Fixes
- Pinned `opencv-python<5`: 5.0.0.93's KleidiCV HAL segfaults inside `cv2.resize` on some real page
  crops (arm64 macOS; reproducible on a crop in this repository). The whole pipeline resizes.
- Dependencies: `ocrmac` dropped; `pyobjc-framework-Vision` added on macOS; extras `rapidocr`, `tesseract`.

## 0.3.0 — 2026-08-26

### Web UI (`ocrmd serve`)
- Local FastAPI server + dependency-free single-page frontend at http://127.0.0.1:8765.
- **Scan** page: live HUD stream (MJPEG), gate state and meters, capture / focus / rotate / zoom / pan
  controls and keys, session panel, event log; start/stop with document, camera, focus, OCR-only,
  duration and page limits.
- **Documents**: status bars, filterable page table, drag-and-drop ingest, verify/build/reprocess,
  download of the built document.
- **Page**: image beside transcription (processed / raw / binarised, zoom), one-click resolution of
  each disagreement (OCR / LLM / custom), Markdown editor and preview, metadata editor, verification
  and provenance, sign-off / reopen, keyboard navigation, review mode.
- **Jobs** with live logs, progress and cancel; **Settings** with `document.yaml` editor, effective
  settings and the doctor report. API docs at `/api/docs`.
- Scan orchestration extracted into `session.ScanSession`, shared by CLI and server; `auto_shutter`
  gained `on_frame`/`commands` so any driver can replace the OpenCV window.
- `doctor` report is now data (`web/health.py`), rendered by both the CLI and the UI.
- **Deletion**: pages (single, selected, or a whole capture), and whole documents (type-the-slug
  confirmation) from the UI; `ocrmd delete` on the CLI. Artifacts and page files go with them; a raw
  capture is purged once nothing refers to it; the frame can then be scanned again.
- New dependencies: fastapi, uvicorn, python-multipart (httpx for tests).

## 0.2.0 — 2026-08-26

Production hardening after the first real sessions over a textbook on a fixed rig.

### Capture
- **Hand gate.** The shutter no longer fires while a hand (or anything else) covers paper that was
  visible a moment ago. Implemented as a reference paper-mask comparison (`occlusion.py`); the HUD
  shows `HAND IN FRAME`. A moved book re-baselines immediately; a non-hand intrusion is accepted
  after ~3 s.
- `--focus settle`: continuous autofocus until the first capture, then locked — the lens stops hunting
  every time a hand comes in. Default in `ocrmd.example.toml`.
- Camera drop-outs are detected and the device reopened (up to 5 times) instead of spinning forever.
- Every capture logs sharpness, occlusion and px/page.
- `scan --duration S` / `--max-pages N` for bounded, unattended runs (and smoke tests).

### Processing
- OCR and the LLM pass run in separate lanes during a scan: the OCR-only page appears immediately,
  a small pool (`llm_workers`) upgrades it when the model answers. Quitting drains the queue; a second
  Ctrl-C hands the remainder to `ocrmd verify`.
- **Degraded mode.** After `llm_failure_limit` consecutive *service* failures (network, auth, quota)
  the scan continues OCR-only instead of stalling on every page.
- Errors are classified: service (`LLMUnavailable`) vs page (`LLMPageError`: refusal, filter, schema).
  Page failures are recorded in the page file (`verification.llm_error`) and in `ocrmd status`.
- Printed page-number continuity: a repeat or a jump is reported per page and in the document
  frontmatter (`label_gaps`).
- The model gets context (left/right half of the spread, previous page's chapter/section/label,
  document title) — prompt `transcribe@3`. Existing pages are **not** re-run unless `verify --refresh`.
- `verify --batch` sends backlogs through the Message Batches API at half price; `--collect` fetches.
- xAI provider (`--provider xai`, `XAI_API_KEY`), OpenAI-compatible JSON-schema request; untested live.
- PDF ingest (`pip install 'ocrmd[pdf]'`), `ingest --watch` for a drop folder.

### Output
- `documents/<slug>/document.yaml`: title/author/citation/tags/project, inherited into every page on
  `build`; chapter/section carried forward across heading-less pages; TOC with printed page labels;
  `needs_reshoot` and `unresolved_flags` in the document frontmatter.
- **Human edits are preserved.** Context, page label, related links and reviewer sign-off in a page
  file survive `verify`; a reviewed page keeps its body.
- `ocrmd review`: walk disagreements, pick OCR/LLM/your own text, sign pages off as `reviewed`.
- `ocrmd status`: per-page table of what needs a re-shoot, a review or an LLM pass, and why.
- Page files are written atomically.

### Operations
- Configuration file (`ocrmd.toml`, `~/.config/ocrmd/config.toml`) and `OCRMD_*` env overrides for
  every setting, validated with clear errors.
- Rotating log file (`ocrmd.log`), `--debug`/`--quiet`.
- SQLite in WAL mode, all access locked, indexes, additive migrations from 0.1 databases.
- `doctor` checks database integrity, missing/orphan artifacts, open batches, dictionary presence.
- `python -m ocrmd`, optional extras (`tesseract`, `pdf`, `dev`), pytest/ruff config, CI workflow.

### Fixed
- `source.type` was always `file` for webcam captures.
- Near-duplicate detection compared against every document; a page resembling one in another book
  was silently dropped. It also relied on a 64-bit dHash alone, which cannot tell two pages of one
  book apart when the layout matches; candidates are now confirmed by frame correlation and OCR text.
- `verify` re-ran the LLM on every page after a prompt-version bump (now opt-in via `--refresh`).
- A page whose LLM pass failed on first ingest had no page file at all.

## 0.1.0 — 2026-08-25
First working version: webcam auto-shutter, Apple Vision OCR, Claude transcription with structured
outputs, reconciliation with flags, per-page Markdown with frontmatter, document build.
