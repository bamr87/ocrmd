# ocrmd — Architecture

Incremental scanning + OCR tool that turns webcam frames or image files (book pages, loose text, receipts) into a standardized Markdown corpus, cross-checked by a second, independent transcription source (Claude with vision; xAI as an alternate provider). Version 0.2 is the production-hardening release, shaped by real sessions with a fixed webcam over a textbook.

## 1. Goals and non-goals

**Goals**
- Point a webcam at a book and turn pages; every stable, new, sharp, *unobstructed* page is captured,
  OCR'd, and appended to the document without pressing anything.
- Drop images/PDFs into a folder (or pass them on the CLI) and get the same output.
- One Markdown format for everything, with enough frontmatter that the corpus is queryable and
  rebuildable, and a document-level metadata file that makes every page citable.
- Every page has *two* independent transcriptions (local OCR engine + vision LLM) that are diffed;
  disagreements are surfaced, never silently resolved; a human can resolve them and the resolution sticks.
- Resumable, idempotent, offline-tolerant: a crash, a dead network or a rate limit never loses a
capture and never stalls a session. Nothing is deleted automatically; nothing costs tokens twice without an explicit flag.
- Pluggable LLM providers behind one interface.

**Non-goals**
- Handwriting, non-Latin scripts beyond what the local engine supports.
- Mobile app. A phone can be a webcam (Continuity Camera / DroidCam) — good enough.
- A GUI beyond the OpenCV preview window with its HUD.
- Multi-user or server deployment. This is a single-operator desktop tool with a SQLite file.

## 2. Assumptions

- **Language:** Python 3.12+. OpenCV (4.x — see §7), Apple Vision through pyobjc, RapidOCR/ONNX
  Runtime, and the official `anthropic` SDK are all first-class.
- **Local OCR engines:** Apple Vision on macOS plus RapidOCR (`ocrmd[rapidocr]`) as a voting second
  engine; RapidOCR alone elsewhere; Tesseract (`ocrmd[tesseract]`) as a measured-weak last resort.
- **Cloud LLM:** Claude Opus 5 (`claude-opus-5`, vision, structured outputs) as the default; xAI Grok
vision models as the alternate. Both are hidden behind `Provider`. Subscription OAuth tokens only reach `claude-haiku-4-5` through the API (measured Aug 2026); ocrmd names that case in its error.
- **Storage:** plain files on disk + one SQLite database (WAL) for state. No server.
- **Rig:** a fixed camera looking down at a book; the operator turns pages by hand. The hand is in
  frame for a large fraction of the session and a paperback may need holding open.

## 3. High-level pipeline

```
 ┌────────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
 │  Capture   │──▶│ Pre-proc  │──▶│ Local OCR  │──▶│ LLM pass  │──▶│ Reconcile │──▶│  Render  │
 │ webcam/    │   │ deskew    │   │ text+boxes │   │ markdown  │   │ diff/     │   │ md +     │
 │ files/pdf  │   │ crop/bin  │   │ +conf      │   │ +struct   │   │ score     │   │ index    │
 └────────────┘   └───────────┘   └────────────┘   └───────────┘   └───────────┘   └──────────┘
        │                │              │                │               │              │
        └────────────────┴──────────────┴────────────────┴───────────────┴──────────────┘
                                   SQLite state + on-disk artifacts (append-only)
```

Every stage writes its result to disk and records completion (stage, version) in SQLite. A run is "for each page, run the stages whose output is missing" — that is what makes it incremental and resumable. Staleness (an older prompt or model, an older preprocessing pipeline) is detectable from the recorded versions but is acted on only when asked (`verify --refresh`, `reprocess`), because acting on it costs tokens.

## 4. Module layout

```
ocrmd/
  cli.py          typer app: scan | ingest | verify | reprocess | review | status | delete | build | serve | doctor | ocr-bench
  config.py       Settings (defaults → toml → OCRMD_* env), .env credentials, logging setup
  store.py        SQLite: documents, pages, captures, stages, llm_calls, batches; migrations; locking
  capture.py      auto-shutter generator: gates (Gate), camera open/reopen, focus settle, HUD loop
  occlusion.py    hand gate: reference paper-mask tracker
  hud.py          preview overlay: gate meters, page outline, work panel, advice
  camview.py      rotation + digital zoom/pan, persisted per camera
  camctl.py       AVFoundation focus/exposure/format control and the OpenCV↔AVFoundation index map
  preprocess.py   page rect, warp, orientation cascade, spread split, deskew, binarise, framing advice
  fingerprint.py  sha256, dHash, frame correlation and text-stroke correlation (dedup)
  ocr.py          engines: Apple Vision (pyobjc), RapidOCR, Tesseract; cross-engine merge; Recognizer; OcrResult
  bench.py        OCR scored against the store's LLM transcriptions (`ocrmd ocr-bench`)
  llm.py          Provider protocol; AnthropicProvider (streaming, structured output, batches);
                  XAIProvider; PageHint; error taxonomy
  pipeline.py     stage runner: ingest/dedup, preprocess, ocr, llm, reconcile+render, continuity,
                  batches, edit preservation, housekeeping
  reconcile.py    OCR vs LLM alignment, flags, agreement/overlap scores, status policy
  review.py       flag parsing/resolution, fingerprint of human-editable parts, merge of human edits
  render.py       page/document writers, document.yaml, build (inheritance, TOC, gaps)
  session.py      ScanSession (capture loop + OCR worker + LLM pool + events), process_page, make_provider
  web/app.py      FastAPI backend: documents, pages, review, jobs, scan control, MJPEG stream
  web/jobs.py     background job manager (verify, build, reprocess, ingest) with per-job logs
  web/health.py   doctor report as data (shared by `ocrmd doctor` and the UI)
  web/static/     the frontend: index.html, app.js (hash-routed SPA, no dependencies), app.css
tests/            pytest suite (synthetic pages, fake provider, temporary roots, FastAPI TestClient)
```

The first draft of this document listed sub-packages (`ocr/`, `llm/`, `reconcile/`, `store/`, `tui/`). The flat layout above is what was built; the modules are small enough that packages would add ceremony without separation.

## 5. Capture

### Camera control

Two device settings dominate image quality and neither is reachable through OpenCV, which returns `-1` for every `CAP_PROP` on macOS and silently ignores writes. `camctl.py` drives them through AVFoundation (`AVCaptureDevice`), which applies to the stream OpenCV is already reading:

- **Focus mode.** Webcams commonly power up *locked* at lens position 0 — every frame blurry, and no
preprocessing recovers it. Measured on the development rig: a locked C525 produced a sharpness (Laplacian variance) of 38; continuous autofocus took it to 299. `scan` sets continuous focus and exposure before opening the stream. **`--focus settle`** (recommended for a fixed rig) keeps continuous autofocus until the first capture and then locks: the page distance never changes, and a hand entering the frame otherwise makes the lens hunt on every page turn.
- **Resolution.** ocrmd enumerates `AVCaptureDevice.formats` and opens the largest mode. Modes above
the sensor size were assumed to be worthless; measurement says otherwise (146 words read at 1920x1080 vs 98 at 1280x960 on the same page) because what matters is pixels landing on the page.
- **Index mapping.** OpenCV enumerates cameras in a different order than AVFoundation — on the
development Mac, exactly reversed. Nothing in either API maps between them, so ocrmd identifies each OpenCV index by probing it with a resolution only one attached camera supports.
- **Drop-outs.** `cap.read()` failing for `camera_reopen_after` consecutive frames releases and reopens
the device; after five reopens the generator raises `CameraLost` and the scan ends cleanly (queue drained, document built).

Digital zoom/pan/rotation (`camview.py`) is a crop-and-rotate of the sensor frame applied identically to the preview and the captured frame, persisted per camera name.

### The auto-shutter gates

`capture.Gate.evaluate` is a pure per-frame decision, ordered so that each test is only asked when the earlier ones cannot answer:

1. **Motion** — mean absolute difference of the blurred 320-px grey frame vs the previous one below
   `motion_threshold` for `stable_frames` consecutive frames. A hand moving, a page turning.
2. **Focus** — the device reports `isAdjustingFocus`; sharpness is meaningless while it hunts.
3. **Hand** — see below. Must come *before* novelty: the hand is exactly what makes the frame "new".
4. **Novelty** — dHash Hamming distance to the *last captured* frame ≥ `novelty_threshold`.
5. **Sharpness** — Laplacian variance inside the detected page outline, on the unmodified frame (the
motion blur must never touch this buffer), above an absolute floor *and* above `sharp_ratio` of the best this scene has produced (decayed, and not updated while a hand is in frame — skin is sharper than paper).

`space` bypasses every gate; `capture_cooldown` limits auto captures to one per second.

### The hand gate (`occlusion.py`)

The first real sessions produced a run of captures with a hand flat across the text. All three original gates passed: the hand stopped moving, it changed the hash, and it was in focus. Two approaches were measured against the 57 real captures on disk:

- *Skin fraction inside the page outline* — flagged 48 of 57, including most clean pages, because the
  wooden desk inside the rotated page rectangle is skin-toned in YCrCb.
- **Reference paper mask** — the approach shipped. At analysis scale, threshold the frame (Otsu),
drop skin-coloured pixels, morphologically close/open with a kernel wider than text, and keep the result as *the paper*. Compare each frame's paper mask with a reference:

  ```
  missing = reference & ~current      paper that has disappeared   (hand, pen, curled page)
  added   = current   & ~reference    paper that has appeared      (book moved, page unfolded)
  ```

A hand removes paper and adds none. A moved book does both, and re-baselines the reference immediately. Skin colour inside the *missing* region only labels the intrusion; a skin-toned intrusion is held up to 3× longer before the tracker gives up and accepts the scene (`occlusion_rebaseline` frames), because a wooden desk revealed by a moved book is skin-toned too and a permanently stuck gate is worse than a late capture. The reference follows the scene between captures (a new page has near-identical paper) but never shrinks to what a hand leaves uncovered.

Offline score against hand-labelled captures: 21 of 26 hand frames caught at `occlusion_cover=0.05`; the misses are fingertips on the margin (cover ≤ 0.05) and two frames where the book moved at the same time. Clean frames compared to a neighbouring clean capture measure ≤ 0.02.

### Files / directories / PDFs

`ingest path...` accepts images and PDFs (`pymupdf`, rendered at 200 dpi); `--watch` polls a directory. The same frames go through the same pipeline; `source` in the page frontmatter records which.

### Dedup

Exact duplicates are caught by content sha256 of the JPEG. Near-duplicates are caught in two steps, **within the same document only** (a global check dropped the first page of every new book that resembled an old one):

1. dHash prefilter within `near_duplicate_distance` bits (loose: 10), then normalised correlation of
the 96×72 grey frame against the candidate's raw capture ≥ `near_duplicate_similarity` (0.985). Measured on real captures: re-captures of the same page score 0.99, consecutive different pages ≤ 0.98 — but two mostly-blank pages (chapter openers) are indistinguishable this way.
2. Confirmation by **OCR text**: the new crops are recognised right there (only when a candidate
exists; the result is kept as the page's OCR stage) and compared with the candidate's OCR text — token overlap ≥ `near_duplicate_text` (0.85) with similar length. Text is invariant to the few pixels of shift a page picks up between captures. When either side has too little text, the fallback is high-pass image correlation ≥ `near_duplicate_detail` (0.5; same page 0.75–0.81, different pages ≤ 0.47 on the real captures).

A duplicate is recorded as a `capture` row pointing at the existing page and goes no further. `ocrmd status` reports how many were skipped.

## 6. Pre-processing

Ordered, each step recorded in `processed/<id>.json`:

1. Page detection: bright blobs (paper against a darker desk) merged when comparably sized (a deep
gutter shadow splits a spread into two blobs) → `minAreaRect` → perspective warp. Contour-based quad detection was tried first and failed on real photographs.
2. Clipped edges: frame edges the sheet touches *and where the edge strip carries ink* — bare
   geometry cried wolf on nearly every capture.
3. Orientation cascade, strongest evidence first: text-line-frequency power of the row profile
(`{0,180}` vs `{90,270}`); dictionary hits from a fast, non-auto-orienting OCR pass; ink asymmetry (ascenders outnumber descenders) as the never-silent tie-breaker.
4. Two-page spread split at the darkest vertical band in the middle third; deskew per half.
5. Denoise + adaptive threshold copy for engines that want it (Tesseract). Apple Vision and the LLM
   get the colour crop.
6. Framing advice: text cut off at an edge, or fewer than `min_page_width` px across the page. Both
   are shown live in the HUD, written to the page, stored on the `pages` row, and listed by `status`.

`PIPELINE_VERSION` (`preprocess@3`) is bumped whenever the crop changes; page ids are the hash of the processed image, so `reprocess` re-derives pages from raw when it does.

## 7. Local OCR (transcription A)

```python
class OcrResult(BaseModel):
    text: str                      # reading-order plain text
    blocks: list[Block]            # bbox (normalised, top-left origin), text, confidence
    mean_confidence: float
    engine: str                    # "apple_vision@26.5", "tesseract@5.3"
    words: list[tuple[str, float]] # per-word confidence in reading order
```

`engine` names every engine that ran (`apple_vision@26.5+rapidocr@3.9.2`); `agreement` is the share of the primary's words a second engine read identically. Engines take the colour crop and prepare it themselves: cubic upscaling until the short side is ≥ `ocr_min_side` (1400 px, ≤ 3×; a 517-px crop yielded 11 words, upscaled 367) — and for Tesseract only, illumination flattening.

### Engines

| engine | how | F1 (35 pages) | s/page | role |
|---|---|---|---|---|
| `apple_vision` | `VNRecognizeTextRequest` via pyobjc, accurate level, `ocr_languages`, language correction on | **0.74** | 0.75 | primary on macOS |
| `rapidocr` | PP-OCR det+cls+rec on ONNX Runtime (`ocrmd[rapidocr]`) | 0.66 | 1.5 | second voter on macOS; primary elsewhere |
| `tesseract` | binary via pytesseract, `--psm 3` on the flattened crop | 0.39 | 1.2 | last resort |
| Apple Live Text (not shipped) | `VKCImageAnalyzer` through `ocrmac` | 0.72 | 0.56 | slower to call, no confidences, no better |

F1 is bag-of-words against the LLM transcription of each page (`bench.py`), on 35 legible webcam pages of a textbook (20 of them under 600 px wide, where every engine drops to ≈0.5 — resolution, not recognition, is the ceiling there). Preparation variants measured for Vision: Lanczos = cubic, upscaling to 2000 px −0.1, unsharp mask −0.2, background flattening −2.7, CLAHE −3.4, language correction off −6.2. None is applied except the upscaling. RapidOCR's larger PP-OCRv5 "server" models measured 0.25 at 13 s/page on these crops, so the default mobile models stay. `ocrmd ocr-bench` reruns this table.

### Cross-engine merge (`ocr.merge`)

The primary's lines are the skeleton; every secondary line that shares ≥ half its height and any width with a primary line is aligned to it word by word (`SequenceMatcher` on lower-cased alphanumerics):

- an `equal` word is **confirmed**: confidence ≥ `CONFIRMED` (0.9 = `ocr_trust`), so in a short
  disagreement with the LLM the OCR reading may win;
- a `replace`/`delete` word is **doubted**: confidence × `DOUBTED` (0.6), so it is flagged, not trusted;
- a secondary line no primary line overlaps is **filled in** if it has ≥ 2 words and confidence ≥ 0.85;
- a primary line under 0.5 confidence with fewer than half its words confirmed is **swapped** for a
  ≥ 0.9 secondary reading of similar length.

Measured effect: the text is the same (F1 0.735 → 0.736; fill and swap rarely fire on a working rig) but the confidence becomes informative. Vision alone puts 10 107 of 10 234 words at ≥ 0.9 with 82 % precision; merged, 6 443 words stay ≥ 0.9 at 89 % and 3 619 land in a 0.6–0.9 band at 70 %. That is the difference between `reconcile` trusting an OCR word over the model and flagging it for a human. `ocr_ensemble = false` (or `ocr_engines = ["apple_vision"]`) restores the single engine.

The OCR stage version is `ocr@2:<engine names>`: a new engine set re-reads a page the next time it is processed (`verify` includes such pages; local, free), and the LLM stage is reused from cache.

**OpenCV is pinned below 5.** `opencv-python 5.0.0.93`'s KleidiCV HAL segfaults in `cv2.resize` on a real crop in this repository (845×1505 → 544×960, arm64 macOS), reproducibly, on the main thread and single-threaded. Every stage resizes.

## 8. LLM pass (transcription B + structure)

### Provider interface

```python
class Provider(Protocol):
    name: str
    model: str
    def transcribe(self, image_bgr, ocr_hint: str, hint: PageHint | None) -> LLMResult: ...
```

`transcribe` returns a structured object (structured outputs on Anthropic, JSON-schema `response_format` on xAI), validated into `Transcription`: kind, markdown body, title, printed page number, language, `legible` + reason, uncertain spans, page context (chapter, section, running head, citations, figures), notes.

`PageHint` is what the pipeline knows before the model looks: which half of a spread this is, the document title and kind from `document.yaml`, and the previous page's label/chapter/section/running head. The prompt asks the model to carry chapter/section forward unless the page starts a new one, which is what keeps heading levels consistent across a textbook and lets `build` produce a TOC.

### Anthropic provider

- `claude-opus-5` default; `output_config.effort = llm_effort` (medium — perception, not reasoning)
and server-side refusal fallbacks (`fallbacks="default"`) on the 4.6+/5 family; neither is sent to models that reject them (Haiku 4.5) nor to the Batches API (which rejects `fallbacks`).
- Streaming with `get_final_message()`; `max_tokens` 16000; client `timeout`/`max_retries` from
  settings (SDK retries 429/5xx/connection errors with backoff).
- Image as a base64 JPEG (≤1568 px, quality 95 — small type suffers at 90), then the OCR draft as a
hint with garbage lines removed (they mislead the model and can trip the output filter; a filter hit is retried from the image alone).
- Prompt caching: the system prompt is the stable prefix with one `cache_control` breakpoint.
- **Batches:** `batch_submit` builds the same params per page under a `custom_id` = page id;
`batch_collect` keys results by `custom_id` (results arrive in any order). Open batches are recorded in SQLite so `verify --collect` can fetch them in a later session.

### Error taxonomy

Two exception families, because a scan must react differently:

| Class | Examples | Effect in `scan` | Effect in `verify` |
|---|---|---|---|
| `LLMUnavailable` | auth rejected, model not served to this token, connection error, 5xx/429 after retries | counted; after `llm_failure_limit` consecutive the session is **degraded** (OCR-only, HUD `LLM OFF`) | stops the run with a message |
| `LLMPageError` | refusal, content filter, `max_tokens`, schema mismatch, 4xx | recorded on the page (`stages.error`, `verification.llm_error`), page stays `unverified`, next page proceeds | same |

In both cases the page file is (re)written OCR-only so the document never has a hole.

### xAI provider

Same interface; OpenAI-compatible chat endpoint with an image content part and strict JSON-schema `response_format`, stdlib `urllib` with exponential retry on 408/409/429/5xx. Implemented against the documented request shape and unit-tested for that shape; not exercised against the live API here.

## 9. Auth

- **Anthropic:** `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` env; `CLAUDE_CODE_OAUTH_TOKEN` is mapped
onto `ANTHROPIC_AUTH_TOKEN` and sent with the `oauth-2025-04-20` beta header; the `ant auth login` profile under `~/.config/anthropic/` is picked up by the SDK's zero-arg client. `.env` is loaded from cwd, the project root, then `~/.config/ocrmd/.env`, never overriding real environment variables.
- **xAI:** `XAI_API_KEY` only. xAI does not offer third-party OAuth; the design's OAuth path was dropped.
- Logs keep HTTP client loggers at WARNING; tokens and request bodies are never written.

## 10. Canonical Markdown format

One file per page, one aggregated file per document, one `document.yaml` per document. The dialect is deliberately narrow so output is stable and diffable.

### Page file: `documents/<slug>/pages/0001.md`

```markdown
---
ocrmd: 1
id: 7f3c…                     # sha256(processed image)[:16]
document: lsat
sequence: 12
captured_at: 2026-08-26T10:23:44-06:00
kind: book_page
page_label: "37"              # as printed
language: en
context:                      # human/document wins; LLM fills blanks
  title: "LSAT Prep"          # inherited from document.yaml on build
  author: []
  chapter: "Analytical Reasoning"   # from the LLM, or carried forward on build
  section: null
  running_head: "Mometrix"
  citation_key: null
  tags: [study]
  project: null
references: { cites: [], figures: [], footnotes: 0, related: [] }
source: { type: webcam, device: "HD Webcam C525", raw: raw/….jpg, sha256: …, phash: …, part: left }
processing: { pipeline: preprocess@3, page_detected: true, clipped_edges: [], orientation: 0,
              part: left, deskew: 0.0, blur: 417.0, glare: 0.0, page_width_px: 936,
              advice: null, continuity: "printed page jumped 37 → 41 — 3 page(s) may have been missed" }
ocr:  { engine: apple_vision@26.5, confidence: 0.97, artifact: ocr/7f3c….json }
llm:  { provider: anthropic, model: claude-opus-5, prompt: transcribe@3, request_id: msg_…,
        tokens: { in: 2217, out: 633, cache_read: 1400 } }
verification:
  status: flagged             # verified | reviewed | flagged | suspect | illegible | blank | unverified
  agreement: 0.87             # order-sensitive
  overlap: 0.93               # order-insensitive
  cer: 0.13
  flagged_spans: 2
  arbiter: null
  reviewed_by: null           # { who, at } set by `ocrmd review`
  llm_uncertain: [...]        # spans the model itself doubted
  llm_error: null             # why the LLM pass failed, if it did
history: [ { stage: preprocess, version: preprocess@3, completed_at: … }, … ]
nav: { prev: pages/0011.md, next: pages/0013.md, document: ../lsat.md }
---

## Chapter 3 — Title as printed

Body text… a word both sources disagreed on ==[ambitious]==[^ocrmd-1] here.

[^ocrmd-1]: ocr="ambiguous" llm="ambitious" — unresolved
```

Dialect rules: ATX headings by document hierarchy; paragraphs rejoined; hyphenation undone; lists stay lists; boxed tips become blockquotes; GFM tables; source footnotes `[^n]`, ocrmd's own `[^ocrmd-n]`; no inline HTML; figures as `> **Figure:** caption`.

### Frontmatter groups and ownership

| Group | Fields | Origin | Who may edit |
|---|---|---|---|
| Identity | `ocrmd, id, document, sequence, captured_at, kind, language` | pipeline | pipeline |
| **Context** | `context.*`, `page_label`, `references.related` | document.yaml → LLM → human | human wins; LLM fills blanks; build inherits |
| Provenance | `source, processing, ocr, llm, history, nav` | pipeline | pipeline |
| Verification | `verification.*` | reconcile + `ocrmd review` | `reviewed_by` human only |

**Edit preservation** is implemented by fingerprinting the Context group + reviewer + body every time the pipeline writes a page (`pages.written_fingerprint`). On the next write, a fingerprint mismatch means a human edited the file: their non-null Context values, page label, related links and reviewer are merged over the fresh output, and if a reviewer is recorded the body is kept as well and the status becomes `reviewed`. Provenance and nav are always regenerated, which is why they are outside the fingerprint — a `build` (which rewrites nav) never looks like a human edit.

### `document.yaml`

Created on first `scan`/`ingest`/`build` with a template; `--title/--tag/--project` seed it. Holds title, author, edition, publisher, year, isbn, citation, citation_key, kind (default `textbook`), tags, project, physical, reading_direction. `build` inherits title/author/citation_key/tags/project into every page where the page has no value.

### Document file: `documents/<slug>/<slug>.md`

Generated by `build`: the document context, page count, first–last printed label, status counts, unresolved flag count, `needs_reshoot` (sequences that are illegible/suspect), `label_gaps` (printed page numbers that skip), aggregate agreement; a TOC from chapter changes with printed labels; then each page body under `<!-- page: 0012 label: 37 id: 7f3c… status: flagged -->`. Never hand-edited; page files are the source of truth.

## 11. Reconciliation

1. Normalize both texts (NFC, whitespace, quote/dash folding, de-hyphenation).
2. Align word sequences with `difflib`; short replaced spans are candidates, long spans and
   insert/delete are the LLM's (OCR garbage is not evidence).
3. Per candidate: numeric mismatch → **flag, unresolved**; OCR confident and near-identical → take
   OCR; OCR middling → flag; otherwise trust the LLM.
4. Two scores: `agreement` (order-sensitive) and `overlap` (order-insensitive containment of content
   tokens). Both are required: OCR reads contents pages and columns out of order.

| Status | Condition |
|---|---|
| `blank` | neither source has content tokens |
| `suspect` | OCR has text and `overlap < suspect_threshold` — one source is not describing this page |
| `verified` | `agreement ≥ flag_threshold`, nothing unresolved |
| `flagged` | otherwise |
| `illegible` | (pipeline) the model declared the page unreadable and produced < 40 chars |
| `reviewed` | (review) a human resolved every flag; body frozen |

5. **Anti-fabrication.** The prompt forbids inventing unreadable text, including text under a
   finger, and requires `legible: false` plus a reason. `suspect` exists to catch what the prompt misses.
6. **Continuity.** After reconciliation the printed page number is compared with the previous
page's: a repeat, a small jump (≤ 20) or a step backwards is reported on the page (`processing.continuity`), in the console, in the HUD, and aggregated in the document (`label_gaps`).

## 12. State and storage

```
<root>/                          (OCRMD_ROOT; default: the project directory)
  ocrmd.db                       SQLite, WAL
  ocrmd.log                      rotating log
  ocrmd.toml                     optional configuration
  raw/<sha256>.jpg               immutable captures
  processed/<id>.png|.bin.png|.json
  ocr/<id>.json
  llm/<id>.<provider>.json
  documents/<slug>/document.yaml, pages/*.md, <slug>.md
  .ocrmd-camera.json             per-camera view
```

Tables: `documents`, `pages` (id, document, sequence, hashes, status, page_label, part, width_px, advice, written_fingerprint, reviewed_by/at), `captures` (every frame incl. duplicates), `stages` (page_id, stage, version, completed_at, error), `llm_calls` (tokens, latency), `batches`. Indexes on (document, sequence), (document, status), captures.page_id, llm_calls.page_id.

Stage versions: `preprocess@N`; `ocr@N:<engines>` (a changed engine set re-reads on the next `verify`, locally); `transcribe@N:<model>` for the LLM pass, so `verify --refresh` re-runs pages made by an older prompt *or* a different model and nothing else does.

**Migrations** are additive (`COLUMNS` in `store.py`): `CREATE TABLE IF NOT EXISTS` never adds columns to an existing table, so every column added after 0.1 is applied by an explicit `ALTER TABLE`. Old pages get `width_px`/`advice` backfilled from `processed/<id>.json` by `status`/`doctor`.

## 13. Concurrency and failure

- **Three lanes during a scan.** The capture loop (main thread, owns the window) only enqueues
frames. One worker thread does ingest/preprocess/OCR — Vision is CPU-bound and serial — and renders an OCR-only page immediately. A `ThreadPoolExecutor(llm_workers)` runs the LLM pass and re-renders when it lands. Queue depth (frames + pending LLM jobs) and the active stages are in the HUD.
- **Store access is serialized** by an `RLock` around every method on one `check_same_thread=False`
  connection; WAL + `synchronous=NORMAL` keeps a crash mid-write from corrupting the file.
- **Page files are written atomically** (temp file + rename).
- **Degraded mode:** `LLMHealth` counts consecutive `LLMUnavailable`; past `llm_failure_limit` the
  pool stops being fed and the HUD shows `LLM OFF`. Page-level errors do not count.
- **Quit:** drain the frame queue, wait for the pool, build the document. A second Ctrl-C cancels
  pending LLM jobs; the pages remain `unverified` with their OCR text and `verify` completes them.
- A stage failure is recorded with the error and the page stays `unverified`; the next `verify`
  retries it. Nothing is ever deleted automatically.
- Refusal (`stop_reason == "refusal"`) after fallback → `LLMPageError`, recorded, continue.

## 13a. Web UI

`ocrmd serve` runs a FastAPI app (uvicorn, `127.0.0.1:8765`, no auth) serving a dependency-free single-page frontend from the package. The design constraints:

- **One orchestration, two drivers.** `session.ScanSession` owns a scanning run (capture generator,
OCR worker, LLM pool, health, event buffer). The CLI calls `run()` on the main thread with the OpenCV window; the web server calls `start()` on a thread with `preview=False`, receives the HUD frame through `auto_shutter(on_frame=…)` as JPEG, and feeds key presses through a command queue (`capture`, `focus_lock`, `rotate`, `zoom_in`, …). Nothing about gating or processing differs.
- **Streaming.** `/api/scan/stream` is MJPEG (`multipart/x-mixed-replace`) at ≤20 fps from the latest
frame under a condition variable; `/api/scan` is polled at 2 Hz for gate numbers, queue, recent pages and events. No WebSocket needed for a local tool.
- **Jobs.** verify/build/reprocess/ingest run on one worker thread (OCR is serial, the store is one
connection), each with a bounded log, progress and a cancel flag; `reprocess` is refused while its document is being scanned. Uploads land in `<root>/inbox/<slug>/` and go through the same ingest.
- **Deletion** (`DELETE /pages/{seq}`, `POST /pages/delete`, `DELETE /documents/{slug}?confirm=slug`)
removes rows, artifacts and page files, and purges the raw capture once no page refers to it. Its capture rows go too, so the same frame can be scanned again without tripping dedup. Sequence numbers are not renumbered (page files are named by sequence and printed labels carry the order). Refused while the document is being scanned.
- **Edits are human edits.** `PUT /pages/{seq}` and `/resolve` write the page file *without* updating
the pipeline fingerprint, so the file counts as hand-edited and `verify` preserves it (§10). `/signoff` records the reviewer and freezes the body; `/reopen` reverts to the computed status.
- **Frontend.** ~700 lines of vanilla JS with a hash router (dashboard, scan, document, page,
review mode, jobs, settings) and a small renderer for the Markdown dialect (headings, lists, tables, blockquotes, footnotes, `==[…]==` marks). Works offline; no build step; served as package data.

## 14. Configuration and observability

`Settings` (pydantic) with defaults → `~/.config/ocrmd/config.toml` → `<root>/ocrmd.toml` → `OCRMD_<FIELD>` env → CLI flags. Validation errors name the field. `ocrmd doctor` prints the effective key settings, checks the OCR engine, credentials, cameras (native format, focus state), camera authorization, database integrity, missing and orphan artifacts, open batches, the dictionary used by the orientation probe, and per-model token spend.

Logging: `ocrmd.*` loggers; console at WARNING (`--debug` DEBUG, `--quiet` ERROR); rotating file at `log_level`. Every capture logs sharpness/occlusion/px-per-page; every LLM failure logs its class.

## 15. CLI surface

```
ocrmd scan      --doc SLUG [--camera N|name] [--focus continuous|locked|settle|off] [--no-llm]
                [--provider anthropic|xai] [--model M] [--resolution WxH] [--rotate D] [--zoom Z]
                [--title T] [--tag T]... [--project P] [--no-preview] [--duration S] [--max-pages N]
ocrmd ingest    PATH... --doc SLUG [--watch] [--no-llm] [--title/--tag/--project]
ocrmd verify    [--doc SLUG] [--flagged-only] [--refresh] [--batch [--no-wait]] [--collect]
ocrmd reprocess [--doc SLUG] [--no-llm] [-y]
ocrmd delete    --doc SLUG (--page N... [--whole-capture] | --all) [--keep-raw] [-y]
ocrmd review    --doc SLUG [--who NAME] [--list]
ocrmd status    [--doc SLUG] [--all]
ocrmd build     [--doc SLUG]
ocrmd doctor    [--costs]
ocrmd ocr-bench [--doc SLUG] [--engines a+b | --compare] [--limit N] [-v]
ocrmd serve     [--host 127.0.0.1] [--port 8765] [--no-open]
ocrmd --debug|--quiet <command>
python -m ocrmd <command>
```

## 16. Testing

`tests/` runs in about a minute on a Mac (`pytest`): synthetic pages rendered with PIL, a temporary root per test, a `FakeProvider` in place of the API. Covered: camera view keys and persistence; orientation, framing advice, clipped-text detection, spread splitting; the hand gate (detection, reference stability, moved book, non-hand intrusion timeout, page turn) and gate ordering; the store (0.1 migration, document-scoped dedup, thread safety, stage versions, batches); configuration precedence and validation; the pipeline end to end (OCR-only then LLM upgrade with OCR cached, reuse vs `--refresh`, human-edit and reviewer preservation, page-number continuity, error recording, illegible/suspect, duplicates, spreads with hints, batch round trip, re-deriving missing artifacts); build (inheritance, chapter carry-forward, TOC, gaps, needs_reshoot); review parsing/resolution; request shapes for both providers; the CLI (ingest/status/review/build/verify, degraded mode, interactive review); and the web API (documents, pages, images, resolve/sign-off/reopen, human edits surviving a re-run, document.yaml, jobs, upload ingest, scan start failure). Tests needing the OCR engine skip where none is installed. CI runs on macOS.

## 17. Build history

1. ~~Skeleton + files path~~ **done** (0.1)
2. ~~Anthropic provider + reconcile~~ **done** (0.1) — plus anti-fabrication prompting, the
`suspect`/`illegible`/`blank` statuses and the order-insensitive overlap metric, none of which were in the original design and all of which real pages forced.
3. ~~Webcam auto-shutter~~ **done** (0.1) — plus focus/exposure/format control, view transforms, threaded worker.
4. ~~Production hardening~~ **done** (0.2) — hand gate, focus settle, camera reopen, LLM lanes and
degraded mode, error taxonomy, continuity, batches, xAI, PDF/watch ingest, document.yaml and inheritance, edit preservation, review, status, configuration, logging, migrations, tests, CI. 4b. ~~Web UI~~ **done** (0.3) — `ocrmd serve`: live scan with HUD stream and controls, document and page browser with images, click-to-resolve review and sign-off, jobs, metadata, doctor.
5. **Receipts:** typed line items, arithmetic check, per-kind prompt variants. *not started*
6. **Cross-provider arbitration** for low-agreement pages. *not started* (the interface allows it)

### What the real scans changed

- The camera was focus-locked; sharpness 38 → 299 once AVFoundation set continuous autofocus.
- Apple Vision needs ≈1400 px of page width; a 517-px crop yielded 11 words, upscaled it yielded 367.
- The model fabricated a table of contents rather than admitting it could not read one.
- A sequence-only agreement metric scored 0% on two correct transcriptions of the same contents page.
- The motion-detection blur was being applied to the buffer used to measure sharpness.
- Processing inline froze the preview window for ~25 s after every capture.
- OpenCV and AVFoundation number the cameras differently, so focus was being set on the other camera.
- The "interpolated" 1080p mode reads 49% more words than the native 4:3 mode.
- Frame averaging, Lanczos, CLAHE and unsharp masking all measured as noise; none were kept.
- Row-profile *variance* let a gutter shadow outvote a page of text; the direction test now measures
  power at text-line frequencies.
- **The shutter fired with a hand flat on the page** (0.2). Skin detection inside the page outline
  failed on a wooden desk; a reference paper mask works.
- **A dHash cannot tell two pages of one book apart** on near-blank pages; frame correlation plus
  text-stroke correlation inside the paper can (0.2).
- **`source.type` was always `file`** — `"webcam" in str(sqlite3.Row)` is never true (0.2).

## 18. Open questions

- Two-page spread handling from a fixed overhead camera: the perspective warp assumes the page is
  roughly planar; a dewarp step (e.g. `page-dewarp`) may be needed for thick books near the spine.
- Whether to ship PaddleOCR (heavy dependency) or keep it a documented extra.
- Resolution remains the binding constraint: a 720p webcam framing a full spread gives ~500 px per
page against the ~900 needed. Options are physical (mount the camera closer, one page at a time) or a stills path — the C525 can produce higher-resolution stills than its video modes, but not through AVFoundation's video capture path.
- Fingertips on the margin pass the hand gate (cover ≤ 0.05). Lowering the threshold trips on page
  curl; the honest fix is a glass plate or weights, which the README now says.
- The hand gate has no reference at session start: a hand already in frame would be baked into the
first baseline and the first capture would include it. The gate re-baselines the moment the hand withdraws (paper *added*), so only the first capture is exposed. A start-up grace period that waits for the largest paper area seen in the first few seconds would close it.
- `illegible_reason` is the model's opinion, not a measurement. In the live smoke test Haiku 4.5
reported a soft left page as "heavily obscured by the hand holding the book open" when no hand was in the frame (the gate's `cover=0.00` was correct; the page curves toward the spine and sat outside the locked focus plane). The prompt now asks for observed symptoms only, but the cross-check — `suspect`/`illegible` from the *numbers* — remains the thing to trust.
- Thick books: with `--focus settle` the lens locks on whatever it settled on; the half of the spread
that curves up toward the spine can be soft while the page-level sharpness gate (measured over the whole detected sheet) still passes. Per-half sharpness, or a dewarp step, is the next lever.
