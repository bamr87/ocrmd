# ocrmd — Architecture

Incremental scanning + OCR tool that turns webcam frames, image files or PDFs (book pages, loose text, receipts) into a standardized Markdown corpus, cross-checked by a second, independent transcription source (Claude with vision; xAI as an alternate provider) — with an inventory of everything scanned, exports in several shapes, a generic outbound contract for handing a page to another system, and a library API for embedding the whole thing. Version 0.2 was the production-hardening release, shaped by real sessions with a fixed webcam over a textbook; 0.9 is the one that made it a general-purpose book scanner rather than a rig for one book.

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
- One root is a **library**: every document scanned into it is catalogued and searchable, and the
  catalogue lives in SQLite so a tool that is not ocrmd can read it.
- The corpus is a **source**, not a destination: it exports to the shapes other tools take (Markdown,
text, JSON, JSON-lines), it can be handed page by page to any receiver that speaks a small HTTP contract, and it can be driven as a library from another program.

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
- **Cloud LLM:** Claude Opus 5 (`claude-opus-5-5`, vision, structured outputs) as the default; xAI Grok
vision models as the alternate. Both are hidden behind `Provider`. Subscription OAuth tokens only reach `claude-haiku-5-5` through the API (measured Aug 2026); ocrmd names that case in its error.
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
                  | library | search | index | export | sinks | send | read | etl | reconstruct | annotate
  api.py          the embedding surface: `Library` — one Settings, one Store, an injectable provider
  config.py       Settings (defaults → toml → OCRMD_* env), .env credentials, SinkConfig, logging setup
  store.py        SQLite: documents (with the catalogue mirrored in), pages, captures, stages, llm_calls, batches,
                  page_text (+ FTS5), doc_exports; migrations; locking
  library.py      the inventory: document.yaml mirrored into the store, page bodies indexed by mtime, search
  export/         md | txt | json | jsonl exporters behind one `export_document`
  sinks.py        destinations: the receiver contract (descriptor + envelope), HttpSink, entry-point sinks
  capture.py      auto-shutter generator: gates (Gate), camera open/reopen, focus bracket, HUD loop
  scene.py        has the desk stopped changing? page-outline stability + the settle state machine
  legibility.py   should a capture become a page at all? (the admission test)
  layout.py       OCR geometry joined to the reconciled text; figure regions
  reconstruct.py  the page rebuilt as SVG + LaTeX, rendered, and scored against the original
  annotate.py     pen and highlighter added to an already-scanned page, folded into the Markdown
  occlusion.py    hand gate: reference paper-mask tracker
  hud.py          preview overlay: gate meters, page outline, work panel, advice
  camview.py      rotation + digital zoom/pan, persisted per camera
  camctl.py       AVFoundation focus/exposure/format control and the OpenCV↔AVFoundation index map
  preprocess.py   page rect, warp, orientation cascade, spread split, deskew, binarise, framing advice
  fingerprint.py  sha256, dHash, frame correlation and text-stroke correlation (dedup)
  ocr.py          engines: Apple Vision (pyobjc), RapidOCR, Tesseract; cross-engine merge; Recognizer; OcrResult
  multifocus.py   the focus bracket resolved line by line: swap/fill across exposures, unresolved regions
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

The decision is split in two. `scene.Scene` answers *has the desk stopped changing?* and `capture.Gate` answers the rest, which are properties of the page rather than of the scene.

**The scene** (`scene.py`), in order, so that each test is only asked when the earlier ones cannot answer:

1. **Motion** — mean absolute difference of the blurred 320-px grey frame vs the previous one, above
   `motion_threshold`. A hand moving, a page turning.
2. **Focus** — the device reports `isAdjustingFocus`; sharpness is meaningless while it hunts.
3. **Hand** — see below. Must come *before* novelty: the hand is exactly what makes the frame "new".
4. **Page outline** — the detected sheet must have held its position for `geometry_frames` (3)
   consecutive detections within `geometry_tolerance` (0.03 of the frame).
5. **Settling** — after all of the above go quiet, `stable_frames` (8) undisturbed frames, or
   `settle_frames` (20) if the disturbance being recovered from was a hand or a moved page.

**The page**, once the scene is settled:

6. **Novelty** — dHash Hamming distance to the *last captured* frame ≥ `novelty_threshold`.
7. **Sharpness** — Laplacian variance inside the detected page outline, on the unmodified frame (the
motion blur must never touch this buffer), above an absolute floor *and* above `sharp_ratio` of the best this scene has produced (decayed, and not updated while a hand is in frame — skin is sharper than paper).

`space` bypasses every gate; `capture_cooldown` limits auto captures to one per second; a capture spends the scene's quiet, so one settled period yields one capture.

**Why steps 4 and 5 exist.** The original chain was memoryless, and every one of its tests can be true *in the middle of a page turn*: the hand lets go, the page is mid-fall, the pixels go quiet for a fraction of a second, the lens has not moved, and the frame is certainly novel. Measured on the store in this repository, one two-minute session produced 21 captures and 16 of the 20 consecutive pairs disagree about where the page is by more than 0.03 of the frame — outline drifts of 0.08 to 0.16, i.e. the page was demonstrably still moving. All 21 became pages.

The page outline is the signal that catches this, and it is a quiet one: re-detecting 30 real captures under sensor noise (450 draws) moved the outline less than 0.001 of the frame 95 % of the time, and past 0.03 in only 0.89 % of draws. Every one of those unstable draws came from a single frame of the thirty — the detector is not randomly jittery, particular scenes are genuinely ambiguous about where the sheet is, and holding the shutter on those is right rather than a false negative. So the test is the strict one (all detections in the window agree) with no outlier tolerance.

Remembering *what kind* of disturbance is the other half. A desk that twitches only has to go quiet again; a desk where a hand just moved a page has to prove it has finished, which is what `settle_frames` buys. A page turn that ends with a moment of ordinary jitter is still a page turn, so the scene keeps the worst disturbance it has seen rather than the last one.

Two deliberate non-behaviours. An outline the detector *never* finds is a stable observation, not a blocker: a page that fills the frame edge to edge is never a bright blob, and refusing to ever capture on such a rig would be worse than the problem being solved. And `minAreaRect` names one sheet either `(w, h, -90)` or `(h, w, 0)`, so `Outline` scales both sides by the same number and holds them as an unordered pair — normalising width by the frame width and height by the frame height makes the same still page look like a 0.37 drift when the naming flips.

### The shutter: burst and focus bracket

When the gates open the shutter takes more than one frame.

**Burst** (`burst_frames`, 3): read the next few frames and keep the sharpest. The scene is still by construction, so they differ only in sensor noise, compression and the lens's last micro-adjustment — and Laplacian variance between consecutive webcam frames of the same page varies by 10-20 %. Free sharpness.

**Focus bracket** (`focus_bracket`, default `auto`): re-shoot at several lens positions. Autofocus optimises one number over the whole frame, and on a book that number is dominated by whatever has the most contrast — a page edge, a gutter shadow, the flatter of two leaves — so the frame that wins on *average* sharpness can be the worse frame for the top third of the page, and no preprocessing recovers what the optics missed there. `capture.focus_bracket` drives `AVCaptureDevice.setFocusModeLockedWithLensPosition` to `focus_bracket_shots` positions (default 3: where the lens is, then ±`focus_bracket_step` = 0.05), waits `focus_bracket_settle` (0.25 s) at each — reading frames throughout, which both lets the lens travel and flushes the frames the camera buffered *before* it moved — and keeps the sharpest as the capture and the next `focus_keep` (2) as **variants**. The lens is always put back: continuous/auto focus is re-enabled, a deliberately locked lens (`--focus settle`) is returned to its position. Leaving a scan focused wherever the last shot landed is the one failure this must not introduce.

`auto` brackets only when the frame is below `focus_bracket_ratio` (0.9) of the best sharpness this scene has produced: when the lens is already on the page a bracket costs ~0.5 s of frozen preview for nothing. `always` brackets every capture, `off` disables it.

**Many webcams cannot be bracketed at all.** `AVCaptureDevice.isLockingFocusWithCustomLensPositionSupported` is false on both cameras of the development rig — the C525 offers `locked` and `continuous` focus modes but no lens position — and `ocrmd doctor` reports it per camera. There, `focus_bracket` reads no extra frames and returns nothing, and the shutter falls back to the **burst's runners-up**: the frames `sharpest_burst` already read and used to discard. They are the same scene at the same focus, differing only by sensor noise, compression and the lens's own micro-adjustment, so they are a far weaker second opinion than a real change of focus — the gain in the table below was measured with genuinely differing focus and does **not** transfer to the fallback, which is unmeasured on real hardware. `focus_keep = 0` disables it.

The variants ride with the capture into `ingest_frame` and are stored as `raw/<sha>.v<n>.jpg`, as immutable as the capture itself. **Page identity, dedup and every stage still derive from the primary frame alone**; variants only ever add evidence about text the primary could not resolve.

### Focus, watched in the image (`focus.py`)

Everything ocrmd knew about focus came from AVFoundation, and on the camera it was built for both signals are inert. Measured on a Logitech C525 over 599 frames including two forced mode changes: **`isAdjustingFocus` was false on every one**, and `lensPosition` reads 0.0 and never moves. The focus gate has therefore never fired on this rig, the HUD's `FOCUSING` state has never been shown, and the focus bracket reports itself unavailable.

The failure that matters is real. The same book under the same lamp, minutes apart, measured page sharpness of **18 and then 1030** — and in the bad state the camera reported `focus=continuous` and stayed there for the twenty seconds it was watched, through a `locked`→`continuous` toggle. A scan started in that state photographs the whole book out of focus, because the shutter's floor was relative to the best *that session* had shown and decays (`sharp_decay`): a session that starts soft quietly lowers its own bar. In the store that shows up as crop blur of 149 median for pages that ended merely flagged, 101 for `suspect`, 66 for `illegible`.

So the lens is judged from the sharpness trace, which every camera has:

- **Moving or still**: the relative spread of page sharpness over a 12-frame window. On a still,
in-focus scene that spread measured p50 0.012, p99 0.026, max 0.027 over 598 frames, so `HUNT_BAND = 0.06` is a little over twice the worst quiet frame. A step down in sharpness also lands inside the band for one window, which is correct — something just changed, wait for it.
- **Sharp or soft**: sharpness against the best this *rig* has ever produced, remembered per camera in
`.ocrmd-camera.json` beside its view. Over 20 s of a working rig the ratio never fell below 0.939; the broken state was 0.017. `SOFT_RATIO = 0.45` sits between them with room for a genuinely low-contrast page.
- **Then something is done about it**: after `SOFT_FRAMES` (~1.5 s) the watch asks for a retrigger —
dropping the focus mode and putting it back, which is all a camera without lens control offers — and escalates to restarting the stream. A retrigger is cheap and harmless, which is the point: a mostly blank page can trip the softness test and the worst that costs is one re-hunt.
- **And the memory is allowed to fall.** A rig moved further from the page really is less sharp, and a
memory that refused to believe that would block every capture forever. After four failed recoveries the watch accepts the new normal, says so on the status line, and carries on.

Two smaller fixes came out of the same measurements. `camctl.focus_state` was called once per frame and cost **6.7 ms** — a fifth of the frame budget at 30 fps — because every call built an AVFoundation discovery session; the device is now cached and it costs 0.003 ms. And a scan that ended with the lens locked (`focus=settle`) left it locked: this rig comes back up `locked` after every such session, which is exactly how a lens ends up parked somewhere useless. The mode the session opened with is now restored on the way out.

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

### Admission: should a capture become a page at all?

Every capture used to become a page. On the store in this repository that is 139 pages, of which the model called 88 of the 120 it judged illegible — 73 % — and each one cost an LLM call to discover.

Scoring the *image* does not work, and was measured before being discarded. Against the model's own `legible` verdict on the real crops: page width AUC 0.63, x-height 0.46, edge sharpness 0.58, ink/paper contrast 0.57, Laplacian variance 0.62, line-spacing gradient 0.64. Nothing above 0.64. A crop that looks fine by every cheap measure can still be unreadable.

Asking the OCR engines does work, and they have already read the page by then. On the same 120 pages, `words < 60` refuses 21 captures of which 20 (95 %) the model independently called illegible, at the cost of one legible page; `confidence < 0.5` refuses 10, 9 of them illegible. Both are the shipped defaults (`admit_min_words`, `admit_min_confidence`), against a 73 % base rate.

Replayed over the whole store, admission refuses 23 of the 139 pages — 91 % of the ones the model had judged were called illegible — and saves those 23 LLM calls. **It is not a fix for a badly set up rig**: the illegible share among the pages that would still be made only falls from 73 % to 69 %, because most bad pages here have plenty of readable words and fail on framing and focus instead. It removes the captures that were never going to be pages; framing advice and the HUD are what address the rest.

Rejection is not deletion. The raw frame stays, the `captures` row records the reason and the document, no page, page file or LLM call is created, and `ocrmd status` lists what was refused. The test applies only to *automatic* captures: a file the operator ingested, or a frame they forced with the space bar, is a decision rather than a guess the shutter made. Re-ingesting a refused frame clears the refusal and admits it (`store.clear_rejection`), which is what makes `ocrmd ingest raw/<sha>.jpg --doc <slug>` the override — without it the refusal row itself reads as an exact duplicate and the override silently does nothing.

That override is load-bearing because the corpus contains no genuinely empty pages, so there is no measured way to tell "a blank page" from "a page we could not read" — both read as no words. Rather than invent a threshold, a blank page is refused like any other unreadable one and costs one command to put back.

## 6. Pre-processing

Ordered, each step recorded in `processed/<id>.json`:

1. Page detection: bright blobs (paper against a darker desk) merged when comparably sized (a deep
gutter shadow splits a spread into two blobs) → `minAreaRect` → perspective warp. Contour-based quad detection was tried first and failed on real photographs.
2. Clipped edges: frame edges the sheet touches *and where the edge strip carries ink* — bare
   geometry cried wolf on nearly every capture.
3. Orientation cascade, strongest evidence first: text-line-frequency power of the row profile
(`{0,180}` vs `{90,270}`); dictionary hits from a fast, non-auto-orienting OCR pass; ink asymmetry (ascenders outnumber descenders) as the never-silent tie-breaker.
4. Two-page spread split at the **gutter, found by absence of ink** — and only when there is one.
5. Denoise + adaptive threshold copy for engines that want it (Tesseract). Apple Vision and the LLM
   get the colour crop.
6. The **flat rendering** (`processed/<id>.flat.png`) — the page as a flatbed scanner would have
   delivered it. Derived from the crop, not a stage: see below.
7. Framing advice: text cut off at an edge, or fewer than `min_page_width` px across the page. Both
   are shown live in the HUD, written to the page, stored on the `pages` row, and listed by `status`.

#### Finding the gutter

The gutter used to be "the darkest column in the middle third", and on the 60 split captures in this store that put the split line on ink 0.109 where the page body measures 0.117 — it was cutting straight through text, and 40 of the 60 would have been split somewhere else. Under a desk lamp the darkest column in the search band is simply the far edge of the band, which is why 18 of them split at almost exactly 1/3 or 2/3 of the sheet and produced a page and a 440-px offcut.

A binding is not dark, it is *empty*. `ink_columns` measures fine detail per column as high-pass energy — the frame minus a heavily blurred copy of itself — which ignores exposure and survives blur. A threshold does neither: measured against a page softened until OCR reads a third of it, an adaptive threshold keeps 13 % of the signal and a Sobel magnitude 16 %, against 39 % for the high pass, and all three read blank paper as zero. Under a threshold a spread with one page out of focus stops looking like a spread at all.

`find_gutter` then requires three things, each of which was put there by a case that failed without it:

- a **trough**, `GUTTER_DEPTH` (0.30) below the page body's detail. Calibrated, not guessed: every one
of the 60 captures was read whole and split and scored on expected correctly-read words (words × mean confidence, the only fair comparison of one page against two). Splitting genuinely helps 36 and hurts 24; depth separates them at AUC 0.797 (medians 0.527 against 0.225) and 0.30 maximises the net gain — +2652 words against +2161 for splitting everything and +0 for splitting nothing.
- **text on both sides**, within `GUTTER_BALANCE` (0.25). Emptiness alone is not a binding: a ragged
right margin, a wide figure or a column break leaves just as empty a stripe, and cutting there turns one page into a page and an offcut. The ratio is deliberately loose, because uneven light across a spread is the commonest complaint on this rig and can leave the far page carrying a quarter of the near page's ink — it has to reject an *empty* side, not a dim one.
- the split placed at the **middle of the blank band**, not at its lowest point. The band is the
binding plus two inner margins, which are rarely the same width, so the minimum sits off to one side; the midpoint keeps the most clearance from text on both pages and does not move with noise.

`estimated_page_width` runs the same test, or the HUD would promise 900 px a page on a sheet the pipeline is about to keep whole.

#### The flat rendering (`preprocess.legible`)

`processed/<id>.png` is the crop as photographed: uneven light, grey paper, soft type. `<id>.bin.png` is the adaptive threshold, which is what an operator looking for "the enhanced view" has had until now. `legible()` is the third thing, and the one meant to be *read*: divide out the illumination, unsharp mask, tone curve, greyscale out.

Scored on the 40 pages of this store that carry a model transcription, the way `ocrmd ocr-bench` scores an engine:

| view | F1 | recall | precision |
|---|---|---|---|
| colour crop (what the engines get) | 0.744 | 0.751 | 0.765 |
| **flat** | 0.741 | 0.765 | 0.735 |
| `bin` | 0.246 | 0.222 | 0.320 |

Three things that measurement decided:

- **No denoising.** A bilateral filter before the unsharp mask makes the page look cleaner and costs
  0.144 F1 (0.597 against 0.741). At webcam x-heights the "noise" it smooths away is the thin strokes.
- **Otsu classes, not a fixed percentile.** Ink coverage runs from ~1 % on a chapter opening to ~15 %
  in a dense table, so a fixed percentile is sometimes the ink and sometimes the paper — and on the sparse page it lands on paper, which is the page that needed the help most.
- **...but not the class medians.** Mapping median-ink to black and median-paper to white flattens half
  of every stroke: 0.693 against 0.741 for the 10th/40th percentiles of the two classes.

`bin` is kept because it is what some engines historically wanted, and because a threshold is sometimes the clearer thing to look at. Neither it nor the flat rendering is read by the pipeline: Apple Vision, RapidOCR and the model all get the colour crop, and Tesseract flattens its own input (`ocr.flatten`).

`legible()` costs 33 ms on a real crop, measured over all 194 in this store, and it is **derived rather than staged** — no version string, no part in the page id, no `reprocess`. `pipeline.ensure_flat` writes it beside the crop the first time anything asks for it, so pages scanned before it existed get one for free; delete the file and the next request rebuilds it.

`PIPELINE_VERSION` (`preprocess@4`) is bumped whenever the crop changes; page ids are the hash of the processed image, so `reprocess` re-derives pages from raw when it does.

**Variant crops.** Each focus variant is cropped by the same code with two constraints (`pipeline.derive_variant_crops`): the page's *own* orientation is forced rather than probed (`process(..., orientation=)`), because a variant that probed to a different rotation could not be compared with its page at all — and the probe is the expensive step here, since it runs an OCR pass; and the result is scaled onto the page crop's exact pixel dimensions so normalised boxes are directly comparable. A variant whose own page detection landed somewhere materially different has a different aspect ratio (> 4 %) and is *dropped* rather than stretched into place: a mis-cropped variant would line its text up against the wrong lines, which is worse than having no second reading.

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

### Cross-focus merge (`multifocus.resolve`)

The same idea one axis over: `ocr.merge` votes across *engines* reading one image, `multifocus.resolve` votes across *exposures* of one page. Each variant crop is read with the same recogniser as the page — every configured engine, the same preparation, the same merge — because the swap compares its confidences with the page's own, and two numbers produced by different procedures are not comparable.

For each of the page's lines, the variant blocks overlapping it (`ocr._same_line`) are joined left to right at the *minimum* of their confidences, and:

- a line at or above `ocr_trust` (0.9) is never second-guessed;
- below it, the most confident variant reading that is **acceptable** replaces the line, if it beats
it by `focus_swap_margin` (0.10). Acceptable means about the same length (0.7-1.4×, `merge`'s rule) **or** substantially longer *and* ≥ `CONFIRMED` (0.9). A shorter reading never qualifies — that is text being dropped;
- a line only a variant saw is **filled in** at ≥ `CONFIRMED` with ≥ 2 words (stricter than `merge`'s
  0.85, because variant crops carry more geometric jitter than two engines on one crop);
- one variant line can be spent on only one of the page's lines, so jitter can never write it twice;
- the page's box is kept after a swap, so reading order cannot be disturbed by the swap itself.

The longer-reading rule is the one `merge` does not have, and it is the load-bearing one. Two engines reading one image disagree word by word; two focus settings do not. Measured on a 520-px page with its top half blurred, every doubtful line came back as a *fragment* — `"milies"` where the page reads `"is so well fixed in the minds of the surrounding families"` — and a symmetric length window rejected all of them, i.e. exactly the captures the bracket exists to rescue.

Every variant is read, including when the page's own lines are all confident. Gating on confidence was tried first and is wrong:

| top-half blur | page alone | bracketed | what fired |
|---|---|---|---|
| 5 px | 165 / 165 | 165 | nothing |
| 7 px | 117 / 129 | 138 / 148 | 2 replaced |
| 9 px | 101 / 119 | 146 / 164 | 4 filled |
| 11 px | 64 / 82 | 165 / 165 | 4 replaced, 6 filled |
| 13 px | 55 / 55 | 165 / 165 | 10 filled |

(correct words / words read, against a 165-word page.) The two largest gains — 9 px and 13 px — are on captures the primary read at ≥ 0.98 mean confidence: it had simply not emitted the unreadable lines, and **confidence says nothing about a line an engine never produced**. The cost is one OCR pass per variant, ~1.2 s with two engines, on the worker thread, for pages the shutter was already unhappy enough about to bracket.

What survives is the point of the exercise. A line no focus read confidently stays in the page at its own low confidence *and* is recorded in `ocr/<id>.focus.json` as an `Unresolved` region with every candidate reading and which exposure produced it. The pipeline never picks the highest number.

### Handing the bracket to the model

When a page has unresolved regions, `pipeline.focus_views` attaches the `focus_llm_images` (2) variants holding the best candidates for them, plus `multifocus.describe`'s region list, to the LLM request. Each image is introduced by its own text block (`llm.content_blocks`) saying it is *another view of the same page* — an unlabelled second image reads as the next page of the book, and a model that believes that merges two readings of one page into one transcription, which is the exact fabrication this pipeline exists to catch. The region list gives fractional positions and labels the candidates as low-confidence machine guesses, with the instruction to transcribe what is actually visible or, failing that, report it in `uncertain` rather than choosing between them. Pages with nothing unresolved send exactly what they always did, and pay nothing.

The OCR stage version is `ocr@3:<engine names>`: a new engine set re-reads a page the next time it is processed (`verify` includes such pages; local, free), and the LLM stage is reused from cache.

**OpenCV is pinned below 4.12.** `opencv-python 5.0.0.93`'s KleidiCV HAL segfaults in `cv2.resize` on a real crop in this repository (845×1505 → 544×960, arm64 macOS), reproducibly, on the main thread and single-threaded. Every stage resizes.

`<5` turned out not to be enough. KleidiCV ships in **4.12+ too**, and 4.14.0.94 crashes on the wider crops the current spread split produces — the C stack named `kleidicv_resize_generic_stripe_u8`. It is a hard process death with no Python traceback, so it reads as a hang or a silent exit rather than an exception, and it took three wrong hypotheses to find (RapidOCR thread-safety, image size, ONNX/OpenCV thread pools — each measured false and reverted). `getBuildInformation()` is the test: 4.14.0.94 contains 14 "kleidicv" strings, **4.11.0.86 contains none**, and it is the last release that does. The pin is `opencv-python>=4.9,<4.12`.

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

- `claude-opus-5-5` default; `output_config.effort = llm_effort` (medium — perception, not reasoning)
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
processing: { pipeline: preprocess@4, page_detected: true, clipped_edges: [], orientation: 0,
              part: left, deskew: 0.0, blur: 417.0, glare: 0.0, page_width_px: 936,
              advice: null, continuity: "printed page jumped 37 → 41 — 3 page(s) may have been missed" }
ocr:  { engine: apple_vision@26.5, confidence: 0.97, artifact: ocr/7f3c….json,
        focus: { variants: 2, improved: 3, filled: 1, unresolved: 0,      # only on bracketed pages
                 artifact: ocr/7f3c….focus.json } }
llm:  { provider: anthropic, model: claude-opus-5-5, prompt: transcribe@4, request_id: msg_…,
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
history: [ { stage: preprocess, version: preprocess@4, completed_at: … }, … ]
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

## 11a. Reconstruction: the round trip

The pipeline reads a photograph into text. `reconstruct` reads the text back into a page, which is the only way to see what the transcription *lost* — a heading flattened into body text, a table turned into prose, a figure nobody mentioned. `ocrmd reconstruct --doc X` writes three artifacts per page and one number.

**Layout** (`layout.py`) is the missing join. Local OCR knows *where* every line was (`Block.bbox`) but reads it imperfectly; the LLM pass knows *what* the page says but returns reflowed Markdown with no geometry. `layout.build` aligns the two with the same `SequenceMatcher` walk `reconcile` uses, so each line carries the reconciled words at the measured position. `replace` spans are adopted as well as `equal` ones — a corrected word differs from the OCR word by definition — but only when the two texts are readings of the same page at all (`MIN_MATCH`), or an unrelated document's words would be stamped onto this page's boxes. Lines the merge duplicated are collapsed (`dedupe`): two engines box one row differently, which is invisible in a reflow and draws them on top of each other in a facsimile.

**Figures** are what is left when the text is subtracted from the page's ink. The trap is that erasing line *boxes* leaves the gaps between them, and closing those gaps joins the whole column into one blob — which then gets pasted back over the reconstruction as a photograph of the page. So a candidate is rejected if it lies inside the rows the text block occupies, or if it covers more than `MAX_FIGURE` of the page.

**Outputs.** LaTeX is the portable master — absolutely positioned with `textpos` at the page's true size, so it is a facsimile rather than a reflow, and it compiles anywhere a TeX distribution exists. SVG is the same layout with no TeX needed, which matters because neither the development machine nor CI has one; it renders through Inkscape, `rsvg-convert` or headless Chrome. Markdown stays the human-editable source.

**Graphics are vectorised where that measurably worked.** A line drawing traces to SVG paths through `cv2.findContours`; the trace is then rasterised and compared with the crop it came from, and anything below `VECTOR_FIDELITY` falls back to the embedded raster. A photograph is never traced. "Where possible" has to mean "where it was checked", not "where it was attempted".

**Fidelity** is a correlation of the two pages as *ink*: both sides are high-pass filtered and normalised before comparing, because the reconstruction is clean black on white and the original is a photograph under a desk lamp — correlating raw pixels would score the lighting. `--review` additionally sends both images to the model and asks what the rebuild got wrong, because a pixel score says how much differs, not what.

## 11b. Annotation sync: the paper stays the document of record

`ocrmd annotate --doc X` puts the camera back on pages that have already been scanned. Underline a phrase, run a highlighter over a sentence, write a note in the margin, and the page file picks it up.

    identify  →  align  →  diff  →  classify  →  apply

**Only the last step ever costs anything.** By this point the page has been read once and every line's position is on disk, so an underline or a highlight is resolved by geometry alone — the words under the mark are found by intersecting it with stored line boxes, and no recognition of any kind runs. Handwriting is the one thing on the page that has never been transcribed, so a *note* costs one LLM call (`Provider.read_handwriting`); local OCR is not asked, because it is trained on print and reads cursive as noise. A page nobody has written on costs a subtraction on a 900-px greyscale and stops there.

- **Identify** reuses the deduplicator's two steps — dHash prefilter, then correlation against the
  stored capture — because it is the same question asked for the opposite purpose.
- **Align** is ECC (`MOTION_EUCLIDEAN`) on plain greyscale. High-passing first was tried and is
measurably wrong: ECC's correlation coefficient is already invariant to brightness and contrast, and removing the low frequencies takes away the structure it converges on. On a real page nudged 5 px and 0.6°, greyscale registers at cc 0.99 and high-passed input at 0.38, recovering the shift as 21 px instead of 3.
- **Diff** uses two channels, because pens and highlighters do opposite things. A pen darkens paper; a
highlighter colours it while leaving it bright and barely registers in luminance at all, so it is found as a rise in saturation instead.
- **Classify**: a long thin rule through the middle of a line strikes it out, at or below the baseline
underlines it, and anything else is handwriting. "Anywhere inside the line box" is not the test — OCR boxes carry descender space, so a rule drawn just under the text still falls inside one. A rule that passes under existing print adds no darkness there and arrives as a row of fragments, so the pen mask is closed horizontally (`RULE_JOIN`) before blobs are found.
- **Apply** writes `==highlight==`, `**underline**`, `~~strike~~` and appends notes under their own
heading. A phrase already marked is left alone, so a camera resting on one page does not wrap it again and again. The page is written *without* refreshing the fingerprint — what someone wrote on the paper is a human edit and must survive the next `verify`.

Two refusals matter. A diff covering more than `MAX_CHANGE` of the page is rejected as a misalignment or the wrong page rather than applied, and blobs against the page edge are warp artifacts rather than annotations. Editing a page from a bad alignment is exactly the silent corruption the rest of the pipeline exists to prevent.

## 11c. Reading one region harder

`pipeline.read_region` — a box dragged on the page image in the web UI, `ocrmd read --region x,y,w,h`, `POST /pages/{seq}/region`. The operator has found a word the pipeline got wrong and is asking what it says.

The selection is cut out of the **flat rendering**, padded by `REGION_PAD` so nothing is clipped, and enlarged on its own terms: `ocr.upscale_for_ocr` aims at the same 1400-px short side it aims at for a page, but a region is a fraction of a page, so the same target buys several times the glyph height (`REGION_MAX_SCALE = 6`, against a page's measured cap of 3). The configured engines read that, and — only when asked — the model reads it too, with the OCR draft as a hint (`Provider.read_region`, `llm.REGION_SYSTEM`: transcribe what is inside the crop, mark what is unreadable `[illegible]`, do not continue the sentence past the edge, do not describe the image).

**What reading by region actually buys.** Measured on the 38 pages of this store that carry a model transcription, by cutting each page into four bands, reading each band as a region and scoring the concatenation the way `ocrmd ocr-bench` scores an engine:

| | F1 | recall | precision |
|---|---|---|---|
| whole page, as the pipeline reads it | 0.740 | 0.747 | 0.764 |
| four regions, colour crop, cap 3 | 0.738 | 0.770 | 0.728 |
| four regions, flat rendering, cap 6 | 0.732 | 0.778 | 0.704 |

F1 barely moves; **recall rises 2–3 points and precision falls by about as much**. That is the honest shape of this feature: reading a piece of a page on its own finds more words and is less choosy about them, which is the right trade when the person who asked is looking at the same crop and can see whether the answer is nonsense — and the wrong trade for the unattended page pass, which is why the pipeline still reads whole pages. These are also pages that were *already* read well; the operator reaches for this on the ones that were not.

**Cost.** ~0.9 s for a line and ~1.3 s for a paragraph with both engines. The enlargement cap makes no measurable difference to either the words read or the time (0.93 s vs 0.94 s at 3× and 6× on a line, 29.8 words both), so 6 is kept for the small selections the measurement has no examples of rather than because it was shown to help.

**It writes nothing.** No stage is marked, no artifact is rewritten, and the answer is handed back beside its own crop so the person who asked can see what the engines were given. The model's reading is returned *next to* the OCR one rather than replacing it: everywhere else in ocrmd two readings are reconciled and their disagreements are marked, and this path has no second reading — so the operator is the reconciler. The UI's **Insert into body** puts the text at the cursor in the editor and nothing reaches the page file until Save, which then counts as a human edit like any other.

## 11d. Extraction: DocETL's operators over the finished pages

A transcribed book is a corpus, and `ocrmd etl` answers the question that follows: *what is in it?* The operator model is [DocETL](https://github.com/ucbepic/docetl)'s (UC Berkeley EPIC lab, MIT) — `map`, `filter`, `extract`, `reduce`, `resolve`, plus `split`, `unnest` and `sample` that cost nothing — each an LLM prompt and an output schema, chained in a YAML spec that is deliberately close to theirs.

**The implementation is not theirs, and that is the design decision.** DocETL brings its own LLM stack (litellm, its own model configuration, its own disk cache, its own accounting) and about thirty runtime dependencies. ocrmd already has a provider layer, credentials, token accounting in `llm_calls`, and one governing idea — every unit of work writes an artifact and a version, so re-running is free. Six operators over `Provider.complete_json` inherit all of that; a vendored second LLM stack would duplicate it. What DocETL has that is genuinely hard to reproduce is its *optimizer*, which rewrites a pipeline (decomposing an overloaded map, inserting a resolve before a reduce, swapping models), so `--export-docetl` writes the same spec plus the rows as a dataset in the form their runner takes, and the README says how to run it.

Three things belong to ocrmd rather than to the operator model:

- **Provenance.** `_document`, `_page` and `_page_id` are attached at the source and merged through
every operator; a reduce keeps `_pages`. Output writers put them first. An extraction you cannot take back to the photograph it came from is a rumour, and this repository exists because a model once invented a table of contents.
- **The same refusal as the transcription stage.** `etl.SYSTEM` is explicit that the text is the whole
of the evidence, that a field the text does not answer is empty, and that doubtful readings keep the `==[...]==` reading rather than being "corrected".
- **Caching that matches the rest of the pipeline.** Each operator has a cache file keyed by a hash of
its prompt, schema, key and model (`etl/<slug>/<op>.json`); change one operator and one operator re-runs. `--dry-run` renders the prompts and reports the call count without sending anything.

Two smaller decisions worth keeping: templating is `{{ input.field }}` and `{{ inputs }}` and nothing more (a template language in a spec file is a way to run code from a spec file, and a missing field raises rather than expanding to an empty string that silently changes the question), and `resolve` blocks candidate pairs with `rapidfuzz` before asking the model, because all-pairs comparison is quadratic in rows and most pairs are obviously unrelated. Blank and illegible pages are not offered to operators at all — an operator asked to find something in an empty page is being invited to invent it.

The web UI reads specs from `documents/<slug>/etl/*.yaml` and `<root>/etl/pipelines/*.yaml`, by file name only: a browser must not be able to point the runner at a YAML file elsewhere on the machine.

## 11e. Destinations: a page is handed to another system

`ocrmd/sinks.py` — the **Send…** button on the page view and in the region readout, `POST /pages/{seq}/send`, `ocrmd send --doc X --page N --to NAME`. The one **outbound** path in this repository: everything else ends at a file on disk, and this hands a page to a receiver of the operator's choosing.

It is generic on purpose. The first receiver built against it is [law-ai](https://github.com/bamr87/law-ai)'s LSAT intake, and the 0.8 draft of this path was a law-ai module inside ocrmd — `lawai_url`, a "→ LSAT" button, section and difficulty pickers. That is the wrong shape for a scanner: ocrmd should know nothing about LSAT questions, and the next receiver (a matter in the same platform, a knowledge base, a notebook) would have wanted its own module, button and settings. So the receiver now describes itself and ocrmd draws what it says:

- `GET url` → the **descriptor**: `{"ocrmd_sink": 1, "name", "description", "accepts": ["page", "region"],
  "options": [{"key", "label", "type": string|enum|integer|number|boolean, "choices", "default", "help"}]}`.
  The CLI prints it, the web UI renders one input per option (a select for an enum, a checkbox for a boolean), and `coerce_options` types and checks the operator's values against it *before* anything leaves — a value outside an enum is refused here, because the receiver's 422 would name a field the person never saw. A receiver with no descriptor still works (the send has no options); one that is down is listed as down.
- `POST url` → the **envelope**: `{"ocrmd": 1, "kind": page|region, "text", "provenance", "document", "options"}`.
  `text` is the page body as it stands on disk — the reconciled reading *including* any disagreement the operator has resolved. `provenance` is what the receiver needs to find the page again and to judge the reading: tool and version, document, page, page id, printed label, capture time, OCR engine, model, both agreement scores, the flag count, the count of unresolved marks in the text sent, the status, a URL to the image and one to the page in this UI, and the `bbox` when the send came from a region. `document` is the catalogue entry from document.yaml. The receiver answers 2xx JSON, ideally `{"summary", "items": [{id, title, detail, status}], "warnings"}`; anything else is shown raw with an honest synthesized summary.

Everything else follows from that division of labour:

- **The text travels; the photograph does not.** A page image and a vision model would let the receiver
do its own single-pass read — which is exactly the thing ocrmd exists to distrust. The page id is the sha256 of the processed image, so it outlives the URL and the process.
- **This is a send, not a stage.** Same contract as `read_region`: no file is written, no stage marked,
no version moved. It is safe to press twice, because a well-behaved receiver deduplicates against its own store and says so — which is how a person actually checks "did that go through?".
- **A region's reading is a better send than a page's.** The region tool already re-reads a box at
several times the page's glyph height; the readout's own **Send…** sends *that* text with the `bbox` recorded, which is what a person boxing one item out of five on a page wants.
- **Failure is reported, never retried into silence.** `SinkError` carries the receiver's own reason;
the web route answers **502**, not 500, because this server is fine and the page simply did not reach the other one.
- **Off unless configured.** No `[[sinks]]` table (or `OCRMD_SINKS`) means no button, no command, no
path — these are the only settings in ocrmd that send page text anywhere other than the transcription provider. Credentials (`api_key` → `X-API-Key`, `bearer` → `Authorization`) never reach the browser: `/api/settings` masks them.
- **A Python receiver** registers through the `ocrmd.sinks` entry-point group instead of a URL:
  anything with a `name`, a `describe()` and a `send(envelope, options)`, constructed with the `Settings`.

**What the far side does with it** is the receiver's business, but two of law-ai's choices are load-bearing here and were built alongside this: its extractor is told to leave a field empty rather than fill in what a page of this kind would probably say (the same refusal `etl.SYSTEM` runs on), and every draft it produces is scored for **grounding** — the share of the draft's content words that actually occur in the page text, taken as the *minimum* over the stem and the five choices, so one invented choice on an otherwise faithful transcription cannot be averaged away by the four real ones. Below 0.75 the draft is dropped before a reviewer ever sees it wearing a page number. Nothing from that path can reach its live question bank; a human approves it or it stays pending.

## 11f. The inventory: catalogue and search

`ocrmd/library.py` — `ocrmd library`, `ocrmd search`, `ocrmd index`; `GET /api/library`, `GET /api/search`; the Library view and the Search view in the web UI.

The store already knew every page and every stage; what it did not know was *which book*. That lived only in `document.yaml`, which is the right home for it (a person edits it, the pipeline never overwrites it) but the wrong place to answer "what have I scanned" from another tool. `sync_document` mirrors the catalogue fields (title, author, kind, year, ISBN, publisher, tags, project, citation key) onto the `documents` row whenever the file is newer than the last mirror — one `stat` per document per catalogue read, a YAML parse only when something changed — so `ocrmd.db` alone is the inventory. A year written as prose (`1994 (2nd printing)`) still becomes a number; a YAML with a typo is logged and the row keeps what it had.

Search is a `page_text` table of every page's body plus an FTS5 virtual table when SQLite has the module (every python.org and Homebrew build does); without it, `Store.search` falls back to a substring scan with an excerpt cut around the match, so the feature degrades to slower rather than to absent. The user's search box is turned into an FTS5 expression by `fts_query`: every word quoted (so `18-3-102` is a token, not a NOT), ANDed, `word*` kept as a prefix, a quoted query passed through as the phrase it is.

The load-bearing decision is **when the index is updated: on read, by file mtime**. Page files are written by the pipeline, the web editor, `/resolve`, sign-off, `ocrmd review`, the annotation sync, an extraction write-back and any text editor; a hook in each of those would be one more thing every future write path has to remember. Instead `reindex` walks the pages of the documents a search touches, compares each file's mtime to the one the index was built from, and re-reads only what changed — ~300 `stat` calls for a book, which is nothing beside the search itself. Pages that disappear (deletion, `reprocess`) leave the index through `Store.delete_page`. `ocrmd index --force` is for after a restore.

## 11g. Exports and the embedding API

`ocrmd/export/` — one module per shape behind `export_document(s, st, slug, fmt)`: `md` (the built document, regenerated then copied), `txt` (plain text with a separator line per page; the disagreement marks kept visible as `ambitious[?ambiguous]` unless a side is chosen on purpose, because a text export has no footnote to keep the refusal in), `json` (the whole document as one object, with layouts and extraction rows), `jsonl` (one object per page carrying identity, verdict and text — the row a search index or a RAG loader wants). Every export keeps the page ids and is recorded in `doc_exports`, so the inventory can say which shapes a document already exists in.

`ocrmd/api.py` — `Library`, the third driver over the same modules the CLI and the web UI use. It owns a `Settings` and a `Store` of its own rather than the CLI's module-level singleton, so a host can point it at any root and hold several; it takes the provider as an argument (`"auto"`, `None` for OCR-only, or any `llm.Provider`), which is what lets a host with its own model-routing layer keep every model call inside that layer rather than have a dependency call a vendor directly; and it never prints — progress goes to an `emit` callback. `ingest`, `ingest_image`, `verify`, `build`, `page`, `pages`, `document(s)`, `export`, `search`, `envelope`, `read_region`, `send`, `delete_*`, `doctor`. law-ai's plugin is written against exactly this surface with `provider=None`, because its provider chain has no vision path yet and a dependency calling Anthropic directly would bypass its audit layer.

## 12. State and storage

```
<root>/                          (OCRMD_ROOT; default: the project directory)
  ocrmd.db                       SQLite, WAL
  ocrmd.log                      rotating log
  ocrmd.toml                     optional configuration
  raw/<sha256>.jpg               immutable captures (including ones refused as unreadable)
  raw/<sha256>.v<n>.jpg          ...and the focus bracket's other exposures of the same capture
  processed/<id>.png|.bin.png|.json
  processed/<id>.flat.png        the flat rendering; derived from the crop, rebuilt on demand
  processed/<id>.v<n>.png        variant crops, on the page crop's own geometry
  ocr/<id>.json|.focus.json
  llm/<id>.<provider>.json
  documents/<slug>/document.yaml, pages/*.md, <slug>.md
  documents/<slug>/etl/*.yaml, extracted/*   extraction pipelines and what they produced
  export/<slug>/<slug>.{md,txt,json,jsonl}   exports (`ocrmd export`), recorded in `doc_exports`
  etl/<slug>/<operation>.json    per-operator answer cache, versioned by the prompt
  .ocrmd-camera.json             per-camera view, and the best sharpness that camera has reached
```

Tables: `documents` (slug, created_at, and the catalogue mirrored from document.yaml: title, author, kind, year, isbn, publisher, tags, project, citation_key, meta_mtime), `pages` (id, document, sequence, hashes, status, page_label, part, width_px, advice, written_fingerprint, reviewed_by/at), `captures` (every frame incl. duplicates and refusals: `duplicate_of`, `rejected`, `document`), `stages` (page_id, stage, version, completed_at, error), `llm_calls` (tokens, latency), `batches`, `exports` (per-page facsimiles), `etl_writes`, `page_text` (page_id, document, sequence, page_label, status, body, mtime) with the `page_fts` FTS5 table beside it when available, `doc_exports` (document, format, path, at). Indexes on (document, sequence), (document, status), captures.page_id, llm_calls.page_id, page_text(document, sequence).

Stage versions: `preprocess@N`; `ocr@N:<engines>` (a changed engine set re-reads on the next `verify`, locally; the cross-focus merge is part of it); `transcribe@N:<model>` for the LLM pass, so `verify --refresh` re-runs pages made by an older prompt *or* a different model and nothing else does.

**Migrations** are additive (`COLUMNS` in `store.py`): `CREATE TABLE IF NOT EXISTS` never adds columns to an existing table, so every column added after 0.1 is applied by an explicit `ALTER TABLE`. Old pages get `width_px`/`advice` backfilled from `processed/<id>.json` by `status`/`doctor`.

## 13. Concurrency and failure

- **Three lanes during a scan.** The capture loop (main thread, owns the window) only enqueues
frames. One worker thread does ingest/preprocess/OCR — Vision is CPU-bound and serial, and a bracketed page costs one more OCR pass per variant — and renders an OCR-only page immediately. A `ThreadPoolExecutor(llm_workers)` runs the LLM pass and re-renders when it lands. Queue depth (frames + pending LLM jobs) and the active stages are in the HUD.
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
- **Jobs.** verify/build/reprocess/ingest/reconstruct/etl run on one worker thread (OCR is serial, the store is one
connection), each with a bounded log, progress and a cancel flag; `reprocess` is refused while its document is being scanned. Uploads land in `<root>/inbox/<slug>/` and go through the same ingest.
- **Deletion** (`DELETE /pages/{seq}`, `POST /pages/delete`, `DELETE /documents/{slug}?confirm=slug`)
removes rows, artifacts and page files, and purges the raw capture once no page refers to it. Its capture rows go too, so the same frame can be scanned again without tripping dedup. Sequence numbers are not renumbered (page files are named by sequence and printed labels carry the order). Refused while the document is being scanned.
- **Edits are human edits.** `PUT /pages/{seq}` and `/resolve` write the page file *without* updating
the pipeline fingerprint, so the file counts as hand-edited and `verify` preserves it (§10). `/signoff` records the reviewer and freezes the body; `/reopen` reverts to the computed status.
- **The page image is four views and a selection tool.** `flat` (the default), `processed`, `bin` and
`raw`; `flat` is rendered on the first request for a page that predates it, since it is a pure function of the crop. Dragging a box on it calls `POST /pages/{seq}/region` (§11c). The rectangle is kept in *page fractions*, not pixels, so it survives zooming, switching views and the re-render after each read.
- **Frontend.** ~800 lines of vanilla JS with a hash router (dashboard, scan, document, page,
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
ocrmd annotate  --doc SLUG [--camera N|name] [--no-llm] [--dry-run] [--duration S] [--max-pages N]
ocrmd reconstruct --doc SLUG [--page N] [--out DIR] [--pdf] [--review]
ocrmd read      --doc SLUG --page N [--region x,y,w,h] [--llm] [--ask Q] [--save PATH]
ocrmd etl       --doc SLUG -p PIPELINE.yaml [--limit N] [--status S]... [--dry-run] [--no-cache]
                [--no-llm] [--out PATH] [--export-docetl DIR]
ocrmd status    [--doc SLUG] [--all]
ocrmd build     [--doc SLUG]
ocrmd library   [--json] [--all]
ocrmd search    QUERY [--doc SLUG] [--limit N] [--json]
ocrmd index     [--doc SLUG] [--force]
ocrmd export    [--doc SLUG] [--format md|txt|json|jsonl] [--out PATH]
ocrmd sinks     [--no-probe]
ocrmd send      --doc SLUG --page N [--to NAME] [-o key=value]... [--region x,y,w,h] [--dry-run]
ocrmd doctor    [--costs]
ocrmd ocr-bench [--doc SLUG] [--engines a+b | --compare] [--limit N] [-v]
ocrmd serve     [--host 127.0.0.1] [--port 8765] [--no-open]
ocrmd --debug|--quiet <command>
python -m ocrmd <command>
```

## 16. Testing

`tests/` runs in about a minute on a Mac (`pytest`): synthetic pages rendered with PIL, a temporary root per test, a `FakeProvider` in place of the API. Covered: camera view keys and persistence; orientation, framing advice, clipped-text detection, spread splitting; the hand gate (detection, reference stability, moved book, non-hand intrusion timeout, page turn) and gate ordering; the store (0.1 migration, document-scoped dedup, thread safety, stage versions, batches); configuration precedence and validation; the pipeline end to end (OCR-only then LLM upgrade with OCR cached, reuse vs `--refresh`, human-edit and reviewer preservation, page-number continuity, error recording, illegible/suspect, duplicates, spreads with hints, batch round trip, re-deriving missing artifacts); build (inheritance, chapter carry-forward, TOC, gaps, needs_reshoot); review parsing/resolution; request shapes for both providers; the CLI (ingest/status/review/build/verify, degraded mode, interactive review); and the web API (documents, pages, images, resolve/sign-off/reopen, human edits surviving a re-run, document.yaml, jobs, upload ingest, scan start failure). `test_etl.py` covers the extraction layer (the type notation, a pipeline naming an operation that does not exist, a missing template field, provenance through map/unnest/reduce, resolve's blocking, the cache reusing and then invalidating, a dry run spending nothing, the DocETL export). `test_region.py` covers the flat rendering (a lamp divided out, ink and paper further apart, a blank crop not stretched into noise, still readable to an engine) and region reads (read on its own terms, only that region, the model asked with the OCR draft as its hint, a provider that cannot do it simply not asked, and nothing written but the flat cache). `test_library.py`, `test_export.py` and `test_sinks.py` seed pages straight into the store and the page files, so the inventory, the exports and the destination contract are tested without an engine: the catalogue mirror (and a YAML with a typo), search with FTS5 and with the substring fallback, an edited page re-indexed by mtime, a deleted page leaving the index; every export shape with its page ids, the text export keeping both readings; descriptors probed, cached and declined, options typed and refused before the wire, the envelope's contents and the photograph's absence, a dry run, credentials as headers, an entry-point sink, a 502 for a receiver that is down, and nothing written by a send. `test_api.py` drives `Library` the way a host program would. Tests needing the OCR engine skip where none is installed. CI runs on macOS.

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
