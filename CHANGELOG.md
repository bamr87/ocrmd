# Changelog

## 0.9.0 — unreleased: a general-purpose book scanner

### An inventory of everything scanned

`ocrmd library`, `ocrmd search`, `ocrmd index`; `GET /api/library`, `GET /api/search`; the Library and
Search views in the web UI. The store always knew every page and every stage but not which *book*:
that lived only in `document.yaml`. `library.sync_document` now mirrors the catalogue fields (title,
author, kind, year, ISBN, publisher, tags, project, citation key) onto the `documents` row whenever the
file is newer than the last mirror, so `ocrmd.db` alone answers "what has been scanned here" — for
ocrmd and for any tool that can read SQLite.

Search is full-text over every page's body: a `page_text` table plus an FTS5 index when SQLite has
the module, a substring scan with a hand-cut excerpt when it does not. The index is brought up to date
**on read, by file mtime**: page files are rewritten by the pipeline, the web editor, `/resolve`,
sign-off, `ocrmd review`, annotation sync, extraction write-backs and any text editor, and none of
them has to know an index exists. Every hit names its page and links to it.

### Exports in the shapes other tools take

`ocrmd export --doc X --format md | txt | json | jsonl`, and download links on every document in the
web UI. `md` is the built document; `txt` is plain text with one separator line per page and the
`==[…]==` marks kept visible as `ambitious[?ambiguous]` rather than silently resolved; `json` is the
whole document as one object (metadata, frontmatter, bodies, layouts, extraction rows); `jsonl` is one
object per page with identity, verdict and text — the row a search index or a RAG loader wants. Every
export keeps the page ids and is recorded in `doc_exports`, so the inventory says which shapes a
document already exists in.

### Destinations: the law-ai sink became a contract

0.8's draft of the outbound path was a law-ai module inside ocrmd — `lawai_url`, a "→ LSAT" button,
section and difficulty pickers. A scanner should know nothing about LSAT questions, and the next
receiver would have wanted its own module. `ocrmd/sinks.py` replaces it with a generic contract a
receiver implements: `GET url` answers with a **descriptor** (what it is, which typed options it takes),
`POST url` takes the **envelope** (the page's reconciled text, its provenance — document, page, page
id, both agreement scores, the flag count, a link back to the image — the document's catalogue entry,
the operator's options). ocrmd draws the options the descriptor declares, types and checks them
*before* anything leaves, and shows the receiver's `summary`, `items` and `warnings` when it answers.
`[[sinks]]` in ocrmd.toml (or `OCRMD_SINKS`) registers an HTTP receiver; the `ocrmd.sinks` entry-point
group registers a Python one. `ocrmd sinks`, `ocrmd send --doc X --page N --to NAME -o key=value`,
`GET /api/sinks`, `POST /pages/{seq}/send`, a **Send…** button on every page and in every region
readout. Everything that was true of the 0.8 path still is: the text travels and the photograph does
not, a send is not a stage, failure is a 502 and never a silent retry, and with nothing configured
nothing leaves the machine. `lawai_*` settings are gone; law-ai's two receivers are described in
ocrmd.example.toml.

### A library API

`from ocrmd.api import Library` — the third driver over the same modules the CLI and the web UI use.
Its own settings and store (never the CLI's singleton), so a host can point it at any root; the model
provider as an argument (`"auto"`, `None` for OCR-only, or any `llm.Provider`), so a host with its own
model-routing layer keeps every call inside it; nothing printed. `ingest`, `ingest_image`, `verify`,
`build`, `page(s)`, `document(s)`, `export`, `search`, `envelope`, `read_region`, `send`, `delete_*`,
`doctor`. [law-ai](https://github.com/bamr87/law-ai) installs ocrmd as an optional plugin on this
surface.

### The camera was never telling us where the lens was

Every focus decision ocrmd made came from AVFoundation. Measured on the Logitech C525 this tool was
built on, over 599 frames including two forced mode changes: **`isAdjustingFocus` was false on every
one**, and `lensPosition` reads 0.0 and never moves. The focus gate has never fired on this rig; the
HUD's `FOCUSING` state has never been shown.

And the failure it should have caught is real: the same book under the same lamp, minutes apart,
measured page sharpness of **18 and then 1030**, with the camera reporting `focus=continuous`
throughout and holding the bad state through a `locked`→`continuous` toggle. A scan started there
photographs the whole book out of focus, because the sharpness floor was relative to the best *that
session* had shown and decays — a session that starts soft lowers its own bar. In this store that
shows up as crop blur of 149 median for pages that ended merely flagged, 101 for `suspect`, 66 for
`illegible`.

**[focus.py](ocrmd/focus.py) now watches the lens in the image**, which every camera allows:

- **is it moving** — the relative spread of page sharpness over 12 frames. On a still, in-focus scene
  that measured p50 0.012, p99 0.026, max 0.027 (598 frames), so the band is 0.06;
- **is it soft** — sharpness against the best this *rig* has ever managed, remembered per camera in
  `.ocrmd-camera.json`. A working rig never dropped below 0.939 of its own best over 20 s; the broken
  state was 0.017, so the line is at 0.45;
- **then do something** — retrigger the camera's focus, escalate to restarting the stream, and after
  four failed attempts accept that the rig has changed and say so, rather than blocking every capture
  forever.

The shutter's floor is anchored to that memory, so a soft session can no longer talk itself down.
`ocrmd doctor` prints what each camera has reached, and the HUD and web scan view show the ratio.

### Also

- `doc_exports` and the catalogue columns are additive migrations; databases from 0.8 open unchanged.
- `ocrmd doctor` reports the inventory (documents, pages, indexed pages, FTS5 or not) and the
  configured destinations; `/api/settings` masks sink credentials.
- The run skill's driver now asserts the inventory, search, every export shape and the empty
  destinations list, and screenshots the Library and Search views.
- **`camctl.focus_state` cost 6.7 ms and was called every frame** — a fifth of the frame budget at
  30 fps, spent building an AVFoundation discovery session to read two booleans off a camera that had
  not moved. The device is cached (invalidated when it disconnects): **0.003 ms**.
- **A scan left the camera locked.** `focus=settle` locks the lens after the first capture and nothing
  ever unlocked it; this rig comes back up `locked` after every such session, which is exactly how a
  lens ends up parked somewhere useless. The mode the session opened with is restored on exit.
- A malformed YAML string in `test_etl.py` (0.9.0 work in progress) made the suite red: the extra
  operations were appended after the `pipeline:` block instead of before it.

## 0.8.0 — 2026-08-28

### A view of the page that is meant to be read

`processed/<id>.flat.png`, the **flat** view in the web UI and the default one on the page screen. The
crop with the illumination divided out, an unsharp mask, and a tone curve that puts the ink on black
and the paper on white — the page as a flatbed scanner would have delivered it, instead of as a
webcam saw it under a desk lamp.

Scored on the 40 pages of this store that carry a model transcription, the way `ocrmd ocr-bench`
scores an engine: **F1 0.741 for the flat view against 0.246 for `bin`**, with the untouched colour
crop the engines actually read at 0.744. The view meant for reading now reads as well as the crop
itself, where the one it replaces was throwing two thirds of the words away.

Three things that measurement decided, each having looked right first:

- **no denoising** — a bilateral filter makes it look cleaner and costs 0.144 F1; at these x-heights
  the "noise" is the thin strokes;
- **Otsu classes rather than a fixed percentile** — ink coverage runs 1–15 % of a page, so a fixed
  percentile lands on the paper of a sparse page, which is the page that needed it most;
- **not the class medians** — mapping median-ink to black flattens half of every stroke: 0.693.

Nothing else changed: the engines and the model still read the colour crop, `bin` is still written, and
neither of those two is read by the pipeline at all.

It is **derived, not staged**: a pure function of `<id>.png`, with no version string, no place in the
page id, and no reprocess. Pages scanned long before it existed get one the first time anything asks
(`pipeline.ensure_flat`); delete the file and the next request rebuilds it.

### Select a region of the page and read it harder

Drag a box on the page image in the web UI. The rectangle is cut out of the flat rendering, enlarged
on its own terms and re-read by the local engines; **Ask the model** sends that one crop to the
provider for the words they still cannot get. The result comes back beside the crop it was read from,
with *Copy* and *Insert into body*. Also `ocrmd read --doc X --page N --region x,y,w,h [--llm]` and
`POST /api/documents/{slug}/pages/{seq}/region`.

Measured, on the 38 pages of this store with a model transcription, by cutting each into four bands
and reading each band as a region: F1 is unchanged (0.740 whole page, 0.738 by region), **recall rises
2–3 points and precision falls by about as much**. That is the shape of the feature — reading a piece
of a page on its own finds more words and is less choosy about them, which is the right trade when a
person is looking at the same crop and the wrong one for the unattended page pass. A read costs ~0.9 s
for a line, ~1.3 s for a paragraph.

**Nothing is written.** No stage is marked, no artifact is rewritten, and the model's answer is
returned *beside* the OCR one rather than replacing it — everywhere else in ocrmd two readings are
reconciled and their disagreements marked, and this path has no second reading, so the operator is the
reconciler. *Insert into body* puts the text at the cursor in the editor; only Save writes.

### Extraction: DocETL's operators over the finished pages

`ocrmd etl --doc X -p pipeline.yaml`, and an **Extract** panel on the document page in the web UI. A
transcribed book is a corpus; this answers what is in it. `map`, `filter`, `extract`, `reduce` and
`resolve`, plus `split`, `unnest` and `sample` that cost nothing — each an LLM prompt and an output
schema, chained in a YAML spec. The operator model, the type notation and the spec shape are
[DocETL](https://github.com/ucbepic/docetl)'s (UC Berkeley EPIC lab, MIT).

The implementation is ocrmd's: six operators over the existing `Provider`, no new dependencies.
DocETL brings its own LLM stack and about thirty runtime packages, all of which ocrmd already has an
answer for — a provider layer, credentials, `llm_calls` accounting, and versioned artifacts that make
re-running free. What it has that is worth reaching for is its optimizer, so `--export-docetl out/`
writes the same pipeline plus the rows as a dataset in the form `docetl run` takes, and
`pip install 'ocrmd[docetl]'` gets you the runner.

Three things are ocrmd's rather than the operator model's:

- **every row names its page** (`_page`, `_page_id`, `_document`, `_pages` after a reduce), because an
  extraction you cannot take back to the photograph is a rumour;
- **nothing is inferred** — the operators run under the same refusal as the transcription stage, and
  blank and illegible pages are never offered to one;
- **answers are cached per operator and versioned by the prompt**, so editing one operator re-runs one
  operator; `--dry-run` shows the prompts and the call count and sends nothing.

[examples/cases.pipeline.yaml](examples/cases.pipeline.yaml) is a worked example: every case a
textbook cites, one row per case, the same case spelled three ways resolved into one.

### Also

- `Provider.complete_json` — one structured answer to one text question, on both providers; xAI's
  retry/backoff loop is now shared by its two calls rather than inlined in `transcribe`.
- The page image column keeps its selection in page fractions, so it survives zooming, switching
  between the flat/processed/bin/raw views, and the re-render after each read.
- `ocr.upscale_for_ocr` takes a `max_scale`; pages keep the measured cap of 3, regions get 6.
- `driver.py api` now asserts the flat view and the region read, so the run skill covers both.
- ARCHITECTURE gained §11c and the flat-rendering section, and records what the opencv `<4.12` pin
  actually is (KleidiCV ships in 4.12+; 4.11.0.86 is the last release without it).

## 0.7.0 — 2026-08-27

### The spread split was cutting through text

Reading the pages captured since 0.6 showed halves like 865/444 and 870/443 px — a page and an
offcut. Measured across all 60 split captures in the store:

- The gutter was found as **the darkest column** in the middle third. Under a desk lamp the darkest
  column in that band is simply the far edge of the band, so 18 of the 60 split at almost exactly 1/3
  or 2/3 of the sheet. The split line sat on ink 0.109 where the page body measures 0.117: it was
  cutting straight through text, and 40 of the 60 would have been split somewhere else.
- A binding is not dark, it is **empty**. `preprocess.ink_columns` now measures per-column fine detail
  as high-pass energy, which ignores exposure and survives blur. Both matter: against a page softened
  until OCR reads a third of it, an adaptive threshold keeps 13 % of the signal and a Sobel magnitude
  16 %, against 39 % for the high pass — under a threshold, a spread with one page out of focus stops
  looking like a spread at all.
- **The split is now conditional.** Each of the 60 captures was read whole and split and scored on
  expected correctly-read words; splitting genuinely helps 36 and hurts 24. Trough depth separates
  those at AUC 0.797 (medians 0.527 against 0.225) and `GUTTER_DEPTH = 0.30` maximises the net gain:
  +2652 words against +2161 for splitting everything and +0 for splitting nothing. A wide crop with no
  binding — a single page photographed at an angle — is now kept whole.
- Two guards found by their own tests: a gutter needs **text on both sides** (`GUTTER_BALANCE`, kept
  loose at 0.25 because uneven light can leave the far page with a quarter of the near page's ink), or
  a ragged right margin reads as a binding; and the cut is placed at the **middle of the blank band**
  rather than its lowest point, since the band is the binding plus two unequal inner margins.
- `estimated_page_width` runs the same test, or the HUD promises 900 px a page on a sheet the pipeline
  is about to keep whole.
- `preprocess@3` → **`preprocess@4`**. Page ids derive from the crop, so `ocrmd reprocess` re-derives.

### Fixes

- **The opencv pin did not cover the bug it was written for.** `cv2.resize` was segfaulting inside
  KleidiCV — the same HAL that caused the `<5` pin — because KleidiCV also ships in opencv 4.12+, and
  4.14 hits it on the wider crops the new spread logic produces. It is a hard process crash, not an
  exception: no traceback, no partial result, in the middle of a scan. Pinned to `<4.12`; 4.11.0.86
  is the last release whose `getBuildInformation()` lists no KleidiCV at all.
  Three other explanations were tried and measured false first — RapidOCR thread-safety, image size,
  and OpenCV/ONNX thread pools — and each of those changes was reverted rather than left in.

### Orientation is measurably broken, and the fix is opt-in

Reading the newest pages turned up crops containing the whole spread, the desk and a hand, rotated
90°, made from a raw frame that was perfectly good. 10 % of the crops in the store have their text
running vertically. Scored against 68 captures labelled by what OCR can actually read from them:

    projection profile picks the right axis    57 %   (chance 50 %)
    ink asymmetry, all four rotations          25 %   (chance 25 %)
    ink asymmetry, given the right axis        53 %   (chance 50 %)

Every stage but the OCR probe is a coin toss, and the cascade multiplies them — about 30 % end to
end. The projection profile is near-tied on 46 % of pages, so its narrowing to two candidates was
frequently arbitrary *and* binding, locking the one test that works out of the answer. Its margin
carries no information either: accuracy by margin band runs 71/50/65/47/50 %.

`orientation_probe = true` replaces it with an OCR probe over all four rotations at both recogniser
levels — fast first, because it does not auto-orient and separates cleanly on a readable page;
accurate second, because it reads the marginal captures the fast level returns nothing at all from.
On the pages the old cascade got wrong it agrees with the label 3 times in 4.

It ships **off**. It costs four to eight Vision passes a page, it doubles the test suite's runtime,
and one focus-bracket test does not yet pass with it on. The measurement is the deliverable here; the
switch is for anyone who wants to re-derive a book with `ocrmd reprocess` and pay for it.

### Reconstruction: image → Markdown/LaTeX → image, and a number for how close

`ocrmd reconstruct --doc X` (and a **Rebuild** job in the web UI) rebuilds each page from its Markdown
and scores it against the photograph it came from. A page that rebuilds badly is a page the
transcription lost something on.

- **`layout.py`** joins what the two halves of the pipeline each know: OCR has the line boxes, the LLM
  pass has the corrected text and no geometry. Lines carry reconciled words at measured positions,
  with a match floor so an unrelated document's words are never stamped onto this page's boxes, and
  duplicate lines from the cross-engine merge collapsed — invisible in a reflow, drawn on top of each
  other in a facsimile.
- **`reconstruct.py`** emits LaTeX (absolutely positioned via `textpos`: the portable master), SVG
  (the same layout with no TeX install needed) and a PNG of the rendering, through Inkscape,
  `rsvg-convert` or headless Chrome. `--pdf` compiles the LaTeX when a TeX engine exists.
- **Figures are vectorised only where it measurably worked**: a line drawing is traced with
  `cv2.findContours`, the trace is rasterised and compared with the crop it came from, and anything
  below `VECTOR_FIDELITY` falls back to the embedded image. Photographs are never traced.
- Figure detection rejects candidates that lie inside the text rows. Without that, erasing line boxes
  leaves the gaps between them, closing joins the whole column into one blob, and the reconstruction
  gets a photograph of the page pasted over it — which is what the first run did.
- **Fidelity** correlates the two pages as *ink* (high-passed and normalised), because the rebuild is
  black on white and the original is a photograph under a lamp. `--review` asks the model what the
  rebuild got wrong, since a pixel score says how much differs, not what.

### Annotation sync: the paper stays the document of record

`ocrmd annotate --doc X` puts the camera back on already-scanned pages. Underline, highlight, strike
through or write a margin note, and the page file follows — `==highlight==`, `**underline**`,
`~~strike~~`, and handwriting as a bullet under `## Notes`.

- **OCR and the LLM run only when something changed.** The page has been read once and every line's
  position is on disk, so an underline or a highlight is resolved by geometry alone and costs nothing
  at all. Only handwriting reaches a model (`Provider.read_handwriting`), because it is the one thing
  on the page never transcribed; local OCR is not asked, being trained on print. A page nobody wrote
  on costs one image subtraction.
- Two channels, because pens and highlighters do opposite things: a pen darkens paper, a highlighter
  colours it while leaving it bright and is found as a rise in saturation.
- Alignment is ECC on **plain greyscale**. High-passing first was tried and is measurably wrong — ECC
  is already invariant to brightness and contrast, and removing the low frequencies takes away what it
  converges on: on a page nudged 5 px and 0.6°, greyscale registers at cc 0.99 and high-passed input
  at 0.38, recovering the shift as 21 px instead of 3.
- Marks already in the file are left alone, so a camera resting on a page does not wrap it repeatedly;
  a diff covering more than `MAX_CHANGE` is refused as a misalignment rather than applied; blobs at
  the page edge are warp artifacts, not annotations; and the page is written without refreshing the
  fingerprint, so what you wrote on the paper survives the next `verify`.
- `--dry-run` reports what it would change and writes nothing.

## 0.6.0 — 2026-08-27

### The auto-shutter, rebuilt

Captures were firing during page turns, and captures nothing could read were becoming pages. Both were
measured on this repository's own store before anything was changed.

- **The page outline is now a gate** (`scene.py`). The old chain was memoryless and every one of its
  tests can be true *in the middle of a page turn*: the hand lets go, the page is mid-fall, the pixels
  go quiet for a moment, the lens has not moved and the frame is certainly novel. In one real
  two-minute session 21 captures were taken and 16 of the 20 consecutive pairs disagree about where
  the page is by more than 0.03 of the frame (drifts of 0.08-0.16) — the book was still moving, and
  all 21 became pages. The shutter now also requires the detected sheet to hold its position across
  `geometry_frames` (3) detections within `geometry_tolerance` (0.03).
  The tolerance is measured, not guessed: re-detecting 30 real captures under sensor noise (450 draws)
  moved the outline under 0.001 of the frame 95 % of the time and past 0.03 in 0.89 % of draws, and
  *every* unstable draw came from one frame of the thirty — particular scenes are genuinely ambiguous
  about where the sheet is, and holding the shutter there is right. So the window test is strict, with
  no outlier tolerance.
- **The scene remembers what disturbed it.** A desk that twitches only has to go quiet again;
  a desk where a hand just moved a page must prove it has finished. `settle_frames` (20) is required
  after a hand or a moved page, `stable_frames` (8) after ordinary noise, and the worst disturbance in
  a run is the one that counts — a page turn ending in a moment of jitter is still a page turn. A
  capture also spends the scene's quiet, so one settled period yields one capture.
- Two deliberate non-behaviours, both found by the new tests: an outline the detector *never* finds is
  a stable observation rather than a blocker (a page filling the frame edge to edge is never a bright
  blob, and blocking there would deadlock that rig entirely), and `Outline` scales both sides of the
  sheet by the same number and holds them unordered, because `minAreaRect` names one sheet either
  `(w, h, -90)` or `(h, w, 0)` and normalising by width and height separately made a still page look
  like a 0.37 drift whenever the naming flipped.
- The HUD and the web scan view now name what they are waiting for — `PAGE MOVING`, `SETTLING AFTER
  HAND 7/20` — and the meter is drawn against the target this scene actually needs.

### Captures that cannot be read are refused, not made into pages

- **Admission test** (`legibility.py`), run after local OCR and before any page exists. Scoring the
  image was tried first and measurably does not work: against the model's own `legible` verdict on the
  real crops, page width scores AUC 0.63, x-height 0.46, edge sharpness 0.58, ink/paper contrast 0.57,
  Laplacian variance 0.62, line-spacing gradient 0.64 — nothing above 0.64. Asking the OCR engines
  does: `words < 60` refuses 21 of 120 scored pages, 20 of them (95 %) independently called illegible,
  losing one legible page, against a 73 % base rate. Defaults `admit_min_words = 60`,
  `admit_min_confidence = 0.5`.
- Replayed over the whole store it refuses 23 of 139 pages, 91 % of the judged ones illegible, and
  saves those 23 LLM calls. **It is not a fix for a badly set up rig**: the illegible share among pages
  still made falls only from 73 % to 69 %, because most bad pages here have plenty of readable words
  and fail on framing and focus. Said plainly in the README next to the framing advice.
- **Nothing is deleted.** The raw frame stays, the `captures` row records the reason and the document
  (two new columns), no page, page file or LLM call is created, and `ocrmd status` lists what was
  refused with the command to force one in. Re-ingesting a refused frame clears the refusal
  (`store.clear_rejection`) — without that the refusal row read as an exact duplicate and the
  documented override silently did nothing.
- The test applies only to *automatic* captures. `ocrmd ingest` and forced (space-bar) captures are
  decisions, not guesses the shutter made, and are never subject to it. `--no-admit` /
  `admit_captures = false` restores the old behaviour.
- Half a spread can be kept when the other half is unreadable, rather than losing a good page with it.

### Fixes
- `store.add_capture` bound its columns positionally, so the first migration to add a `captures`
  column would have written values into the wrong ones. Named columns now.

## 0.5.0 — 2026-08-27

### Focus bracketing

A page that does not lie flat is not uniformly in focus, and autofocus optimises one number over the
whole frame — usually dominated by a page edge, a gutter shadow, or the flatter of two leaves. The
frame that wins on average sharpness can be the worse frame for the top third of the page, and no
preprocessing recovers what the optics missed there. So the shutter now takes more than one frame,
and OCR reads more than one of them.

- **`capture.focus_bracket`**: when the gates open, the page is re-shot at several lens positions
  (`AVCaptureDevice.setFocusModeLockedWithLensPosition`, new in `camctl.set_lens_position`). Default
  `focus_bracket = "auto"` — 3 shots at ±0.05 — brackets only when the capture is below 90 % of the
  best sharpness the scene has shown, since a bracket on an already-sharp frame costs ~0.5 s of frozen
  preview for nothing. `always` and `off` are the other settings; `--focus-bracket` overrides per scan.
  The lens is always restored: continuous autofocus re-enabled, a deliberately locked lens put back
  where it was.
- **Many webcams cannot vary focus at all** — `isLockingFocusWithCustomLensPositionSupported` is
  false on both cameras of the development rig, including the C525, which offers `locked` and
  `continuous` modes but no lens position. `ocrmd doctor` now reports this per camera. There the
  bracket reads no extra frames and the shutter instead keeps the **burst's runners-up** as variants
  rather than discarding them — frames it already read, differing only by sensor noise and the lens's
  own micro-adjustment. Everything downstream is identical, but they are a much weaker second opinion
  and **the measurements below do not transfer to that fallback**, which is unmeasured on real
  hardware. `focus_keep = 0` disables it.
- The losing exposures ride with the capture as **variants** — `raw/<sha>.v<n>.jpg`, cropped to
  `processed/<pid>.v<n>.png` with the page's own orientation forced and scaled onto its exact
  dimensions. **Page identity, dedup and every stage still derive from the primary frame alone.** A
  variant whose own page detection landed elsewhere (different aspect ratio by > 4 %) is dropped
  rather than stretched: a mis-cropped variant aligns its text against the wrong lines.
- **`multifocus.resolve`** votes across exposures the way `ocr.merge` votes across engines. Below
  `ocr_trust`, the most confident variant reading of a line replaces it if it wins by
  `focus_swap_margin` (0.10); a line only a variant saw is filled in at ≥ 0.9; one variant line can
  be spent on only one of the page's lines. New rule `merge` does not have: a *substantially longer*
  reading may replace a short one when it is certain — measured, an out-of-focus band yields a
  fragment (`"milies"` for `"is so well fixed in the minds of the surrounding families"`), not a
  word-for-word misreading, and a symmetric length window rejected every capture the bracket exists
  to rescue. Shorter readings are still refused: that is text being dropped.
- Measured on a 520-px page with its top half blurred, correct words (page alone → bracketed):
  165→165 at 5 px blur (no false positives), 117→138 at 7 px, 101→146 at 9 px, 64→165 at 11 px,
  55→165 at 13 px. **Every variant is read, including for pages whose own lines are all confident** —
  gating on confidence was tried first and is wrong: the two biggest recoveries were on captures the
  primary read at ≥ 0.98 mean confidence, having simply not emitted the unreadable lines. Cost is one
  OCR pass per variant (~1.2 s with two engines) on the worker thread.
- **What no focus could read goes to the model, not to a coin flip.** Lines still under `ocr_trust`
  after the bracket are recorded in `ocr/<id>.focus.json` with every candidate reading and which
  exposure produced it; the LLM request then carries the `focus_llm_images` (2) variants holding the
  best candidates plus that region list. Each extra image is introduced by its own text block saying
  it is another view of the *same* page — unlabelled, a second image reads as the next page and the
  model merges two readings into one, which is exactly the fabrication this pipeline exists to catch.
  Candidates are labelled as low-confidence machine guesses, with the instruction to transcribe what
  is visible or report it in `uncertain`. Pages with nothing unresolved send what they always did.
- Page frontmatter gains `ocr.focus` (variants, improved, filled, unresolved) on bracketed pages,
  shown in the web UI's page detail; the scan reports variants captured and lines recovered.
- Stage versions: `ocr@2` → **`ocr@3`** (`verify` re-reads pages locally, free) and
  `transcribe@3` → **`transcribe@4`** (only `verify --refresh` acts on it, as always).

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
