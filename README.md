# ocrmd

> Scan books and pages into cross-checked Markdown: local OCR diffed against a Claude vision read, with full-text search.

[![CI](https://github.com/bamr87/ocrmd/actions/workflows/ci.yml/badge.svg)](https://github.com/bamr87/ocrmd/actions/workflows/ci.yml) [![License](https://img.shields.io/github/license/bamr87/ocrmd)](LICENSE)

## Overview

Turn physical books and loose pages into cross-checked Markdown, plain text and JSON — from a webcam pointed at the book, a folder of photographs, or a PDF — and keep an inventory of everything you have scanned.

Every page is read **twice**: once by local OCR engines (Apple Vision + RapidOCR voting), once by a vision model (Claude by default). The two readings are diffed into a Markdown page file with rich frontmatter; where they disagree the page says so (`==[word]==[^ocrmd-n]`) rather than quietly picking one, and digits are never auto-resolved. The cross-check is the product: an early version fabricated a table of contents for a page it could not read, and only the diff caught it.

What you get, per root directory:

- **A corpus** — `documents/<slug>/pages/0001.md` … one file per page, and a built `<slug>.md` with a TOC;
  `document.yaml` for what only you know about the book (title, author, ISBN, tags).
- **An inventory** — every document, its catalogue entry, its progress, in SQLite (`ocrmd library`) and
  **full-text search across every page ever scanned** (`ocrmd search`).
- **Exports** — the same pages as Markdown, plain text, one JSON document or JSON-lines (`ocrmd export`).
- **Destinations** — hand a page's reading to another system that speaks a small HTTP contract
  (`ocrmd send`); [law-ai](https://github.com/bamr87/law-ai) is the worked example.
- **A library** — `from ocrmd.api import Library` embeds all of it in another program.

Built and tuned on a fixed camera over a textbook. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the measurements behind each default, and [CHANGELOG.md](CHANGELOG.md) for what changed.

## Features

- Reads every page twice: local OCR (Apple Vision and RapidOCR voting) and a vision model (Claude by default)
- Diffs the two readings and flags disagreements instead of silently picking one
- Ingests from a webcam, a folder of photos or a PDF
- A SQLite inventory with full-text search across every scanned page
- Exports to Markdown, plain text and JSON, plus a local web UI
- Structured-data extraction and hand-off to configured destinations

## Install (macOS)

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'   # add ,pdf for PDF ingest; ,tesseract as a last resort
.venv/bin/ocrmd doctor          # engines, cameras, credentials, database health
.venv/bin/python -m pytest      # 306 checks, ~2 min (needs a local OCR engine for the pipeline ones)
```

**Camera permission:** macOS must authorize the app that runs `ocrmd` (Terminal, iTerm, VS Code…): System Settings → Privacy & Security → Camera. `ocrmd doctor` tells you if it's denied.

**Claude credentials** (put them in `.env` next to the project, or `~/.config/ocrmd/.env`):

| Variable | Notes |
|---|---|
| `ANTHROPIC_API_KEY` | Console API key. All models; default model is `claude-opus-5`. Needed for `verify --batch`. |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code / subscription OAuth token (`sk-ant-oat…`). Works, but as of Aug 2026 the API only serves **`claude-haiku-4-5`** to it — set `OCRMD_MODEL=claude-haiku-4-5` too. Other models return a bare 429 `Error`; ocrmd reports this as "not available to this OAuth token". |
| `ANTHROPIC_AUTH_TOKEN` | Any other bearer token; same rules as above. |
| `XAI_API_KEY` | For `--provider xai` (Grok vision; implemented against xAI's OpenAI-compatible endpoint, not exercised live here). |

Without credentials ocrmd runs OCR-only and you can fill in the LLM pass later with `ocrmd verify`.

**Local OCR engines.** On macOS Apple Vision (through pyobjc, no extra install) is the primary engine; `pip install 'ocrmd[rapidocr]'` adds RapidOCR (PP-OCR on ONNX Runtime, models fetched on first use), which is the primary engine on Linux/Windows. When two engines are installed they both read every page and *vote*: words they agree on are trusted, words they disagree on are flagged for the LLM comparison, lines one missed are filled from the other (`ocr_ensemble = false` turns this off, or set `ocr_engines = ["apple_vision"]`). Tesseract (`ocrmd[tesseract]` + the binary) is supported but measured far behind on camera images. `ocrmd ocr-bench` measures whatever is installed against the LLM transcriptions already in your store, so you can see what a change buys before trusting it.

Page images are sent to the provider you configure; nothing else leaves the machine.

## Configuration

Every setting has a default, can be set in `ocrmd.toml` (project root) or `~/.config/ocrmd/config.toml`, and can be overridden by an `OCRMD_<KEY>` environment variable; CLI flags win over all of them. See [ocrmd.example.toml](ocrmd.example.toml) for the full list with comments. Invalid values fail fast with the offending key named. `ocrmd doctor` prints the effective model/provider/focus/workers.

## Use

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
ocrmd send --doc lsat --page 15 --to law-ai -o hint="question 15"   # hand a page to a destination (see below)
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --dry-run   # extract structured data (see below)
ocrmd ocr-bench --doc lsat --compare           # score every installed OCR engine (and pair) against the LLM transcriptions
ocrmd doctor --costs
ocrmd --debug verify                           # full tracebacks and debug logs
```

Indices shuffle between runs on macOS; `--camera c525` (a device-name substring) is stable.

## Web UI

```sh
ocrmd serve            # http://127.0.0.1:8765, opens a browser tab
```

Everything above, in a browser, plus what a terminal cannot show:

- **Library** — every document as a card with its catalogue entry (title, author, year, kind, printed
page range), status bar, what needs a re-shoot or a review, and download links for every export format; a search box over every page of every document, with hits that link to the page they sit on.
- **Scan** — live HUD stream from the camera with the gate state (`READY`, `HAND IN FRAME`, …), meters,
px/page, and buttons/keys for capture, focus lock/auto, rotate, zoom, pan. Pick the document, camera, focus mode, OCR-only, optional duration/page limit. Event log and queue state update live.
- **Documents** — status bars, page table filterable by status (label, width, agreement, overlap,
advice, continuity, LLM errors), drag-and-drop image ingest, Verify/Build/Reprocess buttons, download of the built `document.md`. Select pages to **delete** them (optionally both halves of a spread); **Delete…** removes the whole document after you type its slug.
- **Page** — the page image beside the transcription, in four views: **flat** (the default — the crop
flattened and contrast-stretched, the page as a flatbed scanner would have delivered it), **processed** (the crop the model sees), **bin** (the adaptive threshold older engines are given) and **raw** (the untouched frame). Zoomable, and **drag a box on it to read that region on its own** — the selection is cut out, enlarged on its own terms and re-read by the local engines, with an **Ask the model** button for the ones they still cannot get. The result comes back beside its crop, to copy or to insert at the cursor in the editor; nothing is written to the page until you save. *Edit* tab: each disagreement with one-click **OCR / LLM / type your own**, editable Markdown body. *Preview* tab: rendered Markdown with the `==[…]==` marks highlighted. *Meta* tab: printed page label, chapter/section/running head/tags, verification numbers, model notes, provenance. **Sign off** marks a page `reviewed` (body frozen); **Reopen** undoes it; **Delete…** removes the page (and offers to take its spread partner with it). Keys: `n`/`p` next/previous, `N` next flagged. Review mode (`Review N flagged` on a document) advances to the next flagged page after each sign-off. **Send…** (present only when a destination is configured) hands the page's reading, or a region's, to a receiver — see below.
- **Jobs** — verify / build / reprocess / rebuild / extract with options, live log, progress, cancel.
- **Settings** — `document.yaml` editor per document, effective settings, full doctor report.

Edits made in the UI are human edits: they survive later `verify` runs exactly like edits made to the files. The server binds to `127.0.0.1` and has no authentication — keep it that way (`--host` exists for a trusted LAN only). The API is documented at `/api/docs`.

A scan writes an OCR-only page file the moment a capture is processed and upgrades it in place when the model answers, so `documents/<slug>/pages/` is always usable, even mid-session. Quitting drains the queue; pressing Ctrl-C a second time leaves the remainder for `ocrmd verify`.

`reprocess` exists because a page's id is the hash of its *processed* image: when preprocessing improves (better cropping, rotation, spread splitting) the pages themselves change. It re-derives every page from the untouched raw captures and discards the stale OCR/LLM results, which means the LLM pass runs again and costs tokens. Raw captures are never modified.

## Extracting structured data from a scanned book

Once a book is transcribed it is a corpus, and the next question is usually *what is in it* — every case this chapter cites, every dose in the table, every term the glossary defines. `ocrmd etl` runs a declarative pipeline over a document's pages, with the operator model of [DocETL](https://github.com/ucbepic/docetl) (UC Berkeley EPIC lab, MIT):

| operator | what it does | costs tokens |
|---|---|---|
| `map` | one prompt per page, its answer merged into the row | yes |
| `filter` | keep the rows a prompt says to keep | yes |
| `extract` | a map that must answer in the page's own words | yes |
| `reduce` | group rows by a field and fold each group with one prompt | yes |
| `resolve` | canonicalise a field across rows (the same case spelled three ways) | only for pairs that pass fuzzy blocking |
| `split` / `unnest` / `sample` | chunk long pages, flatten a list into rows, take the first N | no |

```sh
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --dry-run   # see the prompts, spend nothing
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --limit 5   # try it on five pages
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml             # -> documents/lsat/extracted/cases.csv
```

Output is JSON, CSV or a Markdown table; `output: {fields: [...]}` narrows the columns (the provenance ones are always written). The web UI has the same thing: put a spec in `documents/<slug>/etl/*.yaml` (or `<root>/etl/pipelines/`) and an **Extract** panel appears on the document page, with a dry run, a live log and the result to download. [examples/cases.pipeline.yaml](examples/cases.pipeline.yaml) is a worked example.

Three things are ocrmd's rather than DocETL's, and they are why this is built in rather than bolted on:

- **Every row carries the page it came from** (`_page`, `_page_id`, `_document`; `_pages` after a
  reduce), so an extracted claim can be taken back to the photograph it was read from.
- **Nothing is inferred.** The operators run under the same refusal as the transcription stage: use
  what the page says, leave the field empty when it does not say it.
- **Answers are cached per operator and versioned by the prompt**, so editing one operator re-runs one
  operator and re-running an unchanged pipeline costs nothing — the same bargain as `verify`.

The implementation is ocrmd's own (six operators over its existing provider, no new dependencies). To use DocETL itself — for its optimizer, which rewrites pipelines — export the same spec:

```sh
pip install 'ocrmd[docetl]'
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --export-docetl out/
docetl run out/pipeline.yaml
```

## The inventory: what has been scanned

`ocrmd.db` records every page and every stage; `document.yaml` holds what only you know about the book. `ocrmd library` joins the two — and mirrors the catalogue fields onto the `documents` table, so any tool that reads SQLite can answer "what has been scanned here":

```
slug                   title                                       pages      done
austen                 Pride and Prejudice                        312 page(s)   298 done  [novel · 1813 · Jane Austen]
    printed iv–301 · tags novel · exported jsonl, md · last capture 2026-09-06T10:41
lsat                   LSAT Prep                                  139 page(s)    88 done  [textbook]
```

`ocrmd search "words"` is full-text search over every page's body (SQLite FTS5 where available, a substring scan otherwise); quote a phrase, `word*` for a prefix, `--doc` to narrow. Each hit names its page and links to it in the web UI, because a hit you cannot take back to the page is a rumour. The index is kept up to date **by file mtime, lazily**: a page rewritten by the pipeline, the editor in the web UI, `ocrmd review`, an annotation sync or a text editor is picked up the next time anyone searches, and no write path has to know an index exists (`ocrmd index --force` rebuilds it after a restore).

## Exports

```sh
ocrmd export --doc austen --format md      # the built document: every page under its anchor, a TOC
ocrmd export --doc austen --format txt     # plain text, one page after another behind a separator line
ocrmd export --doc austen --format json    # one JSON object: metadata, every page's frontmatter, body, layout, extractions
ocrmd export --doc austen --format jsonl   # one JSON object per page — for a search index, a RAG loader, an ETL tool
```

They land under `export/<slug>/` (or `--out`), are recorded in the inventory, and every one keeps the page ids so a consumer can come back to the photograph. The web UI offers the same four as downloads on every document. The text export does not silently resolve a disagreement: `ambitious[?ambiguous]` keeps both readings in-line (`flags="llm"` or `"ocr"` in the library API picks a side on purpose).

## Destinations: handing a page to another system

ocrmd's only outbound path. A destination is any URL that speaks a small contract (the whole of it is in [ocrmd/sinks.py](ocrmd/sinks.py)): `GET` answers with a *descriptor* — what the receiver is and which typed options it takes — and `POST` takes the *envelope*: the page's **reconciled** text, its provenance (document, page, page id, both agreement scores, the flag count, a link back to the image), the document's catalogue entry, and your options. Configure one in `ocrmd.toml`:

```toml
[[sinks]]
name = "law-ai"
url = "http://localhost:8000/api/lsat/intake/ocr"    # Law AI: the page becomes a draft LSAT question, pending review
options = { section = "LR" }                         # defaults for the receiver's own options
```

(or `OCRMD_SINKS='[{"name":"law-ai","url":"…"}]'`). A **Send…** button then appears on every page and in every region readout, drawing the receiver's own options from its descriptor; from a terminal:

```sh
ocrmd sinks                                                     # what is configured, and what each one takes
ocrmd send --doc lsat --page 15 --to law-ai -o hint="question 15"
#   1 question(s) queued for review (key D)
#     pending   Which of the following statements best… · LR/med · key D (inferred) · grounding 1.0  [prop-de2bcfe12558]
ocrmd send --doc lsat --page 15 --to law-ai --region 0.1,0.4,0.8,0.3   # read a box first, send *that* reading
ocrmd send --doc lsat --page 15 --to law-ai --dry-run                  # print the envelope, send nothing
```

Three things are worth knowing before you press it:

- **The text goes, the photograph does not.** A receiver with the picture and a vision model would
re-read the page single-pass and overrule the cross-check, which is the one thing this pipeline makes that a single pass cannot. It gets the page id (the sha256 of the processed image) and a link back here, so a reviewer over there can always see the photograph.
- **A send is not a stage.** Nothing is written or marked; pressing the button twice is safe on this
  side, and a well-behaved receiver deduplicates and says so.
- **Failure is reported, never retried into silence.** An unreachable receiver is a 502 from the web
  UI, not a 500: this server is fine, the page just did not get through.

A Python destination registers through the `ocrmd.sinks` entry-point group instead of a URL. [law-ai](https://github.com/bamr87/law-ai) ships two receivers — its LSAT intake, which extracts questions from the text and parks them in a human-review queue (every draft scored for *grounding* against the page, so an invented answer choice is dropped before a reviewer sees it), and a per-matter door that files the page as a source document — and installs ocrmd itself as an optional plugin for OCR-ing uploads inside the platform.

## Embedding ocrmd in another program

```python
from ocrmd.api import Library

lib = Library("~/scans")                                   # any directory; created on first use
report = lib.ingest(["~/photos/austen"], doc="austen", title="Pride and Prejudice")
print(report.summary())                                   # austen: 312 page(s) from 312 file(s), 3 duplicate(s) skipped
page = lib.page("austen", 12)                             # .text, .status, .agreement, .images, .frontmatter
lib.export("austen", "jsonl")
lib.search("Netherfield")[0].sequence
lib.envelope("austen", 12)                                # text + provenance, the same object a destination gets
lib.send("austen", 12, "law-ai", {"hint": "question 15"})
```

`Library` holds its own settings and store (never the CLI's singleton), so a host can point it at any root, and it takes the model provider as an argument: `"auto"` (the configured one, when credentials exist), `None` (OCR only — nothing leaves the machine), or any object implementing `ocrmd.llm.Provider`, which is how a host with its own model-routing layer keeps every call inside it. Nothing in it prints; progress lines go to an `emit(line, level)` callback if you pass one.

## Setting up the rig (textbook under a fixed webcam)

This is the setup the tool was tuned on, and what the defaults assume.

1. **Mount the camera looking straight down**, close enough that the scan window's **px/page** readout
turns green (≥900 px across one page). A C525 framing a whole two-page spread at 1920×1080 gives each page ~700 px; framing one page at a time roughly doubles that. Resolution on the page is the only quality lever that measurably matters — see below.
2. **Dark, matte surface** under the book. Page detection keys on paper being brighter than what
   surrounds it. A pale wooden desk works but a black mat works better.
3. **Even light from the side**, no lamp in the frame. Glare and a bright-lit hand both look like paper.
4. **Watch the focus line in the HUD.** ocrmd judges focus from the picture, not from the camera:
webcams routinely report `continuous` while the lens sits somewhere useless (measured on the C525 this was built on: the same book at sharpness 18 and then 1030 minutes apart, with the driver claiming continuous autofocus throughout). It remembers what your rig has actually managed, warns when a page is far below that, and refocuses the camera itself. If it says it has given up, the lens is not the problem — check the distance, the light, and whether something is between the camera and the page.
5. **Run with `--focus settle`** (or `focus = "settle"` in `ocrmd.toml`): continuous autofocus until
   the first capture, then locked, so the lens stops hunting every time a hand comes in.
6. **Start the scan with your hands out of the frame, then turn pages and get your hand out.** The
hand gate learns what "paper" looks like from the first frames it sees; a hand that is already there at start is part of the baseline and the first capture will include it (the gate corrects itself as soon as the hand leaves). After that the shutter waits while a hand covers the page (`HAND IN FRAME` in the window). A paperback that will not stay open wants a sheet of glass or a couple of weights on the margins — a hand holding it open blocks the shutter by design.
7. Watch the **advice line**: `page runs past the top edge — text is being cut off` and
   `printed page jumped 37 → 41` are both things to fix *now*, not at review time.

## The scan window

`scan` opens a live preview with a heads-up display:

- **Gate state** with meters, so it is obvious *why* the shutter is holding off:

  | State | Meaning |
  |---|---|
  | `MOVING` | frame-to-frame motion above threshold (a page turning, a hand moving) |
  | `FOCUSING` | page sharpness is still moving: the lens has not settled |
  | `HAND IN FRAME` | paper that was visible a moment ago is covered (hand, pen, curled page) |
  | `SAME PAGE` | looks like the last capture — turn the page |
  | `BLURRY` | sharpness below the floor (absolute, and relative to the best this rig has produced) |
  | `READY` | capturing on the next tick |

- **The detected page outline** drawn on the frame, and **px/page** in green once it clears the threshold.
  Frame edges that cut through text are marked in red.
- **Camera state**: resolution, focus mode, `*` while page sharpness is still moving, a warning when
the page is soft for this rig, and the hand gate's numbers (`cover=` how much reference paper is hidden, `skin=` how skin-toned the hidden part is).
- **A work panel**: what is being processed right now (`page 0003: llm`), how many captures are
queued, `LLM OFF` if the service has been given up on for this session, and the status of recent pages colour-coded by verification result.
- **Advice and errors** along the bottom. A white flash confirms each capture.

**Capture and processing run on separate threads**, and the LLM pass on a small pool (`llm_workers`). A page takes seconds (OCR) to tens of seconds (LLM); doing that inline froze the preview.

### Camera view: rotate, zoom, pan

| Key | Action |
|---|---|
| `r` / `R` | rotate 90° clockwise / counter-clockwise |
| `+` / `-` | digital zoom in / out (1×–6×) |
| `w a s d` or arrows | pan the zoom window |
| `f` / `F` | lock focus where it is / back to continuous autofocus |
| `0` | reset view |
| `space` / `q` | force capture (bypasses every gate, hand included) / quit |

The view is saved per camera name in `.ocrmd-camera.json` and reapplied next time; `--rotate 180 --zoom 2` set it from the CLI. The same transform is applied to captured frames, so what you see is what gets OCR'd. Zoom is a crop of the sensor frame, so it improves framing and page detection but not pixel density.

## Rebuilding the page from the Markdown

`ocrmd reconstruct --doc mybook` turns each page's Markdown back into a page and scores it against the photograph it came from:

```
  0129: 1 figure(s) (0 vector), fidelity 0.45 via inkscape
132 page(s) rebuilt into reconstructed; mean fidelity 0.29
```

Three files per page. **`.tex`** is the portable master — absolutely positioned at the page's true size, so it is a facsimile rather than a reflow, and it compiles wherever a TeX distribution exists (`--pdf` compiles it if one is installed). **`.svg`** is the same layout with no TeX needed, which is what makes the round trip work on a machine that has none; it renders through Inkscape, `rsvg-convert` or headless Chrome. **`.png`** is that rendering, for looking at side by side with `processed/<id>.png`.

Figures are traced to SVG paths where that measurably worked: a line drawing is traced, the trace is rasterised and compared with the crop it came from, and anything that does not reproduce it falls back to the embedded image. Photographs are never traced.

**The number is the point.** A page that rebuilds badly is a page the transcription lost something on — a heading flattened into body text, a table turned into prose, a figure nobody mentioned. `--review` sends both images to the model and asks what it got wrong, because the score says how much differs, not what.

## Writing on the paper

`ocrmd annotate --doc mybook` puts the camera back on pages you have already scanned. Underline a phrase, run a highlighter over a sentence, write a note in the margin — the page is recognised, the new marks are diffed against the stored crop, and the Markdown is updated:

```
page 0042: highlight (yellow): a universal look that is beautiful; underline: Helen of Troy  [0 llm call(s)]
page 0043: note: check this against the 1960 debates  [1 llm call(s)]
```

| on the paper | in the Markdown |
|---|---|
| highlighter | `==the marked words==` |
| underline | `**the marked words**` |
| line through the text | `~~the marked words~~` |
| handwriting | a bullet under `## Notes` |

**Nothing is re-read unless something changed.** The page has already been transcribed and every line's position is on disk, so an underline or a highlight is resolved by geometry alone and costs nothing at all. Only handwriting reaches the model, because it is the one thing on the page that was never transcribed. A page nobody has written on costs one image subtraction.

Marks already in the file are left alone, so the camera can rest on a page without wrapping it again and again, and `--dry-run` reports what it would do without writing. What you write on the paper counts as a human edit and survives the next `verify`.

## Output

Everything lives in the repo root alongside the code (override with `OCRMD_ROOT`):

```
raw/<sha256>.jpg                   immutable captures
processed/<page>.png|.bin.png      warped/deskewed crop, binarised copy, + .json metadata
ocr/<page>.json                    local OCR result (text, boxes, confidence)
llm/<page>.<provider>.json         Claude transcription (structured)
documents/<slug>/document.yaml     title, author, citation, tags, project — edit this
documents/<slug>/pages/0001.md     one file per page, frontmatter + canonical markdown
documents/<slug>/<slug>.md         built document: TOC, page anchors, needs_reshoot, label_gaps
export/<slug>/<slug>.{md,txt,json,jsonl}   exports (`ocrmd export`)
ocrmd.db                           SQLite: the inventory — documents (catalogue mirrored from document.yaml), pages,
                                   captures, stages, llm_calls, batches, page_text (+ FTS5 index), doc_exports
ocrmd.log                          rotating log (INFO; --debug for everything)
```

`document.yaml` is created on first use with a template. Fill in title/author/citation once; `build` inherits them into every page's `context` (a page's own value wins) and carries `chapter`/`section` forward across pages with no heading, so every page knows where it sits in the book and the TOC comes out right.

Disagreements between OCR and Claude appear as `==[word]==[^ocrmd-n]` with both readings in the footnote; digits are never auto-resolved.

### Verification status

Every page carries `verification.status`, and the two sources are compared two ways: `agreement` (order-sensitive, how much of the OCR text the model reproduced in sequence) and `overlap` (order-insensitive vocabulary containment). Both are needed — OCR engines read contents pages and multi-column layouts out of order, so a sequence diff alone calls two readings of the *same* page completely different.

| Status | Meaning | What to do |
|---|---|---|
| `verified` | Both sources agree and nothing is unresolved | nothing |
| `reviewed` | A human resolved every disagreement (`ocrmd review`) | nothing; the body is now frozen |
| `flagged` | They agree on the content but differ in places | `ocrmd review` walks the `==[…]==` marks |
| `suspect` | They share almost no vocabulary — one is not describing this page | check the image; the model may have read a page it could not see |
| `illegible` | The model declined to transcribe and OCR found only noise | re-shoot the page closer or better lit |
| `blank` | Neither source found any text | nothing; the page really is empty |
| `unverified` | The LLM pass has not run, or failed (`verification.llm_error` says why) | `ocrmd verify` |

The model is instructed never to invent text it cannot read: on an unreadable page it returns an empty transcription with a reason rather than plausible-looking prose. That is deliberate — an early version of this tool cheerfully produced a fabricated table of contents for a page it could not resolve, and only the cross-check caught it.

### Your edits are kept

The pipeline rewrites page files on `verify` and `reprocess`, but it fingerprints the human-editable parts every time it writes. If you have changed a page's `context`, `page_label`, `references.related` or `verification.reviewed_by`, those values survive the next run; a page you have signed off through `ocrmd review` also keeps its body. Provenance (`source`, `processing`, `ocr`, `llm`, `history`) and `nav` are always the pipeline's.

## Operations

- **Offline / service down.** After `llm_failure_limit` consecutive service failures (network, auth,
quota — not page-level refusals) a scan drops to OCR-only, says so once, and keeps capturing. Every capture is on disk before anything else happens; `ocrmd verify --doc X` finishes the LLM pass later.
- **Costs.** `ocrmd doctor --costs` sums tokens per model. `verify --batch` sends a backlog through the
Message Batches API at half price (`--no-wait` returns immediately; `verify --collect` fetches). A prompt or model change never re-runs pages by itself — `verify --refresh` is the explicit, paid step.
- **Camera drop-outs** are detected (no frames for a few seconds) and the device reopened; after five
  failures the scan ends cleanly and builds what it has.
- **Health.** `ocrmd doctor` checks the SQLite integrity, pages whose raw/processed files went missing,
  orphaned artifacts, open batches, camera focus state and the dictionary the orientation probe needs.
- **Upgrading.** Databases from 0.1 are migrated in place (additive columns). Bumped preprocessing
  (`preprocess@N`) is what `reprocess` is for; bumped prompts (`transcribe@N`) are what `verify --refresh` is for.
- **Logs.** `ocrmd.log` rotates at 2 MB × 3. HTTP client logs are kept at WARNING so tokens and
  request bodies never land in it.
- **Web UI and CLI share one database.** Run one `ocrmd serve` per root; `verify`/`build` from the CLI
  while the UI is open is fine (SQLite WAL + locking), but don't run two scans against one camera.

## Debugging in VS Code

[.vscode/launch.json](.vscode/launch.json) has ready-made configurations: doctor, status, **serve (web UI)** — uvicorn runs in-process so breakpoints in `ocrmd/web/` and the pipeline fire from the browser — scan variants (C525 with settle focus, OCR-only, headless 60 s with debug logs, camera index 0), ingest, verify (missing / flagged only / `--refresh`), reprocess, review, build, pytest (all / current file) and the current file. They run the CLI in-process as a module so breakpoints work anywhere in the package, prompt for the document slug and camera, and load `.env` for credentials. [.vscode/tasks.json](.vscode/tasks.json) adds lint + test (the default build task), serve, doctor, status and install; [.vscode/settings.json](.vscode/settings.json) enables pytest in the Test Explorer and keeps the scan data directories out of search and file watching.

## Getting good captures from a webcam

**Resolution per page is the dominant quality factor — it is the only one that measurably matters.** Frame averaging, sharper interpolation (Lanczos vs cubic), larger upscale targets, CLAHE and unsharp masking were each measured against Apple Vision's word yield on a real page: every one landed inside the noise (143-148 words). Going from 434 to 728 pixels across the page moved it from 98 to 146. Post-processing cannot recover detail the optics never captured, so ocrmd does none of it and tells you about framing instead — live in the scan window, and on every page:

```
page 0001 -> 0001.md  [suspect agreement=0.22 overlap=0.33]
    page is only 521px wide; body text needs ~900px+ — move the camera closer or fill the frame with one page
```

- Orientation is detected automatically and upside-down books are usually fine — but see the warning
  under *How orientation is decided*: on this corpus the cascade is measurably weak.
- Page detection keys on the page being brighter than what surrounds it — a dark desk or mat helps.
- Uneven light across a spread hurts the far page more than anything else.
- ocrmd opens the camera's largest mode. Modes above the sensor size (the C525 advertises 1920x1080
over a 1280x960 sensor) were assumed to add nothing; measured on a real page, the opposite holds: **146 correctly-read words at 1920x1080 against 98 at 1280x960**. Override with `--resolution 1280x960`.
- Webcams often power up with focus *locked* at lens position 0 — every frame blurry. `scan` sets
continuous autofocus (and auto-exposure) through AVFoundation before opening the stream, because OpenCV reports `-1` for every camera property on macOS and ignores writes ([camctl.py](ocrmd/camctl.py)).

### Focus bracketing

Autofocus optimises one number over the whole frame, and on a book that number is dominated by whatever has the most contrast — a page edge, a gutter shadow, the flatter of two leaves. A page that does not lie flat can be sharp at the bottom and soft at the top in the very frame the shutter judged best.

So when a capture is not the sharpest the scene has shown, ocrmd re-shoots it at several lens positions (`focus_bracket = "auto"`, ±0.05 either side, 3 shots) and keeps the losers as **variants** beside the raw capture. Each variant is cropped exactly as the page was, read with the same engines, and its readings are matched to the page's line by line: a line another focus read confidently replaces one the page read badly, and a line the capture missed entirely is filled in. Whatever no focus could read stays in the page at its own low confidence *and* goes to the model — the extra exposures are attached to the LLM request along with what each of them made of the region, so the cross-check happens on the images rather than on the pipeline's guess.

Measured on a 520-px page with its top half out of focus, correct words recovered (page alone → bracketed):

| Gaussian blur on the top half | page alone | bracketed | what fired |
|---|---|---|---|
| 5 px | 165 / 165 | 165 | nothing — no false positives on a capture that was already fine |
| 7 px | 117 / 129 | **138** / 148 | 2 lines replaced |
| 9 px | 101 / 119 | **146** / 164 | 4 lines filled in |
| 11 px | 64 / 82 | **165** / 165 | 4 replaced, 6 filled |
| 13 px | 55 / 55 | **165** / 165 | 10 lines the capture never saw at all |

(correct words / words read; the page holds 165.)

The two biggest gains came from captures the primary read *confidently* — it had simply not emitted the unreadable lines, and confidence says nothing about a line an engine never produced, which is why every variant is read rather than only the doubtful ones. The bill is one OCR pass per variant (~1.2 s with two engines) and, only for pages where a region survives unreadable, ~1.5 k input tokens per attached image.

**Varying the focus needs a camera that can be locked to a custom lens position**, which many webcams cannot — the C525 this was built against offers `locked` and `continuous` focus but no lens position, and `ocrmd doctor` now tells you which of yours can:

```
camera 0:    HD Webcam C525
             ocrmd will use 1920x1080; focus modes: locked, continuous; no custom lens position (focus bracket unavailable)
```

On such a camera the shutter falls back to keeping the **burst's runners-up** as the variants instead of discarding them — the frames `burst_frames` already reads, which differ by sensor noise and the lens's own micro-adjustment. Everything downstream is identical, but they are a much weaker second opinion than a real change of focus, and **the numbers in the table above were measured with genuinely differing focus; the fallback's gain on real hardware is not measured.** `focus_keep = 0` turns it off. `--focus-bracket always` brackets every page, `off` disables the whole thing.

### How orientation is decided

A photographed page can arrive at any of four rotations and no single test is reliable, so [preprocess.py](ocrmd/preprocess.py) cascades three, strongest evidence first:

1. **Projection profile** — text lines form horizontal bands, so the power of the row profile at
   text-line frequencies says whether text runs horizontally. Narrows to `{0,180}` or `{90,270}`.
2. **Dictionary probe** — a fast OCR pass counts real words at each remaining rotation. The *fast*
Vision recogniser does not auto-orient (the accurate one does), which is what makes it usable here. Decides when it finds enough words with a clear margin; abstains otherwise.
3. **Ink asymmetry** — Latin text puts more ink above each line's centre than below (capitals and
ascenders outnumber descenders). Weak but never silent, so it breaks the tie on pages far too small for step 2.

## What the shutter refuses

Two things hold the shutter shut that a naive "is the frame still?" test would let through.

**A page turn is not a still frame.** Halfway through one the hand lets go, the page is mid-fall, the pixels go quiet for a fraction of a second, the lens has not moved and the frame is certainly new. Measured on this repository's own store, one two-minute session produced 21 captures and 16 of the 20 consecutive pairs disagree about where the page is by more than 3 % of the frame — the book was still moving, and all 21 became pages. So the shutter now also requires the *detected page outline* to hold its position for three consecutive detections, and after a hand or a moved page it waits `settle_frames` (20) instead of `stable_frames` (8) before it may fire. The HUD says which it is waiting for:

```
SETTLING AFTER HAND    settling after hand  7/20
PAGE MOVING            drift 0.148
```

**A capture nothing can read does not become a page.** Local OCR has already read the crop by the time the page would be created, so it is asked first: fewer than `admit_min_words` (60) words, or mean confidence below `admit_min_confidence` (0.5), and the frame is kept but no page is made and no LLM call is spent. On the real store that refuses 23 of 139 pages, 91 % of which the model had independently called illegible.

Nothing is deleted. `ocrmd status` lists what was refused and how to force one in:

```
book
  3 capture(s) refused as unreadable — the frames are kept, no pages were made:
    raw/9c1f0ad3e2b1…jpg  16:32:08  only 12 word(s) legible (needs 60)
    force one in with: ocrmd ingest raw/<sha>.jpg --doc book
```

`--no-admit` turns the test off for a scan; forced (space-bar) captures and `ocrmd ingest` are never subject to it.

**This is not a substitute for framing.** On that same store, admission removes the captures that were never going to be pages, but the share of illegible pages among the ones still made only falls from 73 % to 69 % — the rest fail on resolution, focus and angle, which is what the px/page readout and the framing advice are for. Fix the rig first; the shutter can only refuse what it can detect.

## Tuning the auto-shutter

`ocrmd.toml`: `motion_threshold`, `stable_frames`, `settle_frames`, `geometry_frames`, `geometry_tolerance`, `novelty_threshold`, `blur_threshold`, `occlusion_cover`, `occlusion_rebaseline`, `focus_bracket*`, `admit_*`. The preview window shows the live gate state and numbers, and every capture's sharpness/occlusion/px-per-page goes to `ocrmd.log`.

Capturing too seldom? Lower `settle_frames` (the wait after a hand) or raise `geometry_tolerance` (how much the page outline may wander). Still capturing during page turns? Raise them.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'
.venv/bin/ruff check ocrmd tests
.venv/bin/python -m pytest
```

Full commands, conventions and gotchas for humans and AI agents live in [CLAUDE.md](CLAUDE.md).

## Contributing

Contributions are welcome via pull request. Use [Conventional Commits](https://www.conventionalcommits.org/) and keep the CI checks green. The fleet-wide workflow is described in the hub's [CONTRIBUTING.md](https://github.com/bamr87/bamr87/blob/main/CONTRIBUTING.md). AI agents should start with [CLAUDE.md](CLAUDE.md).

## Changelog

Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

Released under the [MIT License](LICENSE).

---

<!-- bamr87-portfolio-footer -->
Part of the [bamr87 portfolio](https://github.com/bamr87) — browse every project on [Amr's GitHub profile](https://github.com/bamr87).
