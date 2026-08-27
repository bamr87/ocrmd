# ocrmd

Incremental scanning + OCR to cross-checked Markdown. Point a webcam at a book, turn pages; each still, new, sharp, **unobstructed** page is captured, OCR'd locally (Apple Vision + RapidOCR voting), transcribed by Claude with vision, and the two readings are diffed into a Markdown page file with rich frontmatter. Built and tuned on a fixed camera over a textbook. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [CHANGELOG.md](CHANGELOG.md) for what changed.

## Install (macOS)

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'   # add ,pdf for PDF ingest; ,tesseract as a last resort
.venv/bin/ocrmd doctor          # engines, cameras, credentials, database health
.venv/bin/python -m pytest      # 69 checks, ~1 min (needs a local OCR engine for the pipeline ones)
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

- **Scan** — live HUD stream from the camera with the gate state (`READY`, `HAND IN FRAME`, …), meters,
px/page, and buttons/keys for capture, focus lock/auto, rotate, zoom, pan. Pick the document, camera, focus mode, OCR-only, optional duration/page limit. Event log and queue state update live.
- **Documents** — status bars, page table filterable by status (label, width, agreement, overlap,
advice, continuity, LLM errors), drag-and-drop image ingest, Verify/Build/Reprocess buttons, download of the built `document.md`. Select pages to **delete** them (optionally both halves of a spread); **Delete…** removes the whole document after you type its slug.
- **Page** — the page image (processed / raw spread / binarised, zoomable) beside the transcription.
*Edit* tab: each disagreement with one-click **OCR / LLM / type your own**, editable Markdown body. *Preview* tab: rendered Markdown with the `==[…]==` marks highlighted. *Meta* tab: printed page label, chapter/section/running head/tags, verification numbers, model notes, provenance. **Sign off** marks a page `reviewed` (body frozen); **Reopen** undoes it; **Delete…** removes the page (and offers to take its spread partner with it). Keys: `n`/`p` next/previous, `N` next flagged. Review mode (`Review N flagged` on a document) advances to the next flagged page after each sign-off.
- **Jobs** — verify / build / reprocess with options, live log, progress, cancel.
- **Settings** — `document.yaml` editor per document, effective settings, full doctor report.

Edits made in the UI are human edits: they survive later `verify` runs exactly like edits made to the files. The server binds to `127.0.0.1` and has no authentication — keep it that way (`--host` exists for a trusted LAN only). The API is documented at `/api/docs`.

A scan writes an OCR-only page file the moment a capture is processed and upgrades it in place when the model answers, so `documents/<slug>/pages/` is always usable, even mid-session. Quitting drains the queue; pressing Ctrl-C a second time leaves the remainder for `ocrmd verify`.

`reprocess` exists because a page's id is the hash of its *processed* image: when preprocessing improves (better cropping, rotation, spread splitting) the pages themselves change. It re-derives every page from the untouched raw captures and discards the stale OCR/LLM results, which means the LLM pass runs again and costs tokens. Raw captures are never modified.

## Setting up the rig (textbook under a fixed webcam)

This is the setup the tool was tuned on, and what the defaults assume.

1. **Mount the camera looking straight down**, close enough that the scan window's **px/page** readout
turns green (≥900 px across one page). A C525 framing a whole two-page spread at 1920×1080 gives each page ~700 px; framing one page at a time roughly doubles that. Resolution on the page is the only quality lever that measurably matters — see below.
2. **Dark, matte surface** under the book. Page detection keys on paper being brighter than what
   surrounds it. A pale wooden desk works but a black mat works better.
3. **Even light from the side**, no lamp in the frame. Glare and a bright-lit hand both look like paper.
4. **Run with `--focus settle`** (or `focus = "settle"` in `ocrmd.toml`): continuous autofocus until
   the first capture, then locked, so the lens stops hunting every time a hand comes in.
5. **Start the scan with your hands out of the frame, then turn pages and get your hand out.** The
hand gate learns what "paper" looks like from the first frames it sees; a hand that is already there at start is part of the baseline and the first capture will include it (the gate corrects itself as soon as the hand leaves). After that the shutter waits while a hand covers the page (`HAND IN FRAME` in the window). A paperback that will not stay open wants a sheet of glass or a couple of weights on the margins — a hand holding it open blocks the shutter by design.
6. Watch the **advice line**: `page runs past the top edge — text is being cut off` and
   `printed page jumped 37 → 41` are both things to fix *now*, not at review time.

## The scan window

`scan` opens a live preview with a heads-up display:

- **Gate state** with meters, so it is obvious *why* the shutter is holding off:

  | State | Meaning |
  |---|---|
  | `MOVING` | frame-to-frame motion above threshold (a page turning, a hand moving) |
  | `FOCUSING` | the lens is hunting |
  | `HAND IN FRAME` | paper that was visible a moment ago is covered (hand, pen, curled page) |
  | `SAME PAGE` | looks like the last capture — turn the page |
  | `BLURRY` | sharpness below the floor (absolute, and relative to the best this rig has produced) |
  | `READY` | capturing on the next tick |

- **The detected page outline** drawn on the frame, and **px/page** in green once it clears the threshold.
  Frame edges that cut through text are marked in red.
- **Camera state**: resolution, focus mode, `*` while the lens is hunting, and the hand gate's
  numbers (`cover=` how much reference paper is hidden, `skin=` how skin-toned the hidden part is).
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
ocrmd.db                           SQLite: pages, captures, stages, llm_calls, batches
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

- Orientation is detected automatically and upside-down books are fine (see below).
- Page detection keys on the page being brighter than what surrounds it — a dark desk or mat helps.
- Uneven light across a spread hurts the far page more than anything else.
- ocrmd opens the camera's largest mode. Modes above the sensor size (the C525 advertises 1920x1080
over a 1280x960 sensor) were assumed to add nothing; measured on a real page, the opposite holds: **146 correctly-read words at 1920x1080 against 98 at 1280x960**. Override with `--resolution 1280x960`.
- Webcams often power up with focus *locked* at lens position 0 — every frame blurry. `scan` sets
continuous autofocus (and auto-exposure) through AVFoundation before opening the stream, because OpenCV reports `-1` for every camera property on macOS and ignores writes ([camctl.py](ocrmd/camctl.py)).

### How orientation is decided

A photographed page can arrive at any of four rotations and no single test is reliable, so [preprocess.py](ocrmd/preprocess.py) cascades three, strongest evidence first:

1. **Projection profile** — text lines form horizontal bands, so the power of the row profile at
   text-line frequencies says whether text runs horizontally. Narrows to `{0,180}` or `{90,270}`.
2. **Dictionary probe** — a fast OCR pass counts real words at each remaining rotation. The *fast*
Vision recogniser does not auto-orient (the accurate one does), which is what makes it usable here. Decides when it finds enough words with a clear margin; abstains otherwise.
3. **Ink asymmetry** — Latin text puts more ink above each line's centre than below (capitals and
ascenders outnumber descenders). Weak but never silent, so it breaks the tie on pages far too small for step 2.

## Tuning the auto-shutter

`ocrmd.toml`: `motion_threshold`, `stable_frames`, `novelty_threshold`, `blur_threshold`, `occlusion_cover`, `occlusion_rebaseline`. The preview window shows the live gate state and numbers, and every capture's sharpness/occlusion/px-per-page goes to `ocrmd.log`.
