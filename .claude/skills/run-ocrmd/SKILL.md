---
name: run-ocrmd
description: Build, run, and drive ocrmd — the webcam/image-to-Markdown scanner. Use when asked to start ocrmd, serve its web UI, run its tests, ingest pages, take a screenshot of the UI, check the camera/scan path, or verify a change works in the real app.
---

ocrmd is a Python CLI (`ocrmd`) plus a FastAPI web UI (`ocrmd serve`) — and, since 0.9, a library (`ocrmd.api.Library`) — that OCRs page images twice — local engines and a vision LLM — and diffs the readings into Markdown, keeps an inventory of everything scanned (`ocrmd library`, `ocrmd search`), exports it as md/txt/json/jsonl, and can hand a page to a configured destination (`ocrmd send`). Drive it with **`.claude/skills/run-ocrmd/driver.py`**: it renders synthetic book pages, runs them through the real pipeline, exercises the JSON API, screenshots every SPA view through Chrome, and can open the live webcam. No book, no camera, and no network required for the default path.

All paths below are relative to the repo root. Verified on macOS 15 (arm64), Python 3.14.

## Prerequisites

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'
```

`ocrmd` and `python` then live in `.venv/bin/`. Extras: `pdf` (PDF ingest), `tesseract` (weak third engine). On macOS Apple Vision comes free with `pyobjc-framework-Vision`; off macOS RapidOCR is the primary engine and the `rapidocr` extra is not optional.

Screenshots use `npx` + the installed **Google Chrome** (`/Applications/Google Chrome.app`). Nothing else to install — see Gotchas for why we don't use Playwright's own browsers.

Credentials are optional. Everything except `driver.py llm` runs offline. For the LLM pass put `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN` + `OCRMD_MODEL=claude-haiku-5-5`) in `.env`.

```bash
.venv/bin/ocrmd doctor
# root:        /Users/you/github/ocrmd
# ocr:         ocr@2:apple_vision+rapidocr   installed: apple_vision, rapidocr, tesseract
# anthropic:   credentials found
# camera 0:    HD Webcam C525
# database:    /Users/you/github/ocrmd/ocrmd.db integrity=ok
```

## Run (agent path)

```bash
.venv/bin/python .claude/skills/run-ocrmd/driver.py all
```

Wipes `/tmp/ocrmd-drive`, then: renders 2 synthetic pages → `ocrmd doctor` → `ingest --no-llm` → `status` → `build` → `library` / `search` / `export` (every format) → `sinks` → serves the API and exercises it (documents, pages, images, region read, edit + sign-off, a build job, the library, search, exports, the empty destinations list) → screenshots seven SPA views. ~90 s, exit 0 only if every assertion held. Last verified run:

```
[driver] pipeline OK: 2 page file(s), 2946 B document, 2 jsonl row(s)
[driver] GET /api/documents/smoke -> 2 page(s), statuses=['unverified', 'unverified']
[driver] GET /api/search?q=Bennet -> 1 hit(s), first: '…My dear Mr [Bennet] said his lady…'
[driver] GET /api/documents/smoke/export.jsonl -> 4211 B
[driver] GET .../pages/1/image/processed -> 936524 B PNG
[driver] PUT body + POST signoff -> status=reviewed, edit present
[driver] POST /api/jobs build -> done
[driver] shot page: /tmp/ocrmd-shots/page.png (206755 B)
```

| command | what it does |
|---|---|
| `all` | `pipeline` + `api` + `shots`. Offline. The default check. |
| `pipeline` | fixtures → ingest → OCR → status → build → library/search/export/sinks; asserts the OCR actually read the text and that search finds page 2 |
| `api` | serves, then GETs documents/pages/images/doctor/library/search/exports/sinks, PUTs a body edit, signs the page off, runs a build job, and checks the job did not clobber the edit |
| `shots` | serves, screenshots library / search / document / page / scan / jobs / settings |
| `camera` | opens the real webcam for 8 s via `POST /api/scan/start`, watches the gate states, saves a HUD frame, then deletes what it captured |
| `llm` | `ocrmd verify` against the real provider — **costs tokens** |
| `serve` | `ocrmd serve` in the foreground against the throwaway root |
| `fixtures` | just write the synthetic page PNGs |

Flags: `--root` (throwaway `OCRMD_ROOT`, default `/tmp/ocrmd-drive`), `--keep` (reuse it instead of wiping), `--port` (default 8799), `--shots-dir` (default `/tmp/ocrmd-shots`), `--seconds` (camera).

**Screenshots land in `/tmp/ocrmd-shots/`** — `library.png`, `search.png`, `document.png`, `page.png`, `scan.png`, `jobs.png`, `settings.png`, and `scan-frame.jpg` from the camera check. `page.png` is the one to look at: processed page image beside the transcription, status badge, edit/preview/meta tabs.

### The LLM pass

Run it right after `pipeline`, before `api` — `verify` skips pages that are already `reviewed`, and `api` signs page 1 off.

```bash
.venv/bin/python .claude/skills/run-ocrmd/driver.py pipeline
.venv/bin/python .claude/skills/run-ocrmd/driver.py llm
#   page 0001 -> 0001.md  [verified agreement=1.00 overlap=1.00]
#       kept your edits: context.title, context.tags
#   page 0002 -> 0002.md  [verified agreement=1.00 overlap=1.00]
#   smoke: 2 page(s) — 2 verified
#   anthropic/claude-haiku-4-5-20251001: 2 calls, in=7348 out=836
```

Two vision calls on synthetic Austen text. `agreement=1.00` means the LLM and local OCR read the page identically; the printed page numbers (12, 13) come back as `label`.

### The camera

```bash
.venv/bin/python .claude/skills/run-ocrmd/driver.py camera --seconds 6
# [driver] GET /api/cameras -> ['HD Webcam C525', 'FaceTime HD Camera']
# [driver] POST /api/scan/start -> running on HD Webcam C525
# [driver] gates seen: ['moving', 'READY', 'same page']
# [driver] cleanup: 4 page(s), 20 artifact(s), 2 raw capture(s) removed
```

This is the only way to check the capture lane without sitting at the rig. It is OCR-only, so nothing leaves the machine, and it deletes the `camera-check` document afterwards. `scan-frame.jpg` shows the live HUD: gate state, meters, the detected page outline, framing advice, work panel.

## Run (human path)

```bash
.venv/bin/ocrmd serve                      # http://127.0.0.1:8765, opens a browser tab
.venv/bin/ocrmd scan --doc lsat --camera c525 --focus settle   # OpenCV preview window, `q` to quit
```

Both use the repo's own store (`raw/`, `processed/`, `documents/`, `ocrmd.db`). `scan` needs a camera pointed at a book and macOS camera permission for the terminal app.

## Sending a page to a destination (Law AI is the worked example)

Off unless a receiver is named, so nothing in the default path touches one. A destination is any URL that speaks the sink contract in [ocrmd/sinks.py](../../../ocrmd/sinks.py) (`GET` → descriptor, `POST` → envelope):

```sh
export OCRMD_SINKS='[{"name":"law-ai","url":"http://localhost:8000/api/lsat/intake/ocr"}]'
.venv/bin/ocrmd sinks                                             # probes the descriptor: name, options, reachable?
.venv/bin/ocrmd send --doc lsat --page 15 --to law-ai -o hint="question 15" --dry-run   # prints the envelope, sends nothing
.venv/bin/ocrmd send --doc lsat --page 15 --to law-ai -o hint="question 15"             # posts; the receiver's summary comes back
```

For law-ai it needs that API up (`bash .claude/skills/run-law-ai/driver.sh` with `KEEP=1`, in the sibling repo) and costs tokens *on law-ai's side* — the extraction runs there, through its provider chain, not through ocrmd's. Its second door, `…/api/matters/<slug>/intake/ocrmd`, files the page as a source document on a matter and needs a user bearer token (`bearer` in the sink config) because matters are per-account. `GET /api/sinks` lists what is configured with descriptors; with none the "Send…" button is absent from the UI and `ocrmd send` refuses.

## Embedding (the library API)

```python
from ocrmd.api import Library
lib = Library("/tmp/ocrmd-drive", provider=None)          # OCR only; "auto" uses the configured provider
lib.ingest(["/tmp/ocrmd-drive/fixtures"], doc="smoke")
lib.search("Bennet")[0].sequence                     # 2
lib.export("smoke", "jsonl")
```

`tests/test_api.py` drives it the way a host program would. law-ai's plugin (`backend/law_ai/ingestionapp/services/ocrmd_plugin.py` in the sibling repo) is built on exactly this surface, with `provider=None`.

## Test

```bash
.venv/bin/ruff check ocrmd tests    # -> All checks passed!
.venv/bin/python -m pytest          # -> 306 checks, ~2 min
```

Both are the CI gates ([.github/workflows/ci.yml](../../../.github/workflows/ci.yml), macOS runner). Single test: `.venv/bin/python -m pytest tests/test_pipeline.py::test_name -q`. Tests that need an OCR engine skip when none is installed, so a green suite does not prove OCR works — `driver.py pipeline` does.

## Gotchas

- **The repo *is* the data root.** With no `OCRMD_ROOT`, ocrmd writes `raw/`, `processed/`, `ocr/`,
`llm/`, `documents/` and `ocrmd.db` next to the source, alongside a real scan store. The driver always sets `OCRMD_ROOT` to a throwaway directory; keep it that way, and pass `--root` rather than pointing anything at the repo.
- **Identical fixture text is silently deduped.** Two pages whose OCR token overlap is ≥ 0.85
within one document collapse to one, reported only as `duplicate: page-02.png`. The driver's two fixture bodies are deliberately different, and it fails loudly if `duplicate:` appears.
- **`GET /api/documents/{slug}` returns `pages` as an integer count.** The per-page table is under
  `page_list`. Indexing `["pages"]` gets you `TypeError: object of type 'int' has no len()`.
- **API field names are not what you'd guess**: sign-off takes `{"who": ...}` (not `reviewer`), job
  creation takes `{"kind", "doc", "options"}` (not `slug`), and job state is `status` (not `state`),
  one of `queued|running|done|failed|cancelled`.
- **The page route is `#/doc/<slug>/page/<seq>`.** `#/doc/<slug>/<seq>` falls through the router to
the *document* view and renders a perfectly plausible wrong screenshot. Every screenshot in the driver is gated on a `--wait-for-selector` for exactly this reason: `load` fires on the empty shell, so a broken route still produces a pretty PNG.
- **Don't run `npx playwright install`.** Playwright's CLI insists on a browser build matching its
own pinned revision and, worse, *prunes* cached builds belonging to other Playwright versions — running it here deleted the chromium the Playwright MCP was using. `--channel chrome` drives the installed Google Chrome instead: no download, no revision matching.
- **uvicorn installs its own SIGINT handler.** Killing the `ocrmd serve` PID with SIGTERM alone can
leave it up; the driver launches it with `start_new_session=True` and signals the whole process group.
- **An old editable install lies about its version.** `pip list` showed `ocrmd 0.1.0` against a
0.4.0 tree until `pip install -e '.[dev,rapidocr]'` was re-run. Re-run it if `doctor` or the UI looks a version behind.
- **Fixture pages need a real font and a dark background.** PIL's `load_default()` bitmap font OCRs
badly, and page detection keys on paper being *brighter than its surround* — the driver draws Times New Roman on cream paper inset in a dark-grey "desk" margin. A page image with no margin gets no detected page rectangle.
- **The HUD renders non-ASCII as `?`.** OpenCV's Hershey fonts are ASCII-only, so the advice line's
  em-dashes come out as `???` in `scan-frame.jpg`. Cosmetic, not a corrupted string.
- **A subscription OAuth token only serves `claude-haiku-5-5`.** Any other model returns a bare 429.
Set `OCRMD_MODEL=claude-haiku-5-5` alongside `CLAUDE_CODE_OAUTH_TOKEN`, or use an `ANTHROPIC_API_KEY`.

## Troubleshooting

- **`Executable doesn't exist at .../chromium_headless_shell-<rev>/chrome-mac/headless_shell`**:
the pinned Playwright wants a browser revision you don't have cached. Don't install it — the driver already passes `--channel chrome`. If Chrome isn't installed, `--channel msedge` works the same way.
- **`[driver] FAIL: screenshot <view> failed (selector '<sel>' never appeared)`**: the SPA route
changed or the view threw. Open `http://127.0.0.1:8799/#/...` via `driver.py serve` and check the browser console; the router is at the bottom of [ocrmd/web/static/app.js](../../../ocrmd/web/static/app.js).
- **`[driver] FAIL: serve exited early`** with `address already in use`: a previous run's server is
  still listening. `lsof -ti:8799 -sTCP:LISTEN | xargs kill`.
- **`[driver] FAIL: OCR did not read the fixture text`**: no OCR engine is installed or importable.
`.venv/bin/ocrmd doctor` prints the installed set; on macOS reinstall `pyobjc-framework-Vision`, elsewhere install the `rapidocr` extra.
- **`[driver] FAIL: no camera reported by /api/cameras`**: macOS camera permission. System
Settings → Privacy & Security → Camera, enable the terminal/IDE running the driver. `ocrmd doctor` prints `camera: NOT authorized` when this is the cause.
- **`[driver] FAIL: a fixture page was dropped as a duplicate`**: the two `BODIES` in `driver.py`
  drifted too close together. Make them share fewer words.
