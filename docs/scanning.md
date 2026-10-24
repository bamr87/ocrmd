# Scanning with a webcam

> How to set up the camera rig, read the scan window, get captures the OCR can use, and tune the auto-shutter — with the measurement behind each default.

`ocrmd scan` watches a webcam over a book and captures each page once it has settled; every option is in the [CLI reference](reference/cli.md#scan). Indices shuffle between runs on macOS; `--camera c525` (a device-name substring) is stable.

A scan writes an OCR-only page file the moment a capture is processed and upgrades it in place when the model answers, so `documents/<slug>/pages/` is always usable, even mid-session. Quitting drains the queue; pressing Ctrl-C a second time leaves the remainder for `ocrmd verify`.

## Setting up the rig (textbook under a fixed webcam)

This is the setup the tool was tuned on, and what the defaults assume.

1. **Mount the camera looking straight down**, close enough that the scan window's **px/page** readout turns green (≥900 px across one page). A C525 framing a whole two-page spread at 1920×1080 gives each page ~700 px; framing one page at a time roughly doubles that. Resolution on the page is the only quality lever that measurably matters — see [Getting good captures](#getting-good-captures-from-a-webcam).
2. **Dark, matte surface** under the book. Page detection keys on paper being brighter than what surrounds it. A pale wooden desk works but a black mat works better.
3. **Even light from the side**, no lamp in the frame. Glare and a bright-lit hand both look like paper.
4. **Watch the focus line in the HUD.** ocrmd judges focus from the picture, not from the camera: webcams routinely report `continuous` while the lens sits somewhere useless (measured on the C525 this was built on: the same book at sharpness 18 and then 1030 minutes apart, with the driver claiming continuous autofocus throughout). It remembers what your rig has actually managed, warns when a page is far below that, and refocuses the camera itself. If it says it has given up, the lens is not the problem — check the distance, the light, and whether something is between the camera and the page.
5. **Run with `--focus settle`** (or `focus = "settle"` in `ocrmd.toml`): continuous autofocus until the first capture, then locked, so the lens stops hunting every time a hand comes in.
6. **Start the scan with your hands out of the frame, then turn pages and get your hand out.** The hand gate learns what "paper" looks like from the first frames it sees; a hand that is already there at start is part of the baseline and the first capture will include it (the gate corrects itself as soon as the hand leaves). After that the shutter waits while a hand covers the page (`HAND IN FRAME` in the window). A paperback that will not stay open wants a sheet of glass or a couple of weights on the margins — a hand holding it open blocks the shutter by design.
7. Watch the **advice line**: `page runs past the top edge — text is being cut off` and `printed page jumped 37 → 41` are both things to fix *now*, not at review time.

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

- **The detected page outline** drawn on the frame, and **px/page** in green once it clears the threshold. Frame edges that cut through text are marked in red.
- **Camera state**: resolution, focus mode, `*` while page sharpness is still moving, a warning when the page is soft for this rig, and the hand gate's numbers (`cover=` how much reference paper is hidden, `skin=` how skin-toned the hidden part is).
- **A work panel**: what is being processed right now (`page 0003: llm`), how many captures are queued, `LLM OFF` if the service has been given up on for this session, and the status of recent pages colour-coded by verification result.
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

## Getting good captures from a webcam

**Resolution per page is the dominant quality factor — it is the only one that measurably matters.** Frame averaging, sharper interpolation (Lanczos vs cubic), larger upscale targets, CLAHE and unsharp masking were each measured against Apple Vision's word yield on a real page: every one landed inside the noise (143-148 words). Going from 434 to 728 pixels across the page moved it from 98 to 146. Post-processing cannot recover detail the optics never captured, so ocrmd does none of it and tells you about framing instead — live in the scan window, and on every page:

```
page 0001 -> 0001.md  [suspect agreement=0.22 overlap=0.33]
    page is only 521px wide; body text needs ~900px+ — move the camera closer or fill the frame with one page
```

- Orientation is detected automatically and upside-down books are usually fine — but see the warning under [How orientation is decided](#how-orientation-is-decided): on this corpus the cascade is measurably weak.
- Page detection keys on the page being brighter than what surrounds it — a dark desk or mat helps.
- Uneven light across a spread hurts the far page more than anything else.
- ocrmd opens the camera's largest mode. Modes above the sensor size (the C525 advertises 1920x1080 over a 1280x960 sensor) were assumed to add nothing; measured on a real page, the opposite holds: **146 correctly-read words at 1920x1080 against 98 at 1280x960**. Override with `--resolution 1280x960`.
- Webcams often power up with focus *locked* at lens position 0 — every frame blurry. `scan` sets continuous autofocus (and auto-exposure) through AVFoundation before opening the stream, because OpenCV reports `-1` for every camera property on macOS and ignores writes ([camctl.py](../ocrmd/camctl.py)).

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

A photographed page can arrive at any of four rotations and no single test is reliable, so [preprocess.py](../ocrmd/preprocess.py) cascades three, strongest evidence first:

1. **Projection profile** — text lines form horizontal bands, so the power of the row profile at text-line frequencies says whether text runs horizontally. Narrows to `{0,180}` or `{90,270}`.
2. **Dictionary probe** — a fast OCR pass counts real words at each remaining rotation. The *fast* Vision recogniser does not auto-orient (the accurate one does), which is what makes it usable here. Decides when it finds enough words with a clear margin; abstains otherwise.
3. **Ink asymmetry** — Latin text puts more ink above each line's centre than below (capitals and ascenders outnumber descenders). Weak but never silent, so it breaks the tie on pages far too small for step 2.

**On real captures this cascade is close to a coin toss**: measured per stage at 57 %, 25 % and 53 % correct, about 30 % end to end, with 10 % of stored crops sideways. `orientation_probe = true` replaces it with an OCR probe over all four rotations, which is measurably better but costs 4–8 Vision passes a page, so it is off by default; its note in the [settings reference](reference/settings.md#preprocess) also records a crash it triggers inside RapidOCR's ONNX detector on the machine it was measured on. Turn it on and run `ocrmd reprocess` to re-derive a book.

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

`ocrmd.toml` (each key's default and note: [settings reference](reference/settings.md)): `motion_threshold`, `stable_frames`, `settle_frames`, `geometry_frames`, `geometry_tolerance`, `novelty_threshold`, `blur_threshold`, `occlusion_cover`, `occlusion_rebaseline`, `focus_bracket*`, `admit_*`. The preview window shows the live gate state and numbers, and every capture's sharpness/occlusion/px-per-page goes to `ocrmd.log`.

Capturing too seldom? Lower `settle_frames` (the wait after a hand) or raise `geometry_tolerance` (how much the page outline may wander). Still capturing during page turns? Raise them.
