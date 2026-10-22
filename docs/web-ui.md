# The web UI

> What each view of `ocrmd serve` shows and does — library, scan, documents, page, jobs and settings — and the rules it shares with the CLI.

```sh
ocrmd serve            # http://127.0.0.1:8765, opens a browser tab
```

Everything above, in a browser, plus what a terminal cannot show:

- **Library** — every document as a card with its catalogue entry (title, author, year, kind, printed page range), status bar, what needs a re-shoot or a review, and download links for every export format; a search box over every page of every document, with hits that link to the page they sit on.
- **Scan** — live HUD stream from the camera with the gate state (`READY`, `HAND IN FRAME`, …), meters, px/page, and buttons/keys for capture, focus lock/auto, rotate, zoom, pan. Pick the document, camera, focus mode, OCR-only, optional duration/page limit. Event log and queue state update live.
- **Documents** — status bars, page table filterable by status (label, width, agreement, overlap, advice, continuity, LLM errors), drag-and-drop image ingest, Verify/Build/Reprocess buttons, download of the built `document.md`. Select pages to **delete** them (optionally both halves of a spread); **Delete…** removes the whole document after you type its slug.
- **Page** — the page image beside the transcription, in four views: **flat** (the default — the crop flattened and contrast-stretched, the page as a flatbed scanner would have delivered it), **processed** (the crop the model sees), **bin** (the adaptive threshold older engines are given) and **raw** (the untouched frame). Zoomable, and **drag a box on it to read that region on its own** — the selection is cut out, enlarged on its own terms and re-read by the local engines, with an **Ask the model** button for the ones they still cannot get. The result comes back beside its crop, to copy or to insert at the cursor in the editor; nothing is written to the page until you save. *Edit* tab: each disagreement with one-click **OCR / LLM / type your own**, editable Markdown body. *Preview* tab: rendered Markdown with the `==[…]==` marks highlighted. *Meta* tab: printed page label, chapter/section/running head/tags, verification numbers, model notes, provenance. **Sign off** marks a page `reviewed` (body frozen); **Reopen** undoes it; **Delete…** removes the page (and offers to take its spread partner with it). Keys: `n`/`p` next/previous, `N` next flagged. Review mode (`Review N flagged` on a document) advances to the next flagged page after each sign-off. **Send…** (present only when a destination is configured) hands the page's reading, or a region's, to a receiver — see [Destinations](integrations.md#destinations-handing-a-page-to-another-system).
- **Jobs** — verify / build / reprocess / rebuild / extract with options, live log, progress, cancel.
- **Settings** — `document.yaml` editor per document, effective settings, full doctor report.

Edits made in the UI are human edits: they survive later `verify` runs exactly like edits made to the files. The server binds to `127.0.0.1` and has no authentication — keep it that way (`--host` exists for a trusted LAN only). The API is documented at `/api/docs`.
