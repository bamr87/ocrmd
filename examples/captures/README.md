# Example captures

> Real webcam frames and two generated PDFs from this project's own scans, for trying the pipeline without a camera — and what each one exercises.

Real webcam frames from this project's own scans (Logitech C525 over a desk), plus two generated PDFs, for trying the pipeline without a camera:

```sh
export OCRMD_ROOT=/tmp/ocrmd-example            # keep the repo's own store untouched
ocrmd ingest examples/captures --doc example --no-llm    # local OCR only, free
ocrmd verify --doc example                               # add the LLM pass (needs credentials)
ocrmd serve                                              # http://127.0.0.1:8765
```

| File | What it exercises |
|---|---|
| `lsat-page-verified-{1,2,3}.jpg` | single textbook pages that reconcile cleanly (~850–940 px across the page) |
| `lsat-page-reviewed.jpg` | a page a person signed off after review |
| `lsat-page-lowres-574px.jpg` | the resolution floor: body text at ~574 px, where OCR starts to fail |
| `spread-two-pages.jpg` | a two-page spread; `preprocess` finds the gutter and splits it into two pages |
| `textbook-spread-wide.jpg` | a wide spread whose right half comes out at ~1100 px |
| `focus-bracket-primary.jpg` + `.v1.jpg` | one capture and a second exposure at another lens position (`multifocus`) — ingest the primary, the variant rides along only from `ocrmd scan` |
| `born-digital.pdf` | a PDF with a text layer and one placed image: the text layer becomes the primary OCR reading, the image a figure block |
| `scanned-image-only.pdf` | a PDF that is only a picture of a page: no text layer, the engines read it |

The raw frames are immutable inputs; nothing in the pipeline modifies them.
