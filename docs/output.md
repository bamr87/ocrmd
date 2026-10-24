# Output

> What ocrmd writes under its root, what each page's verification status means and what to do about it, and which of your edits survive a re-run.

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

### Re-deriving pages

`reprocess` exists because a page's id is the hash of its *processed* image: when preprocessing improves (better cropping, rotation, spread splitting) the pages themselves change. It re-derives every page from the untouched raw captures and discards the stale OCR/LLM results, which means the LLM pass runs again and costs tokens. Raw captures are never modified.
