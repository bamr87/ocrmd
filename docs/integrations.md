# Destinations and the library API

> How a page leaves ocrmd: the sink contract a receiver speaks (law-ai is the worked example), and `ocrmd.api.Library` for embedding the whole pipeline in another program.

## Destinations: handing a page to another system

ocrmd's only outbound path. A destination is any URL that speaks a small contract (the whole of it is in [ocrmd/sinks.py](../ocrmd/sinks.py)): `GET` answers with a *descriptor* — what the receiver is and which typed options it takes — and `POST` takes the *envelope*: the page's **reconciled** text, its provenance (document, page, page id, both agreement scores, the flag count, a link back to the image), the document's catalogue entry, and your options. Configure one in `ocrmd.toml`:

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

- **The text goes, the photograph does not.** A receiver with the picture and a vision model would re-read the page single-pass and overrule the cross-check, which is the one thing this pipeline makes that a single pass cannot. It gets the page id (the sha256 of the processed image) and a link back here, so a reviewer over there can always see the photograph.
- **A send is not a stage.** Nothing is written or marked; pressing the button twice is safe on this side, and a well-behaved receiver deduplicates and says so.
- **Failure is reported, never retried into silence.** An unreachable receiver is a 502 from the web UI, not a 500: this server is fine, the page just did not get through.

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
