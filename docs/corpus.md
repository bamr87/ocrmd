# Inventory, search, exports and extraction

> What to do with pages once they are scanned: the SQLite inventory and full-text search, the four export formats, and structured-data extraction with `ocrmd etl`.

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
| `crosscheck` | diff two list fields that are two readings of the same thing; keep every item and say who read it | no |

```sh
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --dry-run   # see the prompts, spend nothing
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --limit 5   # try it on five pages
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml             # -> documents/lsat/extracted/cases.csv
```

Output is JSON, CSV or a Markdown table; `output: {fields: [...]}` narrows the columns (the provenance ones are always written). The web UI has the same thing: put a spec in `documents/<slug>/etl/*.yaml` (or `<root>/etl/pipelines/`) and an **Extract** panel appears on the document page, with a dry run, a live log and the result to download. [examples/cases.pipeline.yaml](../examples/cases.pipeline.yaml) is a worked example.

Three things are ocrmd's rather than DocETL's, and they are why this is built in rather than bolted on:

- **Every row carries the page it came from** (`_page`, `_page_id`, `_document`; `_pages` after a reduce), so an extracted claim can be taken back to the photograph it was read from.
- **Nothing is inferred.** The operators run under the same refusal as the transcription stage: use what the page says, leave the field empty when it does not say it.
- **Answers are cached per operator and versioned by the prompt**, so editing one operator re-runs one operator and re-running an unchanged pipeline costs nothing — the same bargain as `verify`.

The implementation is ocrmd's own (six operators over its existing provider, no new dependencies). To use DocETL itself — for its optimizer, which rewrites pipelines — export the same spec:

```sh
pip install 'ocrmd[docetl]'
ocrmd etl --doc lsat -p examples/cases.pipeline.yaml --export-docetl out/
docetl run out/pipeline.yaml
```

## What the figures show: components

The transcription stops at a figure's edge on purpose — the page body carries the printed caption and a reference to the crop, never a description of the picture. For a document whose content *is* its pictures (an assembly manual, a parts diagram), the bundled `components` pipeline names what each figure shows, one row per part, fastener, tool or cable:

```sh
ocrmd etl --doc manual -p components --dry-run   # the prompts and the call count; nothing sent
ocrmd etl --doc manual -p components             # -> documents/manual/extracted/components.json
ocrmd etl --doc build -p components --source pages   # whole captured images: a photograph of the parts themselves
```

A picture has no local engine to be read by, so one model pass over a figure would be exactly the single reading the rest of ocrmd distrusts. The pipeline therefore reads every figure **twice, on different evidence** — once as the crop alone, once as the crop with the page's own text — and `crosscheck` diffs the answers the way OCR and the model are diffed for text:

| field | meaning |
|---|---|
| `readings` | `both` — the two readings named it; `first` / `second` — only one did. A single reading is kept and marked, never dropped and never promoted |
| `second_reading` | every field on which the second reading said something else, with what it said |
| `conflicts` | the fields where the two disagree in **digits** (a quantity, a size). Both values stay; neither is chosen |
| `printed_as_found` | whether the label the model says is printed occurs in the page text. `false` is a label no reading of the page contains |
| `_crosscheck` | per figure: how many items both readings named, how many only one did, and `missing` when a reading never happened |

Two names are one item when they read alike with the words in any order (`etl.same_thing`, at 0.80), or when both readings quote the same printed label. Names that differ in a number are never merged — an M3 screw is not an M4 screw — and a name that merely contains another is not the same thing ("fan", "fan bracket"). The matcher is deliberately the kind that under-merges: "PSU" and "power supply unit" come out as two single readings unless the page prints the name both point at.

What this is not yet: **it has not been run against a live model** — the operator, the prompts and the pipeline are tested with a stand-in provider, and the 0.80 was set on 25 hand-written part-name pairs, not on two real readings. Two passes by one model are also a weaker second opinion than OCR against a model; for one that is independent in more than its evidence, copy the spec to `documents/<slug>/etl/components.yaml` and give `look_again` a `model:` of its own. It costs two calls a figure. Over `--source pages` an `image: true` operator is shown the page's whole processed crop instead of a figure crop, which is what a photograph of objects needs — but such a capture still goes through the *page* pipeline first, which was built for paper: the first photograph of a motherboard and a graphics card on a desk was split down the middle as a two-page spread and both halves turned upside down, and a capture with no readable text at all has an empty body and is not offered to operators. A capture mode that keeps a scene whole does not exist yet. And it can only name what the camera resolved: see [illustrated pages](scanning.md#illustrated-pages-manuals-diagrams-assembly-steps) for the framing that makes a figure legible at all.

