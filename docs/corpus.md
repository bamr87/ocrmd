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
