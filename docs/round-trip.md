# Rebuilding and annotating pages

> `ocrmd reconstruct` rebuilds each page from its Markdown and scores it against the photograph; `ocrmd annotate` folds pen and highlighter marks made on scanned paper back into the Markdown.

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
