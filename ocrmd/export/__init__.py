"""Exporters: a finished document turned into the shape something else wants.

Every exporter reads the same two things — the page files for the human-edited text and provenance,
and (where geometry matters) the page model `layout/<pid>.json` — and none re-derives the join between
them. One module per format, each exposing `export_document(s, st, slug, out)`:

    md      the built document (`documents/<slug>/<slug>.md`): every page under its anchor, a TOC
    txt     plain text, one page after another behind a separator line; the `==[…]==` marks reduced
            to a readable form rather than silently resolved
    json    the whole document as one JSON object: metadata, every page's frontmatter, body, layout,
            and the rows every extraction pipeline produced (`jsondoc`)
    jsonl   one JSON object per page — the shape a search index, a RAG loader or an ETL tool wants

`export_document` here is the one entry point; it records what was written in `doc_exports` so the
inventory can say which shapes a document already exists in.
"""
from __future__ import annotations
from pathlib import Path

from ..config import Settings
from ..store import Store

FORMATS = ("md", "txt", "json", "jsonl")


def default_path(s: Settings, slug: str, fmt: str) -> Path:
    return s.export_dir / slug / f"{slug}.{fmt}"


def export_document(s: Settings, st: Store, slug: str, fmt: str = "md", out: Path | None = None) -> Path:
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r} (have {', '.join(FORMATS)})")
    if slug not in st.documents():
        raise KeyError(f"no such document: {slug}")
    if fmt == "md":
        from . import markdown as mod
    elif fmt == "txt":
        from . import text as mod
    elif fmt == "json":
        from . import jsondoc as mod
    else:
        from . import jsonl as mod
    path = mod.export_document(s, st, slug, out)
    st.add_doc_export(slug, fmt, str(path))
    return path
