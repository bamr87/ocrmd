"""The document as one JSON object — the machine-readable shape everything else can be built from.

Nothing is computed here that is not already on disk: `document.yaml`, each page's frontmatter and
body, its `layout/<pid>.json` (built on demand when the stage is behind), and the rows under
`documents/<slug>/extracted/*.json`. It is the union of ocrmd's artifacts with the page as the join key.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..config import Settings
from ..store import Store
from .. import pipeline, render
from . import default_path

FORMAT = "json"


def page(s: Settings, st: Store, row) -> dict:
    slug = row["document"]
    path = render.page_path(s.documents / slug, row["sequence"])
    fm, body = render.read_page(path) if path.exists() else ({}, "")
    try:
        lay = pipeline.page_layout(s, st, row).model_dump()
    except Exception:                           # artifacts gone: the page is still worth exporting
        lay = None
    return {"id": row["id"], "sequence": row["sequence"], "page_label": row["page_label"], "status": row["status"],
            "fidelity": row["fidelity"] if "fidelity" in row.keys() else None,
            "frontmatter": fm, "body": body, "layout": lay}


def extractions(s: Settings, slug: str) -> dict[str, list]:
    out: dict[str, list] = {}
    d = s.documents / slug / "extracted"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(rows, list):
            out[p.stem] = rows
    return out


def document(s: Settings, st: Store, slug: str) -> dict:
    meta = render.load_document_meta(s.documents / slug)
    return {"ocrmd": 1, "document": slug, "meta": meta,
            "pages": [page(s, st, r) for r in st.pages(slug)],
            "extractions": extractions(s, slug)}


def export_document(s: Settings, st: Store, slug: str, out: Path | None = None) -> Path:
    out = out or default_path(s, slug, FORMAT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document(s, st, slug), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(out)
    return out
