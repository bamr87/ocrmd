"""One JSON object per page, one page per line — the shape a search index, a RAG loader or an ETL
tool ingests without a parser. Each row names its page and carries the cross-check's verdict, so a
consumer can filter on `status` or `agreement` and can always get back to the photograph."""
from __future__ import annotations
import json
from pathlib import Path

from ..config import Settings
from ..store import Store
from .. import render, review
from . import default_path

FORMAT = "jsonl"


def page_row(s: Settings, st: Store, slug: str, page) -> dict | None:
    p = render.page_path(s.documents / slug, page["sequence"])
    if not p.exists():
        return None
    fm, body = render.read_page(p)
    v = fm.get("verification") or {}
    src = fm.get("source") or {}
    return {
        "document": slug, "sequence": page["sequence"], "page_id": page["id"],
        "page_label": fm.get("page_label") or page["page_label"], "status": v.get("status") or page["status"],
        "kind": fm.get("kind"), "language": fm.get("language"), "captured_at": fm.get("captured_at") or page["captured_at"],
        "context": fm.get("context") or {},
        "text": body,
        "unresolved": review.unresolved(body),
        "verification": {"agreement": v.get("agreement"), "overlap": v.get("overlap"), "cer": v.get("cer"),
                         "flagged_spans": v.get("flagged_spans"), "reviewed_by": v.get("reviewed_by")},
        "source": {"type": src.get("type"), "device": src.get("device"), "sha256": src.get("sha256"), "part": src.get("part")},
        "ocr": {"engine": (fm.get("ocr") or {}).get("engine"), "confidence": (fm.get("ocr") or {}).get("confidence")},
        "llm": {"provider": (fm.get("llm") or {}).get("provider"), "model": (fm.get("llm") or {}).get("model")} if fm.get("llm") else None,
    }


def rows(s: Settings, st: Store, slug: str) -> list[dict]:
    return [r for r in (page_row(s, st, slug, p) for p in st.pages(slug)) if r is not None]


def export_document(s: Settings, st: Store, slug: str, out: Path | None = None) -> Path:
    out = out or default_path(s, slug, FORMAT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows(s, st, slug):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(out)
    return out
