"""What a client is shown of a document or a page, shared by the browser UI's `/api` and the phone app's `/api/v1`.

Each builder takes the URL prefix its client speaks (`base`), so an image link handed to the phone points at
the authenticated `/api/v1` route and one handed to the browser at the route it has always used.
"""
from __future__ import annotations
import logging

from ..config import Settings
from ..store import Store
from .. import library as library_mod, pipeline, render, review

log = logging.getLogger("ocrmd.web")


def image_paths(s: Settings, row) -> dict:
    """The page's pictures that exist on disk. `flat` is derived from the crop and rendered on first request."""
    pid = row["id"]
    return {k: p for k, p in (("flat", s.processed / f"{pid}.png"), ("processed", s.processed / f"{pid}.png"),
                              ("bin", s.processed / f"{pid}.bin.png"), ("raw", s.raw / f"{row['raw_sha256']}.jpg"))
            if p.exists()}


def page_payload(s: Settings, st: Store, slug: str, row, base: str = "/api") -> dict:
    path = render.page_path(s.documents / slug, row["sequence"])
    fm, body = render.read_page(path) if path.exists() else ({}, "")
    flags = [{"n": f.n, "ocr": f.ocr, "llm": f.llm, "context": f.context(body, 60)} for f in review.parse_flags(body)]
    pages = st.pages(slug)
    seqs = [p["sequence"] for p in pages]
    i = seqs.index(row["sequence"])
    nxt_flag = next((p["sequence"] for p in pages[i + 1:] if p["status"] == "flagged"), None)
    prev_flag = next((p["sequence"] for p in reversed(pages[:i]) if p["status"] == "flagged"), None)
    pid = row["id"]
    here = f"{base}/documents/{slug}/pages/{row['sequence']}"
    return {
        "slug": slug, "sequence": row["sequence"], "id": pid, "status": row["status"],
        "page_label": row["page_label"], "part": row["part"], "width_px": row["width_px"], "advice": row["advice"],
        "reviewed_by": row["reviewed_by"], "llm_error": st.stage_error(pid, "llm"),
        "frontmatter": fm, "body": body, "flags": flags, "unresolved": len(flags),
        "images": {k: f"{here}/image/{k}" for k in image_paths(s, row)},
        "prev": seqs[i - 1] if i > 0 else None, "next": seqs[i + 1] if i + 1 < len(seqs) else None,
        "prev_flagged": prev_flag, "next_flagged": nxt_flag, "exists": path.exists(),
        "fidelity": row["fidelity"] if "fidelity" in row.keys() else None,
        "layout": f"{here}/layout" if pipeline.layout_path(s, pid).exists() else None,
        "figures": [{"n": n, "asset": f.get("asset"), "caption": f.get("caption"), "bbox": f.get("bbox"),
                     "url": f"{here}/figures/{n}.png"}
                    for n, f in enumerate(((fm.get("references") or {}).get("figures") or []), 1)
                    if isinstance(f, dict)],
    }


def doc_payload(s: Settings, st: Store, slug: str, scanning: bool = False, base: str = "/api") -> dict:
    counts = st.counts(slug)
    meta = render.load_document_meta(s.documents / slug)
    pages = st.pages(slug)
    info = library_mod.info(s, st, slug)
    return {"slug": slug, "title": meta.get("title"), "kind": meta.get("kind"), "tags": meta.get("tags") or [],
            "author": info.author, "year": info.year, "isbn": info.isbn, "project": info.project,
            "page_labels": info.page_labels, "last_captured_at": info.last_captured_at, "created_at": info.created_at,
            "exports": info.exports, "done": info.done,
            "pages": len(pages), "counts": counts,
            "needs_reshoot": [p["sequence"] for p in pages if p["status"] in ("illegible", "suspect")],
            "flagged": counts.get("flagged", 0), "unverified": counts.get("unverified", 0) + counts.get("error", 0),
            "duplicates": len(st.duplicates(slug)), "refused": len(st.rejected_captures(slug)),
            "extracted": sorted(p.name for p in (s.documents / slug / "extracted").glob("*")
                                if p.is_file()) if (s.documents / slug / "extracted").exists() else [],
            "scanning": scanning,
            "document_md": f"{base}/documents/{slug}/document.md" if (s.documents / slug / f"{slug}.md").exists() else None}


def page_rows(s: Settings, st: Store, slug: str) -> list[dict]:
    """One row per page for a document's page table: the store's verdict plus both agreement scores from the file."""
    rows = []
    for p in st.pages(slug):
        path = render.page_path(s.documents / slug, p["sequence"])
        agreement = overlap = continuity = None
        if path.exists():
            try:
                fm = render.read_page(path)[0]
                v = fm.get("verification") or {}
                agreement, overlap = v.get("agreement"), v.get("overlap")
                continuity = (fm.get("processing") or {}).get("continuity")
            except Exception as e:                 # a hand-edited page with broken YAML is still a row
                log.warning("page %s unreadable: %s", path, e)
        rows.append({"sequence": p["sequence"], "id": p["id"], "status": p["status"], "page_label": p["page_label"],
                     "part": p["part"], "width_px": p["width_px"], "advice": p["advice"], "continuity": continuity,
                     "agreement": agreement, "overlap": overlap, "llm_error": st.stage_error(p["id"], "llm"),
                     "captured_at": p["captured_at"], "reviewed_by": p["reviewed_by"]})
    return rows
