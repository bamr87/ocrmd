"""FastAPI backend for the ocrmd web UI. Local, single-user, no authentication: bind to 127.0.0.1.

Everything the CLI can do is reachable here, plus what a terminal cannot show: the live HUD stream,
page images beside their transcription, and click-to-resolve review.
"""
from __future__ import annotations
import logging, re, time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Settings
from ..store import Store
from .. import library as library_mod, pipeline, render, review, sinks as sinks_mod
from ..export import FORMATS, export_document
from ..session import ScanOptions, ScanSession, make_provider
from .health import cameras as list_cameras, doctor_report
from .jobs import JobManager, KINDS, BUNDLED_PIPELINES, pipelines as etl_pipelines

log = logging.getLogger("ocrmd.web")
STATIC = Path(__file__).parent / "static"
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
DOC_META_FIELDS = ("title", "author", "edition", "publisher", "year", "isbn", "citation", "citation_key",
                   "kind", "tags", "project", "physical", "reading_direction")


class ScanStart(BaseModel):
    doc: str
    camera: str | int = 0
    focus: str = "settle"
    no_llm: bool = False
    provider: str | None = None
    model: str | None = None
    resolution: str | None = None
    rotate: int | None = None
    zoom: float | None = None
    duration: float | None = None
    max_pages: int | None = None
    title: str | None = None
    tags: list[str] = []
    project: str | None = None


class Command(BaseModel):
    command: str


class PageEdit(BaseModel):
    body: str | None = None
    page_label: str | None = None
    context: dict[str, Any] | None = None


class LayoutEdit(BaseModel):
    """A page model as the browser edited it: the blocks, and the column count if it changed."""
    blocks: list[dict[str, Any]]
    columns: int | None = None


class PipelineSpec(BaseModel):
    yaml: str


class RegionRequest(BaseModel):
    """A rectangle someone dragged on the page image, in page fractions."""
    x: float
    y: float
    w: float
    h: float
    llm: bool = False
    question: str | None = None


class SendRequest(BaseModel):
    """Send this page — or one reading of part of it — to a destination (see ocrmd/sinks.py).

    `text` empty means the page body as it stands on disk (the reconciled reading, including any
    disagreement the operator has already resolved). A region read supplies its own text and the
    `bbox` it came from, so the receiver records where on the page the item was. `options` are the
    receiver's own, as its descriptor declares them.
    """
    sink: str = ""                       # may be omitted when exactly one destination is configured
    options: dict[str, Any] = {}
    text: str | None = None
    bbox: list[float] | None = None
    dry_run: bool = False


class Resolve(BaseModel):
    n: int
    chosen: str


class SignOff(BaseModel):
    who: str = "reviewer"


class JobRequest(BaseModel):
    kind: str
    doc: str | None = None
    options: dict[str, Any] = {}


class DeletePages(BaseModel):
    sequences: list[int]
    purge_raw: bool = True
    whole_capture: bool = False


def _slug(slug: str) -> str:
    if not SLUG_RE.match(slug):
        raise HTTPException(400, "invalid document slug")
    return slug


def create_app(s: Settings) -> FastAPI:
    s.ensure_dirs()
    st = Store(s.db_path)
    app = FastAPI(title="ocrmd", version="0.3.0", docs_url="/api/docs", redoc_url=None)
    holder: dict[str, ScanSession | None] = {"session": None}

    def scan_running(doc: str | None = None) -> bool:
        ses = holder["session"]
        return bool(ses and ses.running and (doc is None or ses.opts.doc == doc))

    jobs = JobManager(s, st, scan_running)
    sink_cache: dict[tuple[str, str], Any] = {}

    def sink_registry() -> list:
        """The configured destinations, with the descriptor cache kept across requests: settings may
        change under a test, so the list is re-read, but a receiver is not re-probed on every click."""
        out = []
        for sink in sinks_mod.registry(s):
            key = (sink.name, getattr(sink, "url", ""))
            out.append(sink_cache.setdefault(key, sink))
        return out

    def sink_by_name(name: str):
        have = sink_registry()
        for sink in have:
            if sink.name == name:
                return sink
        if len(have) == 1 and not name:
            return have[0]
        raise HTTPException(404, f"no destination named {name!r}" + (f"; configured: {', '.join(x.name for x in have)}"
                                                                  if have else " — none configured"))

    # --- helpers ---------------------------------------------------------------------------------
    def page_row(slug: str, seq: int):
        row = st.page_by_sequence(slug, seq)
        if row is None:
            raise HTTPException(404, "no such page")
        return row

    def page_payload(slug: str, row) -> dict:
        path = render.page_path(s.documents / slug, row["sequence"])
        fm, body = render.read_page(path) if path.exists() else ({}, "")
        flags = [{"n": f.n, "ocr": f.ocr, "llm": f.llm, "context": f.context(body, 60)} for f in review.parse_flags(body)]
        pages = st.pages(slug)
        seqs = [p["sequence"] for p in pages]
        i = seqs.index(row["sequence"])
        nxt_flag = next((p["sequence"] for p in pages[i + 1:] if p["status"] == "flagged"), None)
        prev_flag = next((p["sequence"] for p in reversed(pages[:i]) if p["status"] == "flagged"), None)
        pid = row["id"]
        return {
            "slug": slug, "sequence": row["sequence"], "id": pid, "status": row["status"],
            "page_label": row["page_label"], "part": row["part"], "width_px": row["width_px"], "advice": row["advice"],
            "reviewed_by": row["reviewed_by"], "llm_error": st.stage_error(pid, "llm"),
            "frontmatter": fm, "body": body, "flags": flags, "unresolved": len(flags),
            "images": {k: f"/api/documents/{slug}/pages/{row['sequence']}/image/{k}"
                       for k, p in (("flat", s.processed / f"{pid}.png"),   # rendered on first request
                                    ("processed", s.processed / f"{pid}.png"), ("bin", s.processed / f"{pid}.bin.png"),
                                    ("raw", s.raw / f"{row['raw_sha256']}.jpg")) if p.exists()},
            "prev": seqs[i - 1] if i > 0 else None, "next": seqs[i + 1] if i + 1 < len(seqs) else None,
            "prev_flagged": prev_flag, "next_flagged": nxt_flag, "exists": path.exists(),
            "fidelity": row["fidelity"] if "fidelity" in row.keys() else None,
            "layout": f"/api/documents/{slug}/pages/{row['sequence']}/layout" if pipeline.layout_path(s, pid).exists() else None,
            "figures": [{"n": n, "asset": f.get("asset"), "caption": f.get("caption"), "bbox": f.get("bbox"),
                         "url": f"/api/documents/{slug}/pages/{row['sequence']}/figures/{n}.png"}
                        for n, f in enumerate(((fm.get("references") or {}).get("figures") or []), 1)
                        if isinstance(f, dict)],
        }

    def doc_payload(slug: str) -> dict:
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
                "scanning": scan_running(slug),
                "document_md": f"/api/documents/{slug}/document.md" if (s.documents / slug / f"{slug}.md").exists() else None}

    # --- static ----------------------------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    # --- system ----------------------------------------------------------------------------------
    @app.get("/api/doctor")
    def api_doctor(cameras: bool = True):
        return doctor_report(s, st, probe_cameras=cameras)

    @app.get("/api/cameras")
    def api_cameras():
        return list_cameras(s.root)

    @app.get("/api/settings")
    def api_settings():
        d = s.model_dump()
        d["root"] = str(s.root)
        # a destination's credentials never reach the browser
        d["sinks"] = [{**x, "api_key": "•••" if x.get("api_key") else "", "bearer": "•••" if x.get("bearer") else ""}
                      for x in d.get("sinks") or []]
        return d

    # --- the inventory ---------------------------------------------------------------------------
    @app.get("/api/library")
    def api_library():
        """Every document, with its catalogue entry (document.yaml mirrored into the store) and progress."""
        return {"documents": [d.to_dict() for d in library_mod.catalog(s, st)],
                "indexed": st.indexed_count(), "fts": st.fts, "root": str(s.root), "formats": list(FORMATS),
                "scanning": holder["session"].opts.doc if scan_running() else None}

    @app.get("/api/search")
    def api_search(q: str = "", doc: str | None = None, limit: int = 50):
        """Full-text search over every page's body (FTS5 when SQLite has it, a substring scan otherwise)."""
        if doc:
            _slug(doc)
        return {"query": q, "fts": st.fts,
                "hits": [h.to_dict() for h in library_mod.search(s, st, q, doc, max(1, min(limit, 500)))]}

    @app.post("/api/index")
    def api_index(doc: str | None = None, force: bool = False):
        if doc:
            _slug(doc)
        return {"indexed": library_mod.reindex(s, st, doc, force), "total": st.indexed_count(), "fts": st.fts}

    @app.get("/api/costs")
    def api_costs():
        return doctor_report(s, st, probe_cameras=False)["costs"]

    # --- documents -------------------------------------------------------------------------------
    @app.get("/api/documents")
    def api_documents():
        pipeline.backfill_page_meta(s, st)
        return [doc_payload(slug) for slug in st.documents()]

    @app.get("/api/documents/{slug}")
    def api_document(slug: str):
        _slug(slug)
        if slug not in st.documents():
            raise HTTPException(404, "no such document")
        d = doc_payload(slug)
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
                except Exception:
                    pass
            rows.append({"sequence": p["sequence"], "id": p["id"], "status": p["status"], "page_label": p["page_label"],
                         "part": p["part"], "width_px": p["width_px"], "advice": p["advice"], "continuity": continuity,
                         "agreement": agreement, "overlap": overlap, "llm_error": st.stage_error(p["id"], "llm"),
                         "captured_at": p["captured_at"], "reviewed_by": p["reviewed_by"]})
        d["page_list"] = rows
        d["meta"] = render.load_document_meta(s.documents / slug)
        return d

    @app.get("/api/documents/{slug}/document.md")
    def api_document_md(slug: str):
        _slug(slug)
        p = s.documents / slug / f"{slug}.md"
        if not p.exists():
            raise HTTPException(404, "not built yet")
        return FileResponse(p, media_type="text/markdown", filename=f"{slug}.md")

    @app.get("/api/documents/{slug}/meta")
    def api_meta_get(slug: str):
        _slug(slug)
        return render.ensure_document_meta(s.documents / slug, slug)

    @app.put("/api/documents/{slug}/meta")
    def api_meta_put(slug: str, meta: dict[str, Any]):
        _slug(slug)
        doc_dir = s.documents / slug
        current = render.ensure_document_meta(doc_dir, slug)
        for k in DOC_META_FIELDS:
            if k in meta:
                current[k] = meta[k]
        import yaml
        render.document_meta_path(doc_dir).write_text(
            "# Document-level metadata. Inherited into every page's `context` on `ocrmd build`;\n"
            "# a page's own value wins. Edit freely — the pipeline never overwrites this file.\n"
            + yaml.dump(current, Dumper=render._Dumper, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return current

    @app.post("/api/documents/{slug}/ingest")
    async def api_ingest(slug: str, files: list[UploadFile] = File(...), no_llm: bool = False):
        _slug(slug)
        inbox = s.root / "inbox" / slug
        inbox.mkdir(parents=True, exist_ok=True)
        saved = []
        for up in files:
            name = Path(up.filename or "upload").name
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES + (".pdf",):
                continue
            dest = inbox / f"{int(time.time() * 1000)}-{name}"
            dest.write_bytes(await up.read())
            saved.append(str(dest))
        if not saved:
            raise HTTPException(400, "no image files (png/jpg/tiff/webp) in the upload")
        st.ensure_document(slug)
        job = jobs.submit("ingest", slug, {"files": saved, "no_llm": no_llm})
        return job.to_dict()

    @app.get("/api/documents/{slug}/pipelines")
    def api_pipelines(slug: str):
        """The extraction pipelines available to this document (`documents/<slug>/etl/*.yaml`, then
        `<root>/etl/pipelines/*.yaml`), each with the operators it would run."""
        _slug(slug)
        from .. import etl as etl_mod
        out = []
        for path in etl_pipelines(s, slug):
            try:
                pipe = etl_mod.Pipeline.load(path)
                out.append({"file": path.name, "name": pipe.name, "steps": pipe.steps, "source": pipe.source,
                            "llm_steps": pipe.llm_steps, "output": pipe.output, "writes_back": pipe.writes_back,
                            "bundled": path.parent == BUNDLED_PIPELINES, "yaml": path.read_text(encoding="utf-8")})
            except Exception as e:                     # a broken spec must not hide the working ones
                out.append({"file": path.name, "name": path.stem, "error": str(e)[:200]})
        return out

    @app.put("/api/documents/{slug}/pipelines/{name}")
    def api_pipeline_save(slug: str, name: str, spec: PipelineSpec):
        """Save a pipeline into `documents/<slug>/etl/` — by file name only, validated before it lands,
        so the browser can never write a broken spec or write anywhere else."""
        _slug(slug)
        from .. import etl as etl_mod
        fname = Path(name).name
        if not fname.endswith(".yaml") or fname != re.sub(r"[^A-Za-z0-9._-]", "", fname):
            raise HTTPException(400, "pipeline name must be a plain <name>.yaml")
        d = s.documents / slug / "etl"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{fname}.tmp"
        tmp.write_text(spec.yaml, encoding="utf-8")
        try:
            pipe = etl_mod.Pipeline.load(tmp)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, f"invalid pipeline: {e}") from e
        tmp.replace(d / fname)
        return {"file": fname, "name": pipe.name, "steps": pipe.steps, "source": pipe.source,
                "llm_steps": pipe.llm_steps, "output": pipe.output, "writes_back": pipe.writes_back}

    @app.get("/api/documents/{slug}/etl/writes")
    def api_etl_writes(slug: str):
        """Every field an extraction pipeline wrote back into this document's pages, layouts or metadata."""
        _slug(slug)
        return [dict(r) for r in st.etl_writes(slug)]

    @app.get("/api/documents/{slug}/extracted/{name}")
    def api_extracted(slug: str, name: str):
        """Download what a pipeline wrote. Name only — no paths."""
        _slug(slug)
        p = s.documents / slug / "extracted" / Path(name).name
        if not p.exists():
            raise HTTPException(404, "no such extraction")
        return FileResponse(p, filename=p.name)

    # --- pages -----------------------------------------------------------------------------------
    @app.get("/api/documents/{slug}/pages/{seq}")
    def api_page(slug: str, seq: int):
        _slug(slug)
        return page_payload(slug, page_row(slug, seq))

    @app.get("/api/documents/{slug}/pages/{seq}/image/{kind}")
    def api_page_image(slug: str, seq: int, kind: str):
        _slug(slug)
        row = page_row(slug, seq)
        if kind == "flat":                     # derived from the crop; rendered now if it is not on disk
            p = pipeline.ensure_flat(s, row["id"])
        else:
            p = {"processed": s.processed / f"{row['id']}.png", "bin": s.processed / f"{row['id']}.bin.png",
                 "raw": s.raw / f"{row['raw_sha256']}.jpg"}.get(kind)
        if p is None or not p.exists():
            raise HTTPException(404, "no such image")
        return FileResponse(p, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/documents/{slug}/pages/{seq}/region.png")
    def api_region_image(slug: str, seq: int, x: float, y: float, w: float, h: float, flat: bool = True):
        """The selected rectangle as `read_region` cut it, so the operator can see what was read.

        At its own size, not the enlarged one the engines get: the enlargement is interpolation, it
        tells a person nothing their browser cannot do, and a thin selection at 6x is several MB.
        """
        _slug(slug)
        row = page_row(slug, seq)
        crop = pipeline.region_crop(s, row["id"], [x, y, w, h], flat=flat)
        if crop is None:
            raise HTTPException(404, "no such image")
        import cv2
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            raise HTTPException(500, "could not encode the region")
        return Response(content=buf.tobytes(), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=600"})

    @app.post("/api/documents/{slug}/pages/{seq}/region")
    def api_read_region(slug: str, seq: int, req: RegionRequest):
        """Read one region harder than the page was read. Local OCR always; the model only when asked
        for — this is the one path in ocrmd where a person, not a threshold, decides to spend tokens."""
        _slug(slug)
        row = page_row(slug, seq)
        prov = make_provider(s, not req.llm, s.provider, s.model) if req.llm else None
        if req.llm and prov is None:
            raise HTTPException(409, f"no credentials for {s.provider}; the region was not sent to a model")
        try:
            out = pipeline.read_region(s, st, row, [req.x, req.y, req.w, req.h], prov, req.question)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        d = out.__dict__ | {"size": list(out.size)}
        d["image"] = (f"/api/documents/{slug}/pages/{seq}/region.png"
                      f"?x={req.x:.5f}&y={req.y:.5f}&w={req.w:.5f}&h={req.h:.5f}")
        return d

    # --- destinations: a read page is handed to another system (ocrmd/sinks.py) -----------------
    @app.get("/api/sinks")
    def api_sinks(probe: bool = True):
        """The configured destinations with their descriptors, so the UI can draw each one's options —
        and leave the whole button out when there are none."""
        out = []
        for sink in sink_registry():
            d = sink.describe(probe=probe) if isinstance(sink, sinks_mod.HttpSink) else sink.describe()
            out.append({"name": sink.name, "url": getattr(sink, "url", ""), "descriptor": d})
        return out

    @app.post("/api/documents/{slug}/pages/{seq}/send")
    def api_send(slug: str, seq: int, req: SendRequest):
        """Hand this page — or a reading of one region of it — to a destination. Nothing here is
        written or marked; what the receiver did with it comes back as the delivery's summary."""
        _slug(slug)
        row = page_row(slug, seq)
        sink = sink_by_name(req.sink)
        try:
            out = sinks_mod.send_page(s, slug, row, sink, req.options, text=req.text, bbox=req.bbox, dry_run=req.dry_run)
        except sinks_mod.SinkError as e:
            # 502, not 500: this server is fine — the page did not get through to the other one.
            raise HTTPException(502, str(e)) from e
        return out.to_dict()

    # --- the page model ----------------------------------------------------------------------------
    @app.get("/api/documents/{slug}/pages/{seq}/layout")
    def api_layout(slug: str, seq: int):
        """The page model: blocks with roles, boxes, reading order, figures (`layout/<pid>.json`),
        built now if the stage is behind."""
        _slug(slug)
        row = page_row(slug, seq)
        try:
            return pipeline.page_layout(s, st, row).model_dump()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.put("/api/documents/{slug}/pages/{seq}/layout")
    def api_layout_edit(slug: str, seq: int, lay: LayoutEdit):
        """A human's layout decisions — a figure box dragged, a role corrected, a caption typed. Every
        block the browser sends is marked `source: human` and survives the next rebuild
        (`layout.merge_human_layout`); the page fingerprint is untouched, as for any human edit."""
        _slug(slug)
        row = page_row(slug, seq)
        from ..layout import Layout
        try:
            current = pipeline.page_layout(s, st, row)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        try:
            blocks = [b if isinstance(b, dict) else b.model_dump() for b in lay.blocks]
            new = Layout(page_id=row["id"], width=current.width, height=current.height, columns=lay.columns or current.columns,
                         blocks=[{**b, "source": "human"} for b in blocks])
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"invalid layout: {e}") from e
        pipeline.save_layout(s, st, new)
        return new.model_dump()

    @app.get("/api/documents/{slug}/pages/{seq}/boxes")
    def api_boxes(slug: str, seq: int):
        """Everything the image pane can draw: line boxes, block boxes by role, figure boxes, and the
        open flags located on the page by the line that carries their words."""
        _slug(slug)
        row = page_row(slug, seq)
        try:
            lay = pipeline.page_layout(s, st, row)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        path = render.page_path(s.documents / slug, row["sequence"])
        body = render.read_page(path)[1] if path.exists() else ""
        return {"width": lay.width, "height": lay.height, "columns": lay.columns,
                "blocks": [{"id": b.id, "role": b.role, "bbox": b.bbox, "order": b.order, "column": b.column,
                            "caption": b.caption, "asset": b.asset} for b in lay.blocks],
                "lines": [{"text": ln.text, "bbox": ln.bbox, "confidence": ln.confidence, "block": b.id}
                          for b in lay.blocks for ln in b.lines],
                "flags": pipeline.locate_flags(lay, body)}

    @app.get("/api/documents/{slug}/pages/{seq}/figures/{n}.png")
    def api_figure(slug: str, seq: int, n: int):
        _slug(slug)
        row = page_row(slug, seq)
        p = pipeline.figures_dir(s, slug) / f"{row['id']}-{n}.png"
        if not p.exists():
            raise HTTPException(404, "no such figure")
        return FileResponse(p, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/documents/{slug}/export.json")
    def api_export_json(slug: str):
        """The whole document as one JSON object: metadata, every page's frontmatter, body and layout,
        and the rows every extraction pipeline produced."""
        _slug(slug)
        if slug not in st.documents():
            raise HTTPException(404, "no such document")
        from ..export import jsondoc
        return JSONResponse(jsondoc.document(s, st, slug))

    @app.get("/api/documents/{slug}/export.{fmt}")
    def api_export(slug: str, fmt: str):
        """The document in another shape (md | txt | jsonl; json is above), written under <root>/export/
        and handed back as a download. Every shape keeps the page ids."""
        _slug(slug)
        if fmt not in FORMATS:
            raise HTTPException(400, f"format must be one of {', '.join(FORMATS)}")
        if slug not in st.documents():
            raise HTTPException(404, "no such document")
        path = export_document(s, st, slug, fmt)
        media = {"md": "text/markdown", "txt": "text/plain", "jsonl": "application/x-ndjson", "json": "application/json"}[fmt]
        return FileResponse(path, media_type=media, filename=path.name)

    def _write(slug: str, row, fm: dict, body: str) -> dict:
        render.write_page(s.documents / slug, row["sequence"], fm, body)
        return page_payload(slug, st.page_by_sequence(slug, row["sequence"]))

    @app.put("/api/documents/{slug}/pages/{seq}")
    def api_page_edit(slug: str, seq: int, edit: PageEdit):
        """A human edit. The pipeline's fingerprint is deliberately *not* updated, so a later
        `verify` sees the file as hand-edited and keeps these values."""
        _slug(slug)
        row = page_row(slug, seq)
        path = render.page_path(s.documents / slug, seq)
        if not path.exists():
            raise HTTPException(409, "page has no file yet; run verify")
        fm, body = render.read_page(path)
        if edit.body is not None:
            body = edit.body
        if edit.page_label is not None:
            fm["page_label"] = edit.page_label or None
            st.set_status(row["id"], row["status"], fm["page_label"])
        if edit.context:
            ctx = fm.setdefault("context", {})
            for k, v in edit.context.items():
                if k in ("title", "author", "chapter", "section", "running_head", "citation_key", "tags", "project"):
                    ctx[k] = v
        fm.setdefault("verification", {})["flagged_spans"] = review.unresolved(body)
        return _write(slug, row, fm, body)

    @app.post("/api/documents/{slug}/pages/{seq}/resolve")
    def api_resolve(slug: str, seq: int, r: Resolve):
        _slug(slug)
        row = page_row(slug, seq)
        path = render.page_path(s.documents / slug, seq)
        if not path.exists():
            raise HTTPException(409, "page has no file yet")
        fm, body = render.read_page(path)
        if not any(f.n == r.n for f in review.parse_flags(body)):
            raise HTTPException(404, f"flag #{r.n} not found")
        body = review.resolve(body, r.n, r.chosen)
        fm.setdefault("verification", {})["flagged_spans"] = review.unresolved(body)
        return _write(slug, row, fm, body)

    @app.post("/api/documents/{slug}/pages/{seq}/signoff")
    def api_signoff(slug: str, seq: int, so: SignOff):
        _slug(slug)
        row = page_row(slug, seq)
        path = render.page_path(s.documents / slug, seq)
        if not path.exists():
            raise HTTPException(409, "page has no file yet")
        fm, body = render.read_page(path)
        left = review.unresolved(body)
        if left:
            raise HTTPException(409, f"{left} disagreement(s) still unresolved")
        v = fm.setdefault("verification", {})
        v["reviewed_by"] = {"who": so.who, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        v["status"] = "reviewed"
        v["flagged_spans"] = 0
        st.set_reviewed(row["id"], so.who, "reviewed")
        render.write_page(s.documents / slug, seq, fm, body)
        st.set_fingerprint(row["id"], review.fingerprint(fm, body))
        return page_payload(slug, st.page_by_sequence(slug, seq))

    @app.post("/api/documents/{slug}/pages/{seq}/reopen")
    def api_reopen(slug: str, seq: int):
        """Undo a sign-off: the page goes back to flagged/verified by its numbers."""
        _slug(slug)
        row = page_row(slug, seq)
        path = render.page_path(s.documents / slug, seq)
        fm, body = render.read_page(path)
        v = fm.setdefault("verification", {})
        v["reviewed_by"] = None
        v["status"] = "flagged" if review.unresolved(body) else "verified"
        st.set_reviewed(row["id"], None, v["status"])
        return _write(slug, row, fm, body)

    # --- deletion --------------------------------------------------------------------------------
    def _deleted(out: pipeline.Deleted, slug: str, rebuild: bool = True) -> dict:
        if rebuild and slug in st.documents():
            render.build_document(s.documents / slug, slug)
        return {"deleted": out.pages, "files": out.files, "raws": out.raws, "summary": out.summary()}

    @app.delete("/api/documents/{slug}/pages/{seq}")
    def api_page_delete(slug: str, seq: int, purge_raw: bool = True, whole_capture: bool = False):
        _slug(slug)
        row = page_row(slug, seq)
        if scan_running(slug):
            raise HTTPException(409, f"'{slug}' is being scanned; stop the scan first")
        out = pipeline.delete_pages(s, st, slug, [row["id"]], purge_raw=purge_raw, whole_capture=whole_capture)
        return _deleted(out, slug)

    @app.post("/api/documents/{slug}/pages/delete")
    def api_pages_delete(slug: str, req: DeletePages):
        _slug(slug)
        if scan_running(slug):
            raise HTTPException(409, f"'{slug}' is being scanned; stop the scan first")
        ids = [r["id"] for seq in req.sequences if (r := st.page_by_sequence(slug, seq)) is not None]
        if not ids:
            raise HTTPException(404, "no such pages")
        out = pipeline.delete_pages(s, st, slug, ids, purge_raw=req.purge_raw, whole_capture=req.whole_capture)
        return _deleted(out, slug)

    @app.delete("/api/documents/{slug}")
    def api_document_delete(slug: str, confirm: str = "", purge_raw: bool = True):
        """Irreversible. `confirm` must repeat the slug."""
        _slug(slug)
        if slug not in st.documents():
            raise HTTPException(404, "no such document")
        if confirm != slug:
            raise HTTPException(400, "confirm must equal the document slug")
        if scan_running(slug):
            raise HTTPException(409, f"'{slug}' is being scanned; stop the scan first")
        out = pipeline.delete_document(s, st, slug, purge_raw=purge_raw)
        return _deleted(out, slug, rebuild=False)

    # --- jobs ------------------------------------------------------------------------------------
    @app.get("/api/jobs")
    def api_jobs():
        return [j.to_dict() for j in jobs.list()]

    @app.post("/api/jobs")
    def api_job_create(req: JobRequest):
        if req.kind not in KINDS:
            raise HTTPException(400, f"kind must be one of {KINDS}")
        if req.doc:
            _slug(req.doc)
        try:
            return jobs.submit(req.kind, req.doc, req.options).to_dict()
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str, since: int = 0):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        d = job.to_dict(with_log=True)
        d["log"] = d["log"][since:]
        d["log_offset"] = since
        return d

    @app.post("/api/jobs/{job_id}/cancel")
    def api_job_cancel(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(404, "no such job or not cancellable")
        return {"ok": True}

    # --- scan ------------------------------------------------------------------------------------
    @app.get("/api/scan")
    def api_scan():
        ses = holder["session"]
        if ses is None:
            return {"running": False, "doc": None}
        return ses.snapshot()

    @app.post("/api/scan/start")
    def api_scan_start(req: ScanStart):
        _slug(req.doc)
        if scan_running():
            raise HTTPException(409, "a scan is already running")
        lines: list[dict] = []
        prov = make_provider(s, req.no_llm, req.provider or s.provider, req.model or s.model,
                             lambda line, level: lines.append({"line": line, "level": level}))
        res = None
        if req.resolution:
            try:
                w, h = req.resolution.lower().split("x")
                res = (int(w), int(h))
            except ValueError:
                raise HTTPException(400, "resolution must be WxH") from None
        if req.focus not in ("continuous", "locked", "settle", "off"):
            raise HTTPException(400, "focus must be continuous | locked | settle | off")
        opts = ScanOptions(doc=req.doc, camera=req.camera, focus=req.focus, resolution=res, rotate=req.rotate,
                           zoom=req.zoom, duration=req.duration, max_pages=req.max_pages, title=req.title,
                           tags=req.tags, project=req.project)
        ses = ScanSession(s, st, opts, prov, preview=False)
        for line in lines:
            ses.emit(line["line"], line["level"])
        holder["session"] = ses
        ses.start()
        time.sleep(0.3)                                  # let camera resolution fail fast if it will
        return ses.snapshot()

    @app.post("/api/scan/stop")
    def api_scan_stop():
        ses = holder["session"]
        if ses is None or not ses.running:
            return {"running": False}
        ses.command("quit")
        return {"running": True, "stopping": True}

    @app.post("/api/scan/command")
    def api_scan_command(cmd: Command):
        ses = holder["session"]
        if ses is None or not ses.running:
            raise HTTPException(409, "no scan running")
        ses.command(cmd.command)
        return {"ok": True}

    @app.get("/api/scan/frame.jpg")
    def api_scan_frame():
        ses = holder["session"]
        if ses is None:
            raise HTTPException(404, "no scan")
        jpeg, _ = ses.latest_jpeg(timeout=0.5)
        if jpeg is None:
            raise HTTPException(404, "no frame yet")
        return StreamingResponse(iter([jpeg]), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/scan/stream")
    def api_scan_stream():
        ses = holder["session"]
        if ses is None:
            raise HTTPException(404, "no scan")

        def gen():
            seq = -1
            idle = 0
            while True:
                jpeg, new_seq = ses.latest_jpeg(after_seq=seq, timeout=1.0)
                if jpeg is None or new_seq == seq:
                    idle += 1
                    if not ses.running and idle > 2:
                        return
                    continue
                idle, seq = 0, new_seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.05)                          # ~15-20 fps is plenty for a preview

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.exception_handler(Exception)
    async def unhandled(_request, exc: Exception):
        log.exception("unhandled error in web api")
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return app
