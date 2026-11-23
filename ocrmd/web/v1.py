"""`/api/v1`: the versioned, typed API the phone app (mobile/) is built against.

The browser UI's `/api` is the SPA's private surface and changes with it. An app installed from a store
cannot be updated in step with the server, so it gets this one instead: every response has a pydantic
model, the OpenAPI document is generated into `docs/reference/openapi-v1.json` (tools/gen_reference.py)
and the app's TypeScript types are generated from that file, so a breaking change here fails the app's
type check before it reaches a phone. Additive changes keep `API_VERSION`; anything that would break an
installed app is `/api/v2`.

Nothing here is new behaviour: reads come from `views`, writes go through `edits`, captures become the
same `ingest` job a drag-and-drop in the browser starts, and gating and processing stay those of the CLI.
What is phone-specific is transport: a capture is one photo per request with a client id (so a retry
from the app's offline queue never adds a page twice), images can be asked for at a size a phone can
show (`?max=`), and the device that sent a request is known (`auth.py`) and recorded as the capture's
`source` and the reviewer of a sign-off.
"""
from __future__ import annotations
import re, time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__, devices as devices_mod, edits, library as library_mod, llm as llm_mod, ocr, pipeline, render
from ..config import Settings
from ..export import FORMATS
from ..store import Store
from . import views
from .auth import device_of, require_local

API_VERSION = 1
PREFIX = "/api/v1"
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic")
MAX_UPLOAD = 64 * 1024 * 1024        # a 48 MP phone JPEG is ~15 MB; anything past this is not a photo
THUMB = 360                          # px on the long side for list thumbnails: two columns on a phone at 3x


# --- models -------------------------------------------------------------------------------------------
class Out(BaseModel):
    """An answer: every field is always sent, so the schema marks even defaulted ones required and the
    app's generated types need no `?` for them. Request bodies stay plain `BaseModel`, where a default
    means the field may be left out."""
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class Ping(Out):
    """Is this an ocrmd server, which API does it speak, and does this request need (or carry) a token."""
    ocrmd: bool = True
    api: int
    version: str
    access: Literal["local", "paired", "open"]
    local: bool
    paired: bool


class Device(Out):
    id: str
    name: str | None
    platform: str | None
    created_at: str
    paired_at: str | None
    last_seen_at: str | None
    revoked_at: str | None
    pending: bool


class ServerInfo(Out):
    version: str
    api: int
    access: Literal["local", "paired", "open"]
    llm: bool = Field(description="a model provider with credentials is configured: pages get their second reading now")
    provider: str
    model: str
    ocr_engines: list[str]
    scanning: str | None = Field(description="the document a webcam scan on the computer is filling, if any")
    sinks: list[str]
    formats: list[str]


class PairIn(BaseModel):
    code: str = Field(min_length=4, max_length=32)
    name: str = Field(min_length=1, max_length=64, description="what the operator will see in `ocrmd devices`")
    platform: str | None = Field(None, max_length=16, description="ios | android | …")


class PairOut(Out):
    token: str = Field(description="send as `Authorization: Bearer <token>`; shown once, store it in the keychain")
    device: Device
    server: ServerInfo


class Pairing(Out):
    id: str
    code: str
    expires_at: float
    server: str = Field(description="the URL the phone should dial")
    link: str = Field(description="ocrmd://pair?server=…&code=… — what the QR code carries")
    qr_svg: str | None = Field(None, description="the link as an SVG QR code (needs the optional `segno`)")


class PairingIn(BaseModel):
    name: str | None = None


class Document(Out):
    slug: str
    title: str | None = None
    author: list[str] = []
    year: int | None = None
    kind: str | None = None
    tags: list[str] = []
    project: str | None = None
    pages: int
    counts: dict[str, int]
    done: int
    flagged: int
    unverified: int
    needs_reshoot: list[int]
    page_labels: str | None = None
    created_at: str | None = None
    last_captured_at: str | None = None
    scanning: bool = False
    cover: str | None = Field(None, description="thumbnail of the first page")


class PageRow(Out):
    sequence: int
    id: str
    status: str
    page_label: str | None = None
    part: str | None = None
    agreement: float | None = None
    overlap: float | None = None
    captured_at: str | None = None
    reviewed_by: str | None = None
    llm_error: str | None = None
    thumb: str | None = None


class DocumentDetail(Document):
    page_list: list[PageRow]


class NewDocument(BaseModel):
    slug: str = Field(pattern=SLUG_RE.pattern)
    title: str | None = None
    author: list[str] = []
    kind: str | None = None
    tags: list[str] = []
    project: str | None = None


class Flag(Out):
    """One open disagreement: `ocr` and `llm` are the two readings, `context` the text around the mark."""
    n: int
    ocr: str
    llm: str
    context: str


class Page(Out):
    slug: str
    sequence: int
    id: str
    status: str
    page_label: str | None = None
    part: str | None = None
    width_px: int | None = None
    advice: str | None = None
    reviewed_by: str | None = None
    llm_error: str | None = None
    captured_at: str | None = None
    agreement: float | None = None
    overlap: float | None = None
    body: str
    flags: list[Flag]
    unresolved: int
    images: dict[str, str] = Field(description="flat | processed | bin | raw → URL; add ?max=N for a smaller JPEG")
    context: dict[str, Any] = {}
    frontmatter: dict[str, Any] = {}
    prev: int | None = None
    next: int | None = None
    prev_flagged: int | None = None
    next_flagged: int | None = None
    exists: bool


class PageEditIn(BaseModel):
    body: str | None = None
    page_label: str | None = None
    context: dict[str, Any] | None = None


class ResolveIn(BaseModel):
    n: int
    chosen: str


class SignOffIn(BaseModel):
    who: str | None = Field(None, description="defaults to the paired device's name")


class VerifyIn(BaseModel):
    refresh: bool = False
    flagged_only: bool = False
    no_llm: bool = False


class PageRef(Out):
    document: str
    sequence: int


class CapturedPage(Out):
    sequence: int
    status: str


class CaptureResult(Out):
    """What became of one uploaded photo."""
    file: str
    client_id: str | None = None
    pages: list[CapturedPage] = []
    duplicate_of: PageRef | None = None
    rejected: list[str] = []
    error: str | None = None


class LogLine(Out):
    at: float
    level: str
    line: str


class Job(Out):
    id: str
    kind: str
    doc: str | None = None
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    created_at: float
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    done: int
    total: int
    results: list[CaptureResult] = []
    log: list[LogLine] = Field([], description="the last lines of the job's log")


class Hit(Out):
    document: str
    sequence: int
    page_id: str
    page_label: str | None = None
    status: str | None = None
    snippet: str
    title: str | None = None


# --- the router ---------------------------------------------------------------------------------------
def _number(x) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _slug(slug: str) -> str:
    if not SLUG_RE.match(slug):
        raise HTTPException(400, "invalid document slug")
    return slug


def router(s: Settings, st: Store, jobs, scan_doc, guard: devices_mod.ClaimGuard, access: str) -> APIRouter:
    """`scan_doc()` names the document a webcam scan is filling (or None); `jobs` is the app's JobManager."""
    r = APIRouter(prefix=PREFIX, tags=["v1"], generate_unique_id_function=lambda route: route.name)
    uploads: OrderedDict[str, str] = OrderedDict()          # client id -> job id: a retried upload is the same job

    def server_info() -> ServerInfo:
        return ServerInfo(version=__version__, api=API_VERSION, access=access,
                          llm=llm_mod.credentials_available(s.provider), provider=s.provider, model=s.model,
                          ocr_engines=ocr.available_engines(), scanning=scan_doc(),
                          sinks=[x.name for x in _sinks()], formats=list(FORMATS))

    def _sinks():
        from .. import sinks as sinks_mod
        return sinks_mod.registry(s)

    def known(slug: str) -> str:
        _slug(slug)
        if slug not in st.documents():
            raise HTTPException(404, "no such document")
        return slug

    def row_of(slug: str, seq: int):
        row = st.page_by_sequence(known(slug), seq)
        if row is None:
            raise HTTPException(404, "no such page")
        return row

    def thumb(slug: str, row) -> str | None:
        if not (s.processed / f"{row['id']}.png").exists():
            return None
        return f"{PREFIX}/documents/{slug}/pages/{row['sequence']}/image/processed?max={THUMB}"

    def document(slug: str) -> dict:
        d = views.doc_payload(s, st, slug, scan_doc() == slug, base=PREFIX)
        first = next(iter(st.pages(slug)), None)
        d["cover"] = thumb(slug, first) if first is not None else None
        return d

    def page(slug: str, seq: int) -> Page:
        row = row_of(slug, seq)
        p = views.page_payload(s, st, slug, row, base=PREFIX)
        fm = p["frontmatter"] if isinstance(p["frontmatter"], dict) else {}
        v = fm.get("verification") if isinstance(fm.get("verification"), dict) else {}
        ctx = fm.get("context")
        # a hand-edited file may hold anything; the typed answer must not 500 on it
        return Page(**{k: p[k] for k in Page.model_fields if k in p and k != "frontmatter"},
                    frontmatter=fm, captured_at=row["captured_at"], agreement=_number(v.get("agreement")),
                    overlap=_number(v.get("overlap")), context=ctx if isinstance(ctx, dict) else {})

    def job_out(job) -> Job:
        d = job.to_dict(with_log=True)
        d["log"] = d["log"][-20:]
        return Job(**d)

    def edited(slug: str, seq: int, write) -> Page:
        try:
            write()
        except edits.EditConflict as e:
            raise HTTPException(409, str(e)) from e
        except edits.FlagNotFound as e:
            raise HTTPException(404, str(e)) from e
        return page(slug, seq)

    # --- who is there -----------------------------------------------------------------------------
    @r.get("/ping", response_model=Ping)
    def ping(request: Request):
        """Open to anyone who can reach the port: says only that this is ocrmd and whether to pair."""
        return Ping(api=API_VERSION, version=__version__, access=access, local=bool(getattr(request.state, "local", False)),
                    paired=device_of(request) is not None)

    @r.post("/pair", response_model=PairOut)
    def pair(req: PairIn):
        """Exchange a pairing code made on the computer (`ocrmd pair`) for this device's token."""
        got = devices_mod.claim(st, req.code, req.name, req.platform)
        if got is None:
            voided = guard.failed(st)
            raise HTTPException(403, "that code is wrong, expired or already used"
                                + ("; too many wrong codes, so every open code was cancelled — make a new one" if voided else ""))
        guard.succeeded()
        dev, token = got
        return PairOut(token=token, device=Device(**dev.to_dict()), server=server_info())

    @r.get("/me", response_model=Device)
    def me(request: Request):
        """The device this request's token belongs to."""
        dev = device_of(request)
        if dev is None:
            raise HTTPException(404, "this request carries no device token")
        return Device(**dev.to_dict())

    @r.get("/server", response_model=ServerInfo)
    def server():
        """What this server can do: API version, access mode, whether pages get the model's reading now, OCR engines."""
        return server_info()

    # --- device management: this machine only ------------------------------------------------------
    @r.get("/devices", response_model=list[Device], dependencies=[Depends(require_local)])
    def devices(include_revoked: bool = False):
        """Paired devices and pending codes. This computer only."""
        return [Device(**d.to_dict()) for d in devices_mod.list_devices(st, include_revoked)]

    @r.post("/devices/pairings", response_model=Pairing, dependencies=[Depends(require_local)])
    def start_pairing(request: Request, req: PairingIn | None = None):
        """A one-time code for a phone, with the link and QR code that carry it. The server URL is the LAN
        address when `serve --lan` bound every interface."""
        p = devices_mod.start_pairing(st, req.name if req else None)
        host, port = request.url.hostname, request.url.port or 80
        url = devices_mod.server_url(None if devices_mod.is_loopback(host) else host, port)
        link = devices_mod.pair_link(url, p.code)
        return Pairing(id=p.id, code=p.display, expires_at=p.expires_at, server=url, link=link,
                       qr_svg=devices_mod.qr_svg(link))

    @r.delete("/devices/{device_id}", response_model=Device, dependencies=[Depends(require_local)])
    def revoke(device_id: str):
        """Revoke a device (or cancel a pending code) by id or name. This computer only."""
        try:
            dev = devices_mod.find(st, device_id)
        except LookupError as e:
            raise HTTPException(404, str(e)) from e
        st.revoke_device(dev.id)
        return Device(**devices_mod.get(st, dev.id).to_dict())

    # --- the library --------------------------------------------------------------------------------
    @r.get("/documents", response_model=list[Document])
    def documents():
        """Every document in the library, with its catalogue entry and progress."""
        pipeline.backfill_page_meta(s, st)
        return [document(d.slug) for d in library_mod.catalog(s, st)]

    @r.post("/documents", response_model=Document, status_code=201)
    def create_document(req: NewDocument):
        """Start a new document before its first capture, so it has a title from the first page on."""
        slug = _slug(req.slug)
        if slug in st.documents():
            raise HTTPException(409, f"'{slug}' already exists")
        st.ensure_document(slug)
        render.ensure_document_meta(s.documents / slug, slug, title=req.title, author=req.author, kind=req.kind,
                                    tags=req.tags, project=req.project)
        library_mod.sync_document(s, st, slug)
        return document(slug)

    @r.get("/documents/{slug}", response_model=DocumentDetail)
    def document_detail(slug: str):
        """One document with a row (and a thumbnail) per page."""
        d = document(known(slug))
        rows = {row["sequence"]: row for row in st.pages(slug)}
        d["page_list"] = [{**x, "thumb": thumb(slug, rows[x["sequence"]])} for x in views.page_rows(s, st, slug)]
        return d

    @r.get("/search", response_model=list[Hit])
    def search(q: str = Query(..., min_length=1), doc: str | None = None, limit: int = Query(50, ge=1, le=500)):
        """Full-text search over every page of every document (or of `doc`)."""
        if doc:
            _slug(doc)
        return [h.to_dict() for h in library_mod.search(s, st, q, doc, limit)]

    # --- pages ----------------------------------------------------------------------------------------
    @r.get("/documents/{slug}/pages/{seq}", response_model=Page)
    def page_get(slug: str, seq: int):
        """A page: its reconciled text, open disagreements, verdict, images and neighbours."""
        return page(slug, seq)

    @r.get("/documents/{slug}/pages/{seq}/image/{kind}", response_class=Response,
           responses={200: {"content": {"image/png": {}, "image/jpeg": {}}}})
    def image(slug: str, seq: int, kind: Literal["flat", "processed", "bin", "raw"],
              max_side: int | None = Query(None, alias="max", ge=64, le=4096,
                                           description="long side in px; answers a JPEG")):
        """A picture of the page: `flat`, `processed`, `bin` or `raw`; `?max=` scales it down to a JPEG."""
        row = row_of(slug, seq)
        p = pipeline.ensure_flat(s, row["id"]) if kind == "flat" else views.image_paths(s, row).get(kind)
        if p is None or not p.exists():
            raise HTTPException(404, "no such image")
        if max_side is None:
            return FileResponse(p, headers={"Cache-Control": "private, max-age=3600"})
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(404, "no such image")
        h, w = img.shape[:2]
        scale = max_side / float(w if w >= h else h)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500, "could not encode the image")
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @r.put("/documents/{slug}/pages/{seq}", response_model=Page)
    def page_edit(slug: str, seq: int, edit: PageEditIn):
        """A human edit: it survives the next `verify`."""
        row = row_of(slug, seq)
        return edited(slug, seq, lambda: edits.edit_page(s, st, slug, row, body=edit.body, page_label=edit.page_label,
                                                         context=edit.context))

    @r.post("/documents/{slug}/pages/{seq}/resolve", response_model=Page)
    def page_resolve(slug: str, seq: int, req: ResolveIn):
        """Settle one disagreement with the OCR reading, the model's, or your own text."""
        row = row_of(slug, seq)
        return edited(slug, seq, lambda: edits.resolve_flag(s, st, slug, row, req.n, req.chosen))

    @r.post("/documents/{slug}/pages/{seq}/signoff", response_model=Page)
    def page_signoff(slug: str, seq: int, request: Request, req: SignOffIn | None = None):
        """Mark the page reviewed; refused (409) while a disagreement is open. The reviewer defaults to the device's name."""
        row = row_of(slug, seq)
        dev = device_of(request)
        who = (req.who if req and req.who else None) or (dev.name if dev and dev.name else "reviewer")
        return edited(slug, seq, lambda: edits.sign_off(s, st, slug, row, who))

    @r.post("/documents/{slug}/pages/{seq}/reopen", response_model=Page)
    def page_reopen(slug: str, seq: int):
        """Undo a sign-off."""
        row = row_of(slug, seq)
        return edited(slug, seq, lambda: edits.reopen(s, st, slug, row))

    # --- captures and jobs -------------------------------------------------------------------------------
    @r.post("/documents/{slug}/captures", response_model=Job, status_code=202)
    async def capture(slug: str, request: Request, file: UploadFile = File(...),
                      client_id: str | None = Form(None, max_length=64),
                      origin: Literal["camera", "scanner"] = Form("camera"),
                      no_llm: bool = Form(False)):
        """One photo from the phone's camera → a page (or two, for a spread), read twice like any other.

        Answers at once with the ingest job; poll `GET /jobs/{id}` for `results`. Send a `client_id` and a
        retry of the same photo returns the same job instead of a second one — and should the server have
        restarted in between, the capture's own sha256 recognises it as a duplicate rather than a new page.
        `origin` says whether the photo is the camera's frame (`camera`, cropped here like any capture) or
        a page the phone's document scanner already cropped (`scanner`); the page's `source.type` keeps
        the difference (`mobile` / `mobile_scan`), since a scanner's crop is not the raw frame.
        """
        _slug(slug)
        if client_id and client_id in uploads and (job := jobs.get(uploads[client_id])) and job.status != "failed":
            return job_out(job)
        name = Path(file.filename or "capture.jpg").name
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(415, f"not an image file: {name}")
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "photo too large")
        if not data:
            raise HTTPException(400, "empty upload")
        inbox = s.root / "inbox" / slug
        inbox.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", client_id or "")[:64]
        dest = inbox / f"{int(time.time() * 1000)}-{safe + '-' if safe else ''}{name}"
        dest.write_bytes(data)
        dev = device_of(request)
        st.ensure_document(slug)
        job = jobs.submit("ingest", slug, {"files": [str(dest)], "no_llm": no_llm,
                                           "source": f"{'mobile_scan' if origin == 'scanner' else 'mobile'}:"
                                                     f"{(dev.name if dev else None) or 'app'}",
                                           "client_ids": {str(dest): client_id} if client_id else {}})
        if client_id:
            uploads[client_id] = job.id
            while len(uploads) > 2000:
                uploads.popitem(last=False)
        return job_out(job)

    @r.post("/documents/{slug}/verify", response_model=Job, status_code=202)
    def verify(slug: str, req: VerifyIn | None = None):
        """Run the stages a document's pages are missing (the model pass for pages captured OCR-only)."""
        req = req or VerifyIn()
        return job_out(jobs.submit("verify", known(slug), req.model_dump()))

    @r.get("/jobs", response_model=list[Job])
    def job_list(doc: str | None = None, active: bool = False):
        """Recent jobs, newest first; `active` keeps the queued and running ones."""
        out = [j for j in jobs.list() if (doc is None or j.doc == doc) and (not active or j.status in ("queued", "running"))]
        return [job_out(j) for j in out[:50]]

    @r.get("/jobs/{job_id}", response_model=Job)
    def job_get(job_id: str):
        """A job's progress and, for a capture, what became of the photo (`results`)."""
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job_out(job)

    return r


def openapi(app) -> dict:
    """The OpenAPI document for `/api/v1` alone: the contract the phone app's types are generated from.

    Cut from the app's full document (paths under the prefix, and the schemas they reach), because the
    browser's `/api` is not part of the contract and must stay free to change."""
    import copy, json
    full = copy.deepcopy(app.openapi())
    paths = {k: v for k, v in full.get("paths", {}).items() if k.startswith(PREFIX)}
    schemas = full.get("components", {}).get("schemas", {})
    keep: set[str] = set()
    todo = [paths]
    while todo:
        refs = re.findall(r'"#/components/schemas/([^"]+)"', json.dumps(todo.pop()))
        for name in refs:
            if name not in keep and name in schemas:
                keep.add(name)
                todo.append(schemas[name])
    return {"openapi": full.get("openapi", "3.1.0"),
            "info": {"title": "ocrmd /api/v1", "version": str(API_VERSION),
                     "description": "The API the ocrmd phone app speaks. Generated by tools/gen_reference.py from "
                                    "ocrmd/web/v1.py — do not edit by hand."},
            "paths": paths,
            "components": {"schemas": {k: schemas[k] for k in sorted(keep)},
                           "securitySchemes": {"device": {"type": "http", "scheme": "bearer",
                                                          "description": "a paired device's token (`POST /api/v1/pair`)"}}},
            "security": [{"device": []}]}
