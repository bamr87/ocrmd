"""Destinations — a page ocrmd has read is handed to another system.

Everything else in this repository ends at a file on disk. This module is the one outbound path,
and it is generic on purpose: ocrmd knows nothing about what a receiver does with a page. A receiver
is any HTTP endpoint that speaks the small contract below — a study platform turning the page into a
practice question, a knowledge base filing it as a source, a notebook — and every one of them gets
the same button, the same command and the same envelope. The first receiver built against it is
law-ai's LSAT intake, which is why the contract has the shape it has: an operator picks *which* item
on a page, with a couple of typed hints the receiver declares.

**The contract.** One URL per destination, used two ways:

    GET  url  → the *descriptor*: what the receiver is and which options it takes
                {"ocrmd_sink": 1, "name": …, "description": …, "accepts": ["page", "region"],
                 "options": [{"key": "hint", "label": …, "type": "string|enum|integer|number|boolean",
                              "choices": […], "default": …, "help": …}, …]}
    POST url  → the *envelope* (`envelope()` below): the page's reconciled text; its provenance —
                document, page, page id, both agreement scores, the flag count, a URL back to the
                image; the document's catalogue entry; and the operator's options. The receiver
                answers 2xx JSON with, ideally, {"summary": one line, "items": [{id, title, detail,
                status}], "warnings": [str]}; anything else is shown raw.

A receiver with no descriptor still works (the send simply has no options); a receiver that is down
is reported as down. Descriptors are cached briefly so a UI can list destinations without a round
trip per click.

**What travels and what does not.** The text and the provenance. Not the image: a page image and a
vision model would let the receiver do its own single-pass read — which is exactly the thing ocrmd
exists to distrust. The page id (the sha256 of the processed image) goes with it, so it outlives
the URL and the process, and a reviewer over there can always come back to the exact photograph.

**A send is not a stage.** Nothing is written, no stage is marked, no version moves — the same
contract as `pipeline.read_region`. Sending twice is safe on this side; a well-behaved receiver
deduplicates and says so, which is how a person actually checks "did that go through?". Failure is
reported with the receiver's own reason, never retried into a silent success.

Two ways to register a destination: `[[sinks]]` in ocrmd.toml (or `OCRMD_SINKS` as JSON) for an
HTTP receiver, and the `ocrmd.sinks` entry-point group for a Python one — anything with a `name`, a
`describe()` and a `send(envelope, options)`, constructed with the `Settings`.
"""
from __future__ import annotations
import json, logging, time, urllib.error, urllib.request
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol

from . import __version__
from .config import Settings, SinkConfig
from . import render, review

log = logging.getLogger("ocrmd.sinks")
ENVELOPE_VERSION = 1
DESCRIPTOR_TTL = 60.0        # seconds a probed descriptor is believed before it is fetched again
PROBE_TIMEOUT = 4.0          # a descriptor is a small GET; a receiver that cannot answer it is "unreachable"
OPTION_TYPES = ("string", "enum", "integer", "number", "boolean")


class SinkError(RuntimeError):
    """The receiver could not be reached, refused the page, or the options did not fit its descriptor.

    One class for all three, deliberately: to the person who pressed the button the difference is
    the message, not the remedy, and the message is carried."""


@dataclass
class Delivery:
    """What a receiver did with a page. `raw` is its whole answer, for whatever `summary` leaves out."""
    sink: str
    url: str
    status: int = 0
    summary: str = ""
    items: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return self.dry_run or 200 <= self.status < 300

    def to_dict(self) -> dict:
        return {"sink": self.sink, "url": self.url, "status": self.status, "summary": self.summary, "items": self.items,
                "warnings": self.warnings, "raw": self.raw, "dry_run": self.dry_run, "ok": self.ok}

    def describe(self) -> str:
        """A few lines for a terminal: the summary, then one line per item, then the warnings."""
        lines = [self.summary or ("nothing to report" if self.ok else f"HTTP {self.status}")]
        for it in self.items:
            head = " · ".join(str(x) for x in (it.get("title"), it.get("detail")) if x)
            lines.append(f"  {it.get('status') or '·':<8} {head}" + (f"  [{it['id']}]" if it.get("id") else ""))
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


class SinkProtocol(Protocol):
    """What the CLI, the web UI and `Library.send` need from a destination."""
    name: str
    def describe(self) -> dict: ...
    def send(self, envelope: dict, options: dict | None = None) -> Delivery: ...


def parse_delivery(name: str, url: str, status: int, data: dict) -> Delivery:
    """A receiver's answer in the contract's terms. A receiver that answers in its own vocabulary gets
    a summary that says only what can be said without inventing an outcome."""
    d = Delivery(sink=name, url=url, status=status, raw=data)
    items = data.get("items")
    d.items = [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
    warnings = data.get("warnings")
    d.warnings = [str(w) for w in warnings] if isinstance(warnings, list) else []
    d.summary = str(data.get("summary") or "")
    if not d.summary:
        d.summary = f"{len(d.items)} item(s) taken in" if d.items else f"delivered (HTTP {status})"
    return d


def _http_json(req: urllib.request.Request, timeout: float) -> tuple[int, dict]:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode() or "{}")
        return resp.status, data if isinstance(data, dict) else {"result": data}


class HttpSink:
    """A receiver reached over HTTP: the descriptor by GET, the envelope by POST, one URL."""

    def __init__(self, cfg: SinkConfig):
        self.cfg = cfg
        self.name, self.url, self.timeout = cfg.name, cfg.url, cfg.timeout
        self._descriptor: dict | None = None
        self._probed_at = 0.0

    def _headers(self, body: bool) -> dict:
        h = {"Accept": "application/json", "User-Agent": f"ocrmd/{__version__}"}
        if body:
            h["Content-Type"] = "application/json"
        if self.cfg.api_key:
            h["X-API-Key"] = self.cfg.api_key
        if self.cfg.bearer:
            h["Authorization"] = f"Bearer {self.cfg.bearer}"
        return h

    def describe(self, probe: bool = True) -> dict:
        """The receiver's descriptor, probed at most every DESCRIPTOR_TTL seconds. A receiver that has
        none, or is down, still yields a usable entry — `reachable` and `error` say which."""
        if self._descriptor is not None and (not probe or time.monotonic() - self._probed_at < DESCRIPTOR_TTL):
            return {**self._descriptor, "defaults": dict(self.cfg.options)}
        d: dict = {"name": self.name, "description": "", "accepts": ["page", "region"], "options": [],
                   "url": self.url, "reachable": False, "error": None}
        if probe:
            try:
                _status, data = _http_json(urllib.request.Request(self.url, headers=self._headers(body=False)), PROBE_TIMEOUT)
                d["reachable"] = True
                if data.get("ocrmd_sink"):
                    d.update({k: data[k] for k in ("name", "description", "accepts", "options") if k in data})
                    d["options"] = [o for o in (d["options"] or []) if isinstance(o, dict) and o.get("key")]
                else:
                    d["error"] = "no descriptor: the receiver answered, but not with an ocrmd_sink document"
            except urllib.error.HTTPError as e:
                d["reachable"] = e.code in (404, 405)       # an intake with no descriptor: reachable, just mute
                d["error"] = f"HTTP {e.code} on the descriptor"
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                d["error"] = f"unreachable: {getattr(e, 'reason', e)}"
            self._descriptor, self._probed_at = d, time.monotonic()
        return {**d, "defaults": dict(self.cfg.options)}

    def send(self, envelope: dict, options: dict | None = None) -> Delivery:
        payload = dict(envelope)
        payload["options"] = {**self.cfg.options, **(options or {})}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST", headers=self._headers(body=True))
        prov = payload.get("provenance") or {}
        log.info("sink %s: POST %s document=%s page=%s %d chars", self.name, self.url, prov.get("document"),
                 prov.get("page"), len(payload.get("text") or ""))
        try:
            status, data = _http_json(req, self.timeout)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode()).get("detail") or ""
            except Exception:
                pass
            raise SinkError(f"{self.name} refused the page ({e.code}): {str(detail)[:300] or e.reason}") from e
        except urllib.error.URLError as e:
            raise SinkError(f"cannot reach {self.name} at {self.url}: {e.reason}") from e
        except (TimeoutError, OSError) as e:
            raise SinkError(f"{self.name} did not answer within {self.timeout:g}s: {e}") from e
        except json.JSONDecodeError as e:
            raise SinkError(f"{self.name} returned something that is not JSON: {e}") from e
        out = parse_delivery(self.name, self.url, status, data)
        log.info("sink %s: %s", self.name, out.summary)
        return out


# --- registry ------------------------------------------------------------------------------------
def registry(s: Settings) -> list:
    """Every configured destination: the `[[sinks]]` HTTP receivers, then any `ocrmd.sinks` entry point."""
    out: list = [HttpSink(c) for c in s.sinks]
    try:
        eps = list(entry_points(group="ocrmd.sinks"))
    except Exception:                          # an odd metadata backend must not take the button away
        eps = []
    for ep in eps:
        try:
            obj = ep.load()
            sink = obj(s) if callable(obj) else obj
            if getattr(sink, "name", None) and callable(getattr(sink, "send", None)):
                out.append(sink)
        except Exception as e:
            log.warning("sink entry point %s could not be loaded: %s", ep.name, e)
    return out


def configured(s: Settings) -> bool:
    return bool(registry(s))


def get(s: Settings, name: str):
    have = registry(s)
    for sink in have:
        if sink.name == name:
            return sink
    if len(have) == 1 and not name:
        return have[0]
    raise SinkError(f"no destination named {name!r}" + (f"; configured: {', '.join(x.name for x in have)}" if have
                    else " — add a [[sinks]] table to ocrmd.toml, or set OCRMD_SINKS"))


# --- the envelope ----------------------------------------------------------------------------------
def page_text(s: Settings, slug: str, row) -> tuple[str, dict]:
    """The page's Markdown body and frontmatter as they stand on disk.

    The body, not the OCR artifact and not the LLM's answer: the body is the reconciled reading
    *including* any human resolution of the disagreements, which is the best text this repo has
    about the page. A page nobody has written yet (an interrupted scan, a failed stage) has no body,
    and that is an error here rather than an empty send."""
    path = render.page_path(s.documents / slug, row["sequence"])
    if not path.exists():
        raise SinkError(f"page {row['sequence']} has no page file yet — run `ocrmd verify --doc {slug}` first")
    fm, body = render.read_page(path)
    if not body.strip():
        raise SinkError(f"page {row['sequence']} is empty — nothing to send")
    return body, fm


def provenance(s: Settings, slug: str, row, fm: dict | None = None, bbox: list[float] | None = None) -> dict:
    """Everything a receiver needs to find this page again, and to judge the reading.

    The two agreement scores go with it because they are the cross-check's verdict: a reviewer
    looking at whatever the receiver made of the page should be able to see that the OCR and the
    model read it identically, or that they did not."""
    fm = fm or {}
    ver, ocr, llm = fm.get("verification") or {}, fm.get("ocr") or {}, fm.get("llm") or {}
    seq = row["sequence"]
    prov = {
        "tool": "ocrmd", "tool_version": __version__, "envelope": ENVELOPE_VERSION,
        "document": slug, "page": seq, "page_id": row["id"], "page_label": row["page_label"] or "",
        "captured_at": str(fm.get("captured_at") or row["captured_at"] or ""),
        "ocr_engine": str(ocr.get("engine") or ""), "llm_model": str(llm.get("model") or ""),
        "agreement": ver.get("agreement"), "overlap": ver.get("overlap"), "flagged_spans": ver.get("flagged_spans"),
        "status": row["status"],
        "image_url": f"{s.web_url}/api/documents/{slug}/pages/{seq}/image/flat",
        "page_url": f"{s.web_url}/#/doc/{slug}/page/{seq}",
    }
    if bbox:
        prov["bbox"] = [round(float(v), 5) for v in bbox]
        prov["image_url"] = (f"{s.web_url}/api/documents/{slug}/pages/{seq}/region.png"
                             f"?x={bbox[0]:.5f}&y={bbox[1]:.5f}&w={bbox[2]:.5f}&h={bbox[3]:.5f}")
    return prov


def envelope(s: Settings, slug: str, row, *, text: str | None = None, bbox: list[float] | None = None) -> dict:
    """The transfer format: what a destination receives. `text` overrides the page body — that is how
    a region read reaches this path: the operator boxed one item, `read_region` answered, and *that*
    reading travels with the box it came from. `bbox` only records where; nothing is cropped here."""
    fm: dict = {}
    if text is None:
        text, fm = page_text(s, slug, row)
    else:
        path = render.page_path(s.documents / slug, row["sequence"])
        if path.exists():
            fm, _ = render.read_page(path)
    prov = provenance(s, slug, row, fm, bbox)
    prov["unresolved"] = review.unresolved(text)
    meta = render.load_document_meta(s.documents / slug)
    doc = {"slug": slug, **{k: meta.get(k) for k in ("title", "author", "kind", "year", "isbn", "citation_key", "tags", "project")}}
    return {"ocrmd": ENVELOPE_VERSION, "kind": "region" if bbox else "page", "text": text, "provenance": prov, "document": doc}


# --- options ----------------------------------------------------------------------------------------
def coerce_options(descriptor: dict, options: dict) -> dict:
    """Cast and check the operator's options against what the receiver declared. Unknown keys pass
    through untouched (a receiver may take more than it advertises); a blank means "let the receiver
    decide" and is dropped; a value outside an enum is refused *here*, because the receiver's 422
    would name a field the person never saw."""
    spec = {o["key"]: o for o in (descriptor.get("options") or []) if isinstance(o, dict) and o.get("key")}
    out: dict[str, Any] = {}
    for k, v in (options or {}).items():
        if v in (None, ""):
            continue
        o = spec.get(k)
        if o is None:
            out[k] = v
            continue
        t = o.get("type") or "string"
        try:
            if t == "integer":
                v = int(v)
            elif t == "number":
                v = float(v)
            elif t == "boolean":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
            elif t == "enum":
                choices = [str(c) for c in (o.get("choices") or [])]
                if choices and str(v) not in choices:
                    raise SinkError(f"{k} must be one of {', '.join(choices)} (got {v!r})")
                v = str(v)
            else:
                v = str(v)
        except (TypeError, ValueError) as e:
            raise SinkError(f"{k}: {e}") from e
        out[k] = v
    return out


def parse_option_args(pairs: list[str]) -> dict:
    """`--option key=value` from a command line. Values stay strings; `coerce_options` types them."""
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep or not k.strip():
            raise SinkError(f"--option wants key=value, not {p!r}")
        out[k.strip()] = v
    return out


def send_page(s: Settings, slug: str, row, sink, options: dict | None = None, *, text: str | None = None,
              bbox: list[float] | None = None, dry_run: bool = False) -> Delivery:
    """Send a page, or a caller-supplied reading of part of one, to a destination.

    Checks what can be checked before anything leaves: that there is text, that the receiver takes
    this kind of thing, and that the options fit its descriptor. `dry_run` returns the envelope that
    would have gone, as `raw`."""
    env = envelope(s, slug, row, text=text, bbox=bbox)
    if not env["text"].strip():
        raise SinkError("nothing to send — the text is empty")
    desc = sink.describe() if callable(getattr(sink, "describe", None)) else {}
    accepts = desc.get("accepts") or ["page", "region"]
    if env["kind"] not in accepts:
        raise SinkError(f"{sink.name} does not take a {env['kind']} (it accepts {', '.join(accepts)})")
    opts = coerce_options(desc, {**(desc.get("defaults") or {}), **(options or {})})
    if dry_run:
        return Delivery(sink=sink.name, url=getattr(sink, "url", ""), summary=f"would send {len(env['text'])} chars to {sink.name}",
                        raw={**env, "options": opts}, dry_run=True)
    return sink.send(env, opts)


def unresolved_flags(body: str) -> int:
    """How many disagreements the text still carries. Not a gate — a page with open flags is still
    worth sending, and the receiver is told the count — but both drivers say the same sentence before
    sending: *this text has n words the two readings disagreed about.*"""
    return len(review.parse_flags(body))
