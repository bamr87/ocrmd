"""Declarative extraction over a scanned document: DocETL's operator model, on ocrmd's own pages.

Scanning a book produces a corpus, and a corpus invites the next question — *what is in it?* Every case
this chapter cites, every dose in the table, every term the glossary defines, one row per page and then
folded together. That is exactly what DocETL (github.com/ucbepic/docetl, MIT, UC Berkeley EPIC lab) is
for: `map` / `filter` / `reduce` / `resolve` / `split` / `unnest`, each an LLM prompt plus an output
schema, chained into a declarative pipeline. The operator model here is theirs, and the spec files are
deliberately close enough to theirs that `--export-docetl` writes a pipeline their runner accepts.

The implementation is not theirs, and that is deliberate. DocETL brings its own LLM stack (litellm,
its own model config, its own disk cache, its own accounting) and about thirty runtime dependencies;
ocrmd already has a provider layer, credentials, token accounting in `llm_calls`, and one governing
idea — *every unit of work writes an artifact and a version, so re-running is free*. Bolting a second
LLM stack next to the first would double all of that. Six operators over `Provider.complete_json` do
not.

Three things are ocrmd's rather than DocETL's, and they are the reason this lives here at all:

- **Every row carries the page it came from.** `_page`, `_page_id` and `_document` are attached at the
  source and merged through every operator, so an extracted claim can be taken back to the photograph
  it was read from. An extraction you cannot check is a rumour.
- **Nothing is inferred.** The system prompt is the same refusal the transcription stage runs on: use
  what the page says, leave the field empty when it does not say it, never fill in what a page of this
  kind would probably contain.
- **Results are cached per operation and versioned by the prompt.** Change a prompt and only that
  operator re-runs; change nothing and a re-run costs nothing. Same bargain as `verify`.

Since 0.9 the operators read the *page model* as well as the page text, and can write back into it:

- **Sources** (`source:` in the spec): `pages` — one row per page, with the reconciliation signals
  the frontmatter records (`agreement`, `cer`, `flagged_spans`, …); `blocks` — one row per layout
  block with its role, box and reading order; `figures` — one row per figure with its crop, so an
  operator with `image: true` can look at the picture; `extracted:<name>` — a previous pipeline's
  rows, for chaining.
- **Sinks** (`output.type`): `frontmatter`, `layout` and `document` write fields back into the page
  files, the page model and `document.yaml`, through the same rules as a human edit — a later
  `verify` keeps them — and every such write is logged (`etl_writes`), so an enrichment is as
  traceable as an extraction.
- **A rectangle, not just a page number.** A row from `blocks` or `figures` carries `_block` and
  `_bbox`; `extract` locates each quoted span on a line of the page and returns it with the line's
  box (`_spans`), which is what `--score` measures as *locality*: the share of quoted spans that are
  actually there. A span that cannot be found is the fabrication signal this repository exists for.
"""
from __future__ import annotations
import csv, hashlib, json, logging, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from rapidfuzz import fuzz

from .config import Settings
from .llm import LLMError, strict_schema
from .store import Store
from . import pipeline as pipeline_mod, render

log = logging.getLogger("ocrmd.etl")

SYSTEM = """You are answering questions about pages of a document that has already been transcribed.

The text you are given is the whole of your evidence. Use only what is in it.

If the text does not say something, that field is empty — an empty string, an empty list, null. Do not
infer it, do not complete it from what a document of this kind usually contains, and do not carry over
something you saw in another page. A wrong answer here is worse than a missing one: it will be read as
though it came off the page, and nobody will check it against the paper again.

Quote spellings, numbers, dates and names exactly as the text has them, including ones that look like
mistakes — the transcription marks its own doubtful readings with ==[...]== and you should keep the
reading inside the brackets. Return only the structured answer."""

LLM_OPS = ("map", "filter", "extract", "reduce", "resolve")          # the ones that can spend tokens
OPERATORS = LLM_OPS + ("unnest", "split", "sample")
SOURCES = ("pages", "blocks", "figures")                              # plus extracted:<name>
SINKS = ("json", "csv", "md", "markdown", "frontmatter", "layout", "document")
# what a sink may touch: the fields a person may edit, and nothing the pipeline owns
FRONTMATTER_KEYS = ("page_label", "context.", "references.cites", "references.tags", "references.related")
LAYOUT_KEYS = ("role", "caption", "kind", "caption_of", "level")
DOCUMENT_KEYS_REFUSED = ("slug",)
_TEMPLATE = re.compile(r"\{\{\s*([a-zA-Z_][\w.]*)\s*\}\}")


# --- output schemas (DocETL's notation) ----------------------------------------------------------
_SCALARS = {"string": "string", "str": "string", "text": "string", "integer": "integer", "int": "integer",
            "number": "number", "float": "number", "boolean": "boolean", "bool": "boolean"}


def parse_type(spec: Any) -> dict:
    """DocETL's type notation -> JSON Schema. `list[{name: string, year: integer}]`, `enum[a, b]`, ..."""
    if isinstance(spec, dict):                      # already a mapping of field -> type
        return object_schema(spec)
    s = str(spec).strip()
    if s.lower() in _SCALARS:
        return {"type": _SCALARS[s.lower()]}
    if s.startswith("list[") and s.endswith("]"):
        return {"type": "array", "items": parse_type(s[5:-1])}
    if s.startswith("enum[") and s.endswith("]"):
        return {"type": "string", "enum": [v.strip() for v in s[5:-1].split(",") if v.strip()]}
    if s.startswith("{") and s.endswith("}"):
        return object_schema(_inline_fields(s[1:-1]))
    raise ValueError(f"unknown output type {spec!r} (try string, integer, number, boolean, "
                     "list[...], enum[a, b], or {field: type})")


def _inline_fields(body: str) -> dict:
    """`name: string, cites: list[string]` -> {name: 'string', cites: 'list[string]'}, respecting nesting."""
    out, depth, buf = {}, 0, ""
    for ch in body + ",":
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            if buf.strip():
                k, _, v = buf.partition(":")
                if not v.strip():
                    raise ValueError(f"field {buf.strip()!r} has no type")
                out[k.strip()] = v.strip()
            buf = ""
        else:
            buf += ch
    return out


def object_schema(fields: dict) -> dict:
    """Every field required and no extras: structured outputs demand it, and a model that may omit a
    field will omit the hard one."""
    props = {k: parse_type(v) for k, v in fields.items()}
    return strict_schema({"type": "object", "properties": props, "required": list(props)})


# --- the spec ------------------------------------------------------------------------------------
@dataclass
class Operation:
    name: str
    type: str
    prompt: str = ""
    output: dict = field(default_factory=dict)         # field -> type notation
    reduce_key: str | list[str] | None = None          # reduce: group by this (None/"_all" = one group)
    resolve_key: str | None = None                     # resolve: the field to canonicalise
    comparison_prompt: str = ""                        # resolve: are these two the same thing?
    blocking_threshold: float = 0.72                   # resolve: fuzzy score below which two are never compared
    unnest_key: str | None = None                      # unnest: the list field to flatten
    split_key: str = "text"                            # split: the field to chunk
    words: int = 400                                   # split: words per chunk
    overlap: int = 40                                  # split: words repeated into the next chunk
    n: int = 10                                        # sample: how many
    max_tokens: int = 4000
    model: str | None = None                           # this operator's model, when not the runner's
    image: bool = False                                # map/extract/filter over `figures`: send the crop too
    extract_key: str = "text"                          # extract: the field the spans must be quoted from

    @property
    def schema(self) -> dict:
        return object_schema(self.output)

    def version(self, model: str) -> str:
        """What invalidates this operation's cache: its prompt, its schema, and the model that answered."""
        blob = json.dumps({"t": self.type, "p": self.prompt, "c": self.comparison_prompt,
                           "o": self.output, "k": self.reduce_key or self.resolve_key, "m": self.model or model,
                           "i": self.image}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def boolean_field(self) -> str | None:
        """The *named* boolean a filter decides on — the first one the schema declares, not the first
        one the answer happens to carry."""
        return next((k for k, v in self.output.items() if str(v).strip().lower() in ("boolean", "bool")), None)


@dataclass
class Pipeline:
    name: str
    operations: list[Operation]
    steps: list[str]                                    # operation names, in order
    source: str = "pages"
    status: list[str] = field(default_factory=list)     # only pages with these statuses
    output: dict = field(default_factory=lambda: {"type": "json"})

    @classmethod
    def load(cls, path: Path) -> "Pipeline":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        ops = {}
        for o in raw.get("operations") or []:
            spec = dict(o)
            out = spec.pop("output", {}) or {}
            spec["output"] = out.get("schema", out) if isinstance(out, dict) else out
            known = {f.name for f in Operation.__dataclass_fields__.values()}
            unknown = set(spec) - known
            if unknown:
                raise ValueError(f"operation {spec.get('name')!r}: unknown key(s) {', '.join(sorted(unknown))}")
            op = Operation(**spec)
            if op.type not in OPERATORS:
                raise ValueError(f"operation {op.name!r}: unknown type {op.type!r} "
                                 f"(have {', '.join(sorted(OPERATORS))})")
            if op.type in ("map", "filter", "reduce") and not op.output:
                raise ValueError(f"operation {op.name!r}: {op.type} needs an output schema")
            if op.type == "extract" and not op.output:
                op.output = {"spans": "list[string]"}
            if op.output:
                _ = op.schema        # a bad type is an error now, not after the first page has been paid for
            if op.type == "filter" and op.boolean_field is None:
                raise ValueError(f"filter {op.name!r} needs a boolean field in its output schema")
            ops[op.name] = op
        pipe = raw.get("pipeline") or {}
        steps = pipe.get("operations") or [s for st in (pipe.get("steps") or []) for s in st.get("operations", [])]
        if not steps:
            steps = list(ops)
        missing = [s for s in steps if s not in ops]
        if missing:
            raise ValueError(f"pipeline names operation(s) that are not defined: {', '.join(missing)}")
        source = pipe.get("input") or raw.get("input") or raw.get("source") or "pages"
        if source not in SOURCES and not source.startswith("extracted:"):
            raise ValueError(f"unknown source {source!r} (use {', '.join(SOURCES)} or extracted:<name>)")
        output = pipe.get("output") or raw.get("output") or {"type": "json"}
        kind = output.get("type", "json")
        if kind not in SINKS:
            raise ValueError(f"unknown output type {kind!r} (use {', '.join(SINKS)})")
        if kind in ("frontmatter", "layout", "document"):
            fields = output.get("fields")
            if not isinstance(fields, dict) or not fields:
                raise ValueError(f"output type {kind!r} needs `fields: {{row field: target}}`")
            for target in fields.values():
                if kind == "frontmatter" and not (target in FRONTMATTER_KEYS or any(target.startswith(k) for k in FRONTMATTER_KEYS if k.endswith("."))):
                    raise ValueError(f"frontmatter sink may not write {target!r} (allowed: {', '.join(FRONTMATTER_KEYS)})")
                if kind == "layout" and target not in LAYOUT_KEYS:
                    raise ValueError(f"layout sink may not write {target!r} (allowed: {', '.join(LAYOUT_KEYS)})")
                if kind == "document" and target in DOCUMENT_KEYS_REFUSED:
                    raise ValueError(f"document sink may not write {target!r}")
        return cls(name=raw.get("name") or Path(path).stem, operations=list(ops.values()), steps=steps,
                   source=source, status=list(raw.get("status") or []), output=output)

    def op(self, name: str) -> Operation:
        return next(o for o in self.operations if o.name == name)

    @property
    def llm_steps(self) -> list[str]:
        return [s for s in self.steps if self.op(s).type in LLM_OPS]

    @property
    def writes_back(self) -> bool:
        return (self.output or {}).get("type") in ("frontmatter", "layout", "document")


# --- the rows ------------------------------------------------------------------------------------
def page_rows(s: Settings, st: Store, slug: str, statuses: list[str] | None = None,
              limit: int | None = None) -> list[dict]:
    """One row per page of the document: its text, its labels, and where it came from.

    Blank pages and pages the pipeline refused as illegible are left out by default — they carry no
    text to reason about, and an operator asked to find something in an empty page is being invited to
    invent it.
    """
    rows = []
    for page in st.pages(slug):
        if statuses and page["status"] not in statuses:
            continue
        if not statuses and page["status"] in ("blank", "illegible"):
            continue
        path = render.page_path(s.documents / slug, page["sequence"])
        if not path.exists():
            continue
        fm, body = render.read_page(path)
        if not body.strip():
            continue
        rows.append(_page_row(s, slug, page, fm, body))
        if limit and len(rows) >= limit:
            break
    return rows


def _page_row(s: Settings, slug: str, page, fm: dict, body: str) -> dict:
    ctx, ver, ocr_ = fm.get("context") or {}, fm.get("verification") or {}, fm.get("ocr") or {}
    return {"_document": slug, "_page": page["sequence"], "_page_id": page["id"],
            "text": body.strip(), "page_label": page["page_label"] or "",
            "status": page["status"], "chapter": ctx.get("chapter") or "",
            "section": ctx.get("section") or "", "title": ctx.get("title") or "",
            "document_title": (render.load_document_meta(s.documents / slug).get("title") or slug),
            # the reconciliation signals: what an operator is honestly told about how well the page was read
            "agreement": ver.get("agreement"), "overlap": ver.get("overlap"), "cer": ver.get("cer"),
            "flagged_spans": ver.get("flagged_spans") or 0, "llm_uncertain": ver.get("llm_uncertain") or [],
            "ocr_confidence": ocr_.get("confidence"), "fidelity": page["fidelity"] if "fidelity" in page.keys() else None,
            "part": page["part"] or ""}


def block_rows(s: Settings, st: Store, slug: str, statuses: list[str] | None = None,
               limit: int | None = None, figures_only: bool = False) -> list[dict]:
    """One row per block of the page model — or per figure — with its geometry beside its text.

    `_block` and `_bbox` are the provenance: an extraction from a block row can be taken back to a
    rectangle on the photograph, not just to a page. A figure row also carries `_asset`, the path of
    its crop, which an operator with `image: true` is shown.
    """
    rows = []
    for prow in page_rows(s, st, slug, statuses, None):
        page = st.page(prow["_page_id"])
        try:
            lay = pipeline_mod.page_layout(s, st, page)
        except Exception as e:                       # artifacts gone: no geometry, no rows for this page
            log.warning("etl: no layout for page %s (%s)", prow["_page"], e)
            continue
        base = {k: v for k, v in prow.items() if k not in ("text",)}
        page_text = prow["text"]
        for b in sorted(lay.blocks, key=lambda b: b.order):
            if figures_only != (b.role == "figure"):
                continue
            row = base | {"_block": b.id, "_bbox": [round(v, 4) for v in b.bbox], "role": b.role, "order": b.order,
                          "column": b.column, "level": b.level, "confidence": b.confidence, "source": b.source,
                          "caption": b.caption or "", "text": (page_text if b.role == "figure" else b.text)}
            if b.role == "figure":
                row["kind"] = b.kind
                row["_asset"] = str(pipeline_mod.figures_dir(s, slug) / b.asset) if b.asset else ""
            else:
                row["_lines"] = [[ln.text, [round(v, 4) for v in ln.bbox]] for ln in b.lines]
            rows.append(row)
            if limit and len(rows) >= limit:
                return rows
    return rows


def extracted_rows(s: Settings, slug: str, name: str) -> list[dict]:
    """A previous pipeline's output as this one's input — the rows it wrote, read back."""
    p = s.documents / slug / "extracted" / f"{Path(name).stem}.json"
    if not p.exists():
        raise FileNotFoundError(f"no extraction {name!r} for {slug} (looked for {p.name})")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{p.name} is not a list of rows")
    return rows


def rows_for(s: Settings, st: Store, slug: str, source: str, statuses: list[str] | None = None,
             limit: int | None = None) -> list[dict]:
    """The rows a pipeline starts from, by its `source`."""
    if source == "pages":
        return page_rows(s, st, slug, statuses, limit)
    if source == "blocks":
        return block_rows(s, st, slug, statuses, limit)
    if source == "figures":
        return block_rows(s, st, slug, statuses, limit, figures_only=True)
    if source.startswith("extracted:"):
        rows = extracted_rows(s, slug, source.split(":", 1)[1])
        return rows[:limit] if limit else rows
    raise ValueError(f"unknown source {source!r}")


def template(text: str, row: dict | None = None, rows: list[dict] | None = None) -> str:
    """`{{ input.field }}` for one row, `{{ inputs }}` for a group. Deliberately not Jinja: a template
    language in a spec file is a way to run code from a spec file, and a missing field must be an error
    that names itself rather than an empty string that silently changes the question."""
    def sub(m: re.Match) -> str:
        ref = m.group(1)
        if ref == "inputs":
            return render_rows(rows or ([row] if row else []))
        if ref.startswith("input."):
            key = ref[6:]
            src = row if row is not None else {}
            if key not in src:
                raise KeyError(f"{{{{ {ref} }}}} is not a field of this row (have: {', '.join(sorted(src))})")
            return _as_text(src[key])
        raise KeyError(f"unknown template reference {{{{ {ref} }}}} (use input.<field> or inputs)")
    return _TEMPLATE.sub(sub, text)


def _as_text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1)


def render_rows(rows: list[dict]) -> str:
    """A group of rows as text for a reduce prompt, each one labelled with the page it came from."""
    out = []
    for r in rows:
        head = f"--- page {r.get('_page', '?')}" + (f" (printed {r['page_label']})" if r.get("page_label") else "") + " ---"
        body = {k: v for k, v in r.items() if not k.startswith("_")}
        out.append(head + "\n" + (body.get("text") if set(body) == {"text"} else _as_text(body)))
    return "\n\n".join(out)


# --- running it ----------------------------------------------------------------------------------
@dataclass
class Stats:
    calls: int = 0
    cached: int = 0
    skipped: int = 0                 # LLM calls not made because there is no provider
    tokens_in: int = 0
    tokens_out: int = 0
    errors: list[str] = field(default_factory=list)

    def describe(self) -> str:
        out = f"{self.calls} call(s), {self.cached} cached"
        if self.tokens_in or self.tokens_out:
            out += f", tokens {self.tokens_in}->{self.tokens_out}"
        if self.skipped:
            out += f", {self.skipped} not asked (no provider)"
        if self.errors:
            out += f", {len(self.errors)} error(s)"
        return out


class Runner:
    """Runs a pipeline's operators over rows, caching each operator's answers on disk."""

    def __init__(self, s: Settings, st: Store, slug: str, provider=None, use_cache: bool = True,
                 dry_run: bool = False, provider_for: Callable[[str], Any] | None = None):
        self.s, self.st, self.slug, self.provider = s, st, slug, provider
        self.use_cache, self.dry_run = use_cache, dry_run
        self.stats = Stats()
        self.model = getattr(provider, "model", "none")
        self.prompts: list[tuple[str, str]] = []       # (operation, prompt) — what --dry-run shows
        self.provider_for = provider_for               # model name -> provider, for an operator's own `model:`
        self._providers: dict[str, Any] = {}

    def _provider(self, op: Operation):
        if not op.model or op.model == self.model:
            return self.provider
        if op.model not in self._providers:
            if self.provider_for is None:
                log.warning("etl %s asks for model %s but no provider factory was given; using %s",
                            op.name, op.model, self.model)
                self._providers[op.model] = self.provider
            else:
                self._providers[op.model] = self.provider_for(op.model)
        return self._providers[op.model]

    # -- cache ------------------------------------------------------------------------------------
    def cache_path(self, op: Operation) -> Path:
        return self.s.root / "etl" / self.slug / f"{op.name}.json"

    def _cache(self, op: Operation) -> dict:
        p, ver = self.cache_path(op), op.version(self.model)
        if not self.use_cache or not p.exists():
            return {}
        try:
            blob = json.loads(p.read_text())
        except ValueError:
            return {}
        if blob.get("version") != ver:
            log.info("etl: %s changed since its cache was written; it will re-run", op.name)
            return {}
        return blob.get("rows") or {}

    def _save(self, op: Operation, rows: dict) -> None:
        if not self.use_cache or self.dry_run or not rows:
            return
        p = self.cache_path(op)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": op.version(self.model), "rows": rows}, indent=1))
        tmp.replace(p)

    # -- the one place that spends anything --------------------------------------------------------
    def ask(self, op: Operation, prompt: str, cache: dict, key: str, page_id: str | None,
            image_path: str | None = None) -> dict | None:
        """One structured answer, from the cache if it is there. None when it could not be had."""
        h = hashlib.sha256((prompt + (f"\n@{image_path}" if image_path else "")).encode()).hexdigest()[:24]
        if h in cache:
            self.stats.cached += 1
            return cache[h]
        self.prompts.append((op.name, prompt))
        if self.dry_run:
            return None
        provider = self._provider(op)
        if provider is None or not hasattr(provider, "complete_json"):
            self.stats.skipped += 1
            return None
        image = None
        if image_path:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                log.warning("etl %s: figure crop %s missing; asking without it", op.name, image_path)
        try:
            res = (provider.complete_json(SYSTEM, prompt, op.schema, op.max_tokens, image_bgr=image) if image is not None
                   else provider.complete_json(SYSTEM, prompt, op.schema, op.max_tokens))
        except LLMError as e:
            self.stats.errors.append(f"{op.name}: {e}")
            log.warning("etl %s failed on one row: %s", op.name, e)
            return None
        self.stats.calls += 1
        self.stats.tokens_in += int(res.usage.get("input_tokens") or 0)
        self.stats.tokens_out += int(res.usage.get("output_tokens") or 0)
        self.st.log_llm(page_id, res.provider, res.model, res.request_id, res.usage, res.latency_ms)
        cache[h] = res.data
        return res.data

    # -- operators --------------------------------------------------------------------------------
    def run(self, pipe: Pipeline, rows: list[dict]) -> list[dict]:
        for name in pipe.steps:
            op = pipe.op(name)
            before = len(rows)
            rows = getattr(self, f"_{op.type}")(op, rows)
            log.info("etl %s (%s): %d row(s) -> %d", op.name, op.type, before, len(rows))
        return rows

    def _map(self, op: Operation, rows: list[dict]) -> list[dict]:
        cache, out = self._cache(op), []
        for r in rows:
            data = self.ask(op, template(op.prompt, r), cache, op.name, r.get("_page_id"),
                            image_path=(r.get("_asset") or None) if op.image else None)
            out.append(r | (data or {}))
        self._save(op, cache)
        return out

    def _extract(self, op: Operation, rows: list[dict]) -> list[dict]:
        """A map that must answer in the page's own words, and is then held to it.

        The prompt says so and the default schema is a list of strings, because the thing that makes
        an extraction checkable is that you can find it. Each returned span is then *looked for* —
        in the block's lines when the row has geometry, in its text otherwise — and comes back as
        `_spans: [{text, bbox, found}]`. A span that is not there is kept, marked `found: false`: it
        is the fabrication signal, and hiding it would be the one thing this module must not do.
        """
        asked = Operation(**(op.__dict__ | {"prompt": op.prompt + "\n\nQuote the document's own words "
                                            "exactly, and return nothing that is not in it verbatim."}))
        out = []
        for r in self._map(asked, rows):
            spans = next((v for v in r.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)), None)
            if spans is None:
                spans = r.get(next(iter(op.output), "spans")) if isinstance(r.get(next(iter(op.output), "spans")), list) else []
            r = dict(r)
            r["_spans"] = [locate_span(x, r, op.extract_key) for x in (spans or [])]
            out.append(r)
        return out

    def _filter(self, op: Operation, rows: list[dict]) -> list[dict]:
        cache, out = self._cache(op), []
        key = op.boolean_field
        for r in rows:
            data = self.ask(op, template(op.prompt, r), cache, op.name, r.get("_page_id"),
                            image_path=(r.get("_asset") or None) if op.image else None)
            if data is None:                     # unanswered: keep the row rather than silently dropping it
                out.append(r)
                continue
            keep = data.get(key) if key else next((v for v in data.values() if isinstance(v, bool)), None)
            if keep is None:
                raise ValueError(f"filter {op.name!r} needs a boolean field in its output schema")
            if keep:
                out.append(r | data)
        self._save(op, cache)
        return out

    def _reduce(self, op: Operation, rows: list[dict]) -> list[dict]:
        keys = [op.reduce_key] if isinstance(op.reduce_key, str) else (op.reduce_key or [])
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            groups.setdefault(tuple(str(r.get(k, "")) for k in keys), []).append(r)
        cache, out = self._cache(op), []
        for gk, members in groups.items():
            data = self.ask(op, template(op.prompt, members[0], members), cache, op.name, None)
            row = {"_document": self.slug, "_pages": sorted({m.get("_page") for m in members if m.get("_page")}),
                   "_rows": len(members)}
            row |= {k: v for k, v in zip(keys, gk, strict=True)}
            out.append(row | (data or {}))
        self._save(op, cache)
        return out

    def _resolve(self, op: Operation, rows: list[dict]) -> list[dict]:
        """Canonicalise one field across rows: the same case, drug or person spelled three ways.

        Fuzzy blocking first (`rapidfuzz`, the library the reconciler already uses), so the model is
        only asked about pairs that could plausibly be the same thing — an all-pairs comparison is
        quadratic in rows and most pairs are obviously unrelated.
        """
        key = op.resolve_key or "name"
        values = sorted({str(r.get(key, "")) for r in rows if str(r.get(key, "")).strip()})
        parent = {v: v for v in values}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]; v = parent[v]
            return v

        cache = self._cache(op)
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                if find(a) == find(b):
                    continue
                score = fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0
                if score < op.blocking_threshold:
                    continue
                same = a.lower() == b.lower()
                if not same and op.comparison_prompt:
                    data = self.ask(op, _pair_prompt(op, a, b), cache, op.name, None)
                    same = bool(next((v for v in (data or {}).values() if isinstance(v, bool)), False))
                elif not same:
                    same = score >= 0.95          # no prompt given: only near-identical strings merge
                if same:
                    parent[find(b)] = find(a)
        self._save(op, cache)
        clusters: dict[str, list[str]] = {}
        for v in values:
            clusters.setdefault(find(v), []).append(v)
        canonical = {v: max(members, key=lambda x: (sum(1 for r in rows if r.get(key) == x), len(x)))
                     for root, members in clusters.items() for v in members}
        out = []
        for r in rows:
            v = str(r.get(key, ""))
            out.append(r | ({key: canonical[v], f"{key}_as_written": v} if v in canonical and canonical[v] != v else {}))
        merged = sum(1 for members in clusters.values() if len(members) > 1)
        log.info("etl %s: %d value(s) in %d cluster(s), %d merged", op.name, len(values), len(clusters), merged)
        return out

    # -- operators that cost nothing ---------------------------------------------------------------
    def _unnest(self, op: Operation, rows: list[dict]) -> list[dict]:
        key = op.unnest_key or next(iter(op.output), None)
        if not key:
            raise ValueError(f"unnest {op.name!r} needs `unnest_key: <list field>`")
        out = []
        if not any(isinstance(r.get(key), list) for r in rows):
            # every row lacking the list is not "nothing found": it is an operator that never ran (no
            # provider, every call failed) or a field name that does not match what produces it
            log.warning("etl %s: no row has a list field %r; nothing to flatten", op.name, key)
        for r in rows:
            v = r.get(key)
            if not isinstance(v, list):
                continue
            for item in v:
                base = {k: x for k, x in r.items() if k != key}
                out.append(base | (item if isinstance(item, dict) else {key: item}))
        return out

    def _split(self, op: Operation, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            words = str(r.get(op.split_key, "")).split()
            step = max(1, op.words - op.overlap)
            chunks = [words[i:i + op.words] for i in range(0, max(1, len(words)), step)] or [[]]
            for i, c in enumerate(chunks):
                out.append(r | {op.split_key: " ".join(c), "_chunk": i, "_chunks": len(chunks)})
        return out

    def _sample(self, op: Operation, rows: list[dict]) -> list[dict]:
        return rows[:op.n]                    # deliberately the first n: a sample you cannot repeat is not a test


def _pair_prompt(op: Operation, a: str, b: str) -> str:
    return template(op.comparison_prompt, {"a": a, "b": b}) if "{{" in op.comparison_prompt else (
        f"{op.comparison_prompt}\n\nA: {a}\nB: {b}")


# --- holding an extraction to the page -----------------------------------------------------------
def _norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]", "", w.lower()) for w in str(s).split() if re.sub(r"[^a-z0-9]", "", w.lower()))


def locate_span(span: str, row: dict, key: str = "text") -> dict:
    """Where a quoted span sits: the line whose words contain it, when the row has lines; otherwise
    whether the row's text contains it at all. `found: false` is a quote that is not on the page."""
    target = _norm_text(span)
    out = {"text": span, "bbox": None, "found": False}
    if not target:
        return out
    for text, bbox in row.get("_lines") or []:
        if target in _norm_text(text):
            out["bbox"], out["found"] = bbox, True
            return out
    if target in _norm_text(row.get(key, "")):
        out["found"], out["bbox"] = True, row.get("_bbox")
    return out


def locality(rows: list[dict]) -> tuple[int, int]:
    """(spans found on the page, spans quoted) across every row that carries `_spans`."""
    spans = [sp for r in rows for sp in (r.get("_spans") or [])]
    return sum(1 for sp in spans if sp.get("found")), len(spans)


@dataclass
class Score:
    """How an extraction did against known answers, and how much of it can be found on the page."""
    key: str
    expected: int = 0
    produced: int = 0
    hits: int = 0
    located: int = 0
    quoted: int = 0

    @property
    def precision(self) -> float | None:
        return self.hits / self.produced if self.produced else None

    @property
    def recall(self) -> float | None:
        return self.hits / self.expected if self.expected else None

    @property
    def locality(self) -> float | None:
        return self.located / self.quoted if self.quoted else None

    def describe(self) -> str:
        f = lambda v: "n/a" if v is None else f"{v:.2f}"
        return (f"{self.key}: precision {f(self.precision)} ({self.hits}/{self.produced}), recall {f(self.recall)} "
                f"({self.hits}/{self.expected}), locality {f(self.locality)} ({self.located}/{self.quoted} spans found)")


def score(rows: list[dict], truth: list[dict], key: str) -> Score:
    """Rows against known answers, matched on (`_page`, `key`) with the same normalisation the
    reconciler uses; plus locality over every quoted span."""
    want = {(r.get("_page"), _norm_text(r.get(key, ""))) for r in truth if _norm_text(r.get(key, ""))}
    got = {(r.get("_page"), _norm_text(r.get(key, ""))) for r in rows if _norm_text(r.get(key, ""))}
    sc = Score(key=key, expected=len(want), produced=len(got), hits=len(want & got))
    sc.located, sc.quoted = locality(rows)
    return sc


# --- writing back --------------------------------------------------------------------------------
def _set_path(d: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {}) if isinstance(d.get(k), dict) or k not in d else d[k]
    d[keys[-1]] = value


def _merge_value(old: Any, new: Any) -> Any:
    """Lists accumulate (deduplicated, order kept); scalars are replaced; empties never overwrite."""
    if new in (None, "", [], {}):
        return old
    if isinstance(old, list) or isinstance(new, list):
        old_l = old if isinstance(old, list) else ([old] if old not in (None, "") else [])
        new_l = new if isinstance(new, list) else [new]
        return old_l + [v for v in new_l if v not in old_l]
    return new


def write_back(s: Settings, st: Store, slug: str, pipe: Pipeline, rows: list[dict], operation: str | None = None) -> dict:
    """Send rows into the page files, the page model or document.yaml, per the spec's `output.fields`.

    Every write goes the way a human edit goes: a page file is rewritten *without* refreshing the
    pipeline's fingerprint (so `verify` treats the values as a person's and keeps them — see
    `review.merge_human_edits`, which now also carries `references.cites`/`tags`); a layout block is
    marked `source: etl` (kept by `layout.merge_human_layout`); document.yaml is the user's file and
    is only ever added to. Each write is logged in `etl_writes`. Returns counts per target.
    """
    out = pipe.output or {}
    kind, fields = out.get("type"), out.get("fields") or {}
    counts = {"pages": 0, "blocks": 0, "document": 0, "fields": 0}
    if kind == "frontmatter":
        by_page: dict[int, dict[str, Any]] = {}
        for r in rows:
            seq = r.get("_page")
            if seq is None:
                continue
            for src, target in fields.items():
                if src in r:
                    by_page.setdefault(seq, {})[target] = _merge_value(by_page.get(seq, {}).get(target), r[src])
        for seq, values in by_page.items():
            page = st.page_by_sequence(slug, seq)
            path = render.page_path(s.documents / slug, seq)
            if page is None or not path.exists():
                continue
            fm, body = render.read_page(path)
            written = []
            for target, v in values.items():
                cur = fm
                for k in target.split(".")[:-1]:
                    cur = cur.get(k) if isinstance(cur.get(k), dict) else {}
                merged = _merge_value(cur.get(target.split(".")[-1]) if cur else None, v)
                if merged in (None, "", []):
                    continue
                _set_path(fm, target, merged)
                written.append(target)
            if written:
                render.write_page(s.documents / slug, seq, fm, body)      # no fingerprint refresh: a human edit
                if "page_label" in written:
                    st.set_status(page["id"], page["status"], fm.get("page_label"))
                st.log_etl_write(slug, page["id"], pipe.name, operation, "frontmatter", written)
                counts["pages"] += 1; counts["fields"] += len(written)
    elif kind == "layout":
        by_page: dict[str, list[dict]] = {}
        for r in rows:
            if r.get("_page_id") and r.get("_block"):
                by_page.setdefault(r["_page_id"], []).append(r)
        for pid, prows in by_page.items():
            page = st.page(pid)
            if page is None:
                continue
            try:
                lay = pipeline_mod.page_layout(s, st, page)
            except Exception as e:
                log.warning("etl: cannot write layout for %s (%s)", pid, e); continue
            written = []
            for r in prows:
                b = lay.block(r["_block"])
                if b is None:
                    continue
                changed = False
                for src, target in fields.items():
                    v = r.get(src)
                    if v in (None, "", []):
                        continue
                    if target == "level":
                        v = int(v)
                    if getattr(b, target) != v:
                        setattr(b, target, v); changed = True
                if changed:
                    b.source = "etl"
                    written.append(f"{b.id}:{','.join(fields.values())}")
            if written:
                pipeline_mod.save_layout(s, st, lay)
                st.log_etl_write(slug, pid, pipe.name, operation, "layout", written)
                counts["blocks"] += len(written); counts["pages"] += 1
    elif kind == "document":
        doc_dir = s.documents / slug
        meta = render.ensure_document_meta(doc_dir, slug)
        written = []
        for r in rows:
            for src, target in fields.items():
                if src in r and r[src] not in (None, "", []):
                    merged = _merge_value(meta.get(target), r[src])
                    if merged != meta.get(target):
                        meta[target] = merged; written.append(target)
        if written:
            render.document_meta_path(doc_dir).write_text(
                "# Document-level metadata. Inherited into every page's `context` on `ocrmd build`;\n"
                "# a page's own value wins. Edit freely — the pipeline never overwrites this file.\n"
                + yaml.dump(meta, Dumper=render._Dumper, sort_keys=False, allow_unicode=True), encoding="utf-8")
            st.log_etl_write(slug, None, pipe.name, operation, "document", sorted(set(written)))
            counts["document"] = 1; counts["fields"] += len(set(written))
    else:
        raise ValueError(f"{kind!r} is not a write-back sink")
    return counts


# --- output --------------------------------------------------------------------------------------
def columns(rows: list[dict], fields: list[str] | None = None) -> list[str]:
    """Provenance first, then every field any row carries, in first-seen order.

    `fields` in the spec's output block narrows it — a map over pages carries the page text into every
    row, which is the right default (the answer and its evidence in one place) and unreadable in a
    spreadsheet. Provenance is kept whatever the list says: a row that cannot be traced is not output.
    """
    seen: list[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    lead = [k for k in ("_document", "_page", "_pages", "_page_id", "_block", "_bbox") if k in seen]
    if fields:
        return lead + [k for k in fields if k in seen and k not in lead]
    return lead + [k for k in seen if k not in lead]


def write(rows: list[dict], path: Path, kind: str = "json", fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields:
        cols = columns(rows, fields)
        rows = [{k: r[k] for k in cols if k in r} for r in rows]
    if kind != "json":                              # line geometry is for the JSON reader, not a spreadsheet
        rows = [{k: v for k, v in r.items() if k != "_lines"} for r in rows]
    if kind == "json":
        path.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
    elif kind == "csv":
        cols = columns(rows, fields)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (v if isinstance(v, (str, int, float)) or v is None
                                else json.dumps(v, ensure_ascii=False)) for k, v in r.items()})
    elif kind in ("md", "markdown"):
        cols = columns(rows, fields)
        cells = [[str(r.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols] for r in rows]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines += ["| " + " | ".join(c) + " |" for c in cells]
        path.write_text("\n".join(lines) + "\n")
    else:
        raise ValueError(f"unknown output type {kind!r} (json, csv or md)")
    return path


def output_path(s: Settings, slug: str, pipe: Pipeline, override: Path | None = None) -> tuple[Path, str, list[str] | None]:
    out = pipe.output or {}
    kind, fields = out.get("type", "json"), out.get("fields")
    if override:
        return override, (override.suffix.lstrip(".") or kind), fields
    return s.documents / slug / "extracted" / (out.get("path") or f"{pipe.name}.{kind}"), kind, fields


# --- handing the same pipeline to DocETL itself ---------------------------------------------------
def litellm_model(model: str) -> str:
    """DocETL speaks litellm, which wants the provider in the model id."""
    if "/" in model:
        return model
    if model.startswith("claude"):
        return f"anthropic/{model}"
    if model.startswith("grok"):
        return f"xai/{model}"
    return model


def export_docetl(pipe: Pipeline, rows: list[dict], out_dir: Path, model: str) -> dict:
    """Write `pipeline.yaml` + `pages.json` that DocETL's own runner can execute.

    The point is its optimizer: DocETL can rewrite a pipeline — decompose a map that is doing too much,
    add a resolve before a reduce, swap in a cheaper model — and that is a body of work worth reaching
    for rather than reimplementing. What crosses over is the spec and the rows; what does not is the
    cache and the page provenance, so treat the result as a study of the pipeline, not as ocrmd output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pages.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
    ops = []
    for name in pipe.steps:
        op = pipe.op(name)
        d: dict = {"name": op.name, "type": op.type}
        if op.model:
            d["model"] = litellm_model(op.model)
        if op.type == "extract":
            # DocETL's extract names the field to quote from and where the quotes land; ours is a prompt
            d["document_keys"] = [op.extract_key]
            d["extraction_key"] = next(iter(op.output), "spans")
            d["prompt"] = op.prompt
            ops.append(d); continue
        if op.prompt:
            d["prompt"] = op.prompt
        if op.output:
            d["output"] = {"schema": op.output}
        if op.type == "reduce":
            d["reduce_key"] = op.reduce_key or "_all"
        if op.type == "resolve":
            d["comparison_prompt"] = op.comparison_prompt
            d["blocking_threshold"] = op.blocking_threshold
            # their runner requires a resolution prompt that picks the canonical form; ours takes the
            # commonest spelling, so the same rule is spelled out for their model
            d["resolution_prompt"] = ("Given these spellings of one entity:\n{% for input in inputs %}- "
                                      "{{ input." + (op.resolve_key or "name") + " }}\n{% endfor %}\n"
                                      "Return the most complete, correctly spelled form as `" + (op.resolve_key or "name") + "`.")
            d["output"] = {"schema": {(op.resolve_key or "name"): "string"}}
        if op.type == "unnest":
            d["unnest_key"] = op.unnest_key
        if op.type == "split":
            d["split_key"] = op.split_key
            d["method"] = "token_count"
            d["method_kwargs"] = {"num_tokens": op.words}
        if op.type == "sample":
            d["samples"] = op.n
        ops.append(d)
    spec = {"default_model": litellm_model(model),
            "datasets": {"pages": {"type": "file", "path": str((out_dir / "pages.json").resolve())}},
            "operations": ops,
            "pipeline": {"steps": [{"name": pipe.name, "input": "pages", "operations": pipe.steps}],
                         "output": {"type": "file", "path": str((out_dir / f"{pipe.name}.json").resolve())}}}
    (out_dir / "pipeline.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, width=100))
    return spec
