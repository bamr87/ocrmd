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
"""
from __future__ import annotations
import csv, hashlib, json, logging, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz

from .config import Settings
from .llm import LLMError, strict_schema
from .store import Store
from . import render

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

    @property
    def schema(self) -> dict:
        return object_schema(self.output)

    def version(self, model: str) -> str:
        """What invalidates this operation's cache: its prompt, its schema, and the model that answered."""
        blob = json.dumps({"t": self.type, "p": self.prompt, "c": self.comparison_prompt,
                           "o": self.output, "k": self.reduce_key or self.resolve_key, "m": model},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


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
            if op.output:
                _ = op.schema        # a bad type is an error now, not after the first page has been paid for
            ops[op.name] = op
        pipe = raw.get("pipeline") or {}
        steps = pipe.get("operations") or [s for st in (pipe.get("steps") or []) for s in st.get("operations", [])]
        if not steps:
            steps = list(ops)
        missing = [s for s in steps if s not in ops]
        if missing:
            raise ValueError(f"pipeline names operation(s) that are not defined: {', '.join(missing)}")
        return cls(name=raw.get("name") or Path(path).stem, operations=list(ops.values()), steps=steps,
                   source=(pipe.get("input") or raw.get("input") or "pages"),
                   status=list(raw.get("status") or []),
                   output=pipe.get("output") or raw.get("output") or {"type": "json"})

    def op(self, name: str) -> Operation:
        return next(o for o in self.operations if o.name == name)

    @property
    def llm_steps(self) -> list[str]:
        return [s for s in self.steps if self.op(s).type in LLM_OPS]


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
        ctx = fm.get("context") or {}
        if not body.strip():
            continue
        rows.append({"_document": slug, "_page": page["sequence"], "_page_id": page["id"],
                     "text": body.strip(), "page_label": page["page_label"] or "",
                     "status": page["status"], "chapter": ctx.get("chapter") or "",
                     "section": ctx.get("section") or "", "title": ctx.get("title") or "",
                     "document_title": (render.load_document_meta(s.documents / slug).get("title") or slug)})
        if limit and len(rows) >= limit:
            break
    return rows


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
                 dry_run: bool = False):
        self.s, self.st, self.slug, self.provider = s, st, slug, provider
        self.use_cache, self.dry_run = use_cache, dry_run
        self.stats = Stats()
        self.model = getattr(provider, "model", "none")
        self.prompts: list[tuple[str, str]] = []       # (operation, prompt) — what --dry-run shows

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
    def ask(self, op: Operation, prompt: str, cache: dict, key: str, page_id: str | None) -> dict | None:
        """One structured answer, from the cache if it is there. None when it could not be had."""
        h = hashlib.sha256(prompt.encode()).hexdigest()[:24]
        if h in cache:
            self.stats.cached += 1
            return cache[h]
        self.prompts.append((op.name, prompt))
        if self.dry_run:
            return None
        if self.provider is None or not hasattr(self.provider, "complete_json"):
            self.stats.skipped += 1
            return None
        try:
            res = self.provider.complete_json(SYSTEM, prompt, op.schema, op.max_tokens)
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
            data = self.ask(op, template(op.prompt, r), cache, op.name, r.get("_page_id"))
            out.append(r | (data or {}))
        self._save(op, cache)
        return out

    def _extract(self, op: Operation, rows: list[dict]) -> list[dict]:
        """A map that must answer in the page's own words. The prompt says so and the schema is a list
        of strings, because the thing that makes an extraction checkable is that you can find it."""
        op = Operation(**(op.__dict__ | {"prompt": op.prompt + "\n\nQuote the document's own words "
                                         "exactly, and return nothing that is not in it verbatim.",
                                         "output": op.output or {"spans": "list[string]"}}))
        return self._map(op, rows)

    def _filter(self, op: Operation, rows: list[dict]) -> list[dict]:
        cache, out = self._cache(op), []
        for r in rows:
            data = self.ask(op, template(op.prompt, r), cache, op.name, r.get("_page_id"))
            if data is None:                     # unanswered: keep the row rather than silently dropping it
                out.append(r)
                continue
            keep = next((v for v in data.values() if isinstance(v, bool)), None)
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
    lead = [k for k in ("_document", "_page", "_pages", "_page_id") if k in seen]
    if fields:
        return lead + [k for k in fields if k in seen and k not in lead]
    return lead + [k for k in seen if k not in lead]


def write(rows: list[dict], path: Path, kind: str = "json", fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields:
        cols = columns(rows, fields)
        rows = [{k: r[k] for k in cols if k in r} for r in rows]
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
        if op.prompt:
            d["prompt"] = op.prompt
        if op.output:
            d["output"] = {"schema": op.output}
        if op.type == "reduce":
            d["reduce_key"] = op.reduce_key or "_all"
        if op.type == "resolve":
            d["comparison_prompt"] = op.comparison_prompt
            d["blocking_threshold"] = op.blocking_threshold
        if op.type == "unnest":
            d["unnest_key"] = op.unnest_key
        if op.type == "split":
            d["split_key"] = op.split_key
            d["method"] = "token_count"
            d["method_kwargs"] = {"num_tokens": op.words}
        if op.type == "sample":
            d["samples"] = op.n
        ops.append(d)
    spec = {"default_model": model,
            "datasets": {"pages": {"type": "file", "path": str((out_dir / "pages.json").resolve())}},
            "operations": ops,
            "pipeline": {"steps": [{"name": pipe.name, "input": "pages", "operations": pipe.steps}],
                         "output": {"type": "file", "path": str((out_dir / f"{pipe.name}.json").resolve())}}}
    (out_dir / "pipeline.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, width=100))
    return spec
