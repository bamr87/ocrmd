"""Canonical Markdown page + document writer, and the document-level metadata file.

documents/<slug>/document.yaml holds what no page can know on its own — title, author, citation,
tags, project. `build` inherits it into every page's `context` (only where the page has no value),
carries chapter/section forward across pages that have no heading, regenerates nav links, and writes
the aggregated document file with a TOC.
"""
from __future__ import annotations
import re
from pathlib import Path
import yaml

from . import review

DOCUMENT_FIELDS = ("title", "author", "edition", "publisher", "year", "isbn", "citation", "citation_key",
                   "tags", "project", "kind", "physical")
INHERIT_FIELDS = ("title", "author", "citation_key", "tags", "project")


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", "null"))


def page_frontmatter(**fm) -> str:
    return "---\n" + yaml.dump(fm, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100) + "---\n"


def page_path(doc_dir: Path, sequence: int) -> Path:
    return doc_dir / "pages" / f"{sequence:04d}.md"


def write_page(doc_dir: Path, sequence: int, fm: dict, body: str) -> Path:
    p = page_path(doc_dir, sequence)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".md.tmp")
    tmp.write_text(page_frontmatter(**fm) + "\n" + body.rstrip() + "\n", encoding="utf-8")
    tmp.replace(p)                      # atomic: a crash mid-write never leaves a half page
    return p


def read_page(p: Path) -> tuple[dict, str]:
    return split_frontmatter(p.read_text(encoding="utf-8"))


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    _, fm, body = text.split("---\n", 2)
    return yaml.safe_load(fm) or {}, body.lstrip("\n")


# --- document.yaml ---------------------------------------------------------------------------------
def document_meta_path(doc_dir: Path) -> Path:
    return doc_dir / "document.yaml"


def load_document_meta(doc_dir: Path) -> dict:
    p = document_meta_path(doc_dir)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"{p}: invalid YAML: {e}") from e


def ensure_document_meta(doc_dir: Path, slug: str, **seed) -> dict:
    """Create document.yaml on first use (a template the user fills in), or update it with any
    non-empty seed values (--title/--tag/--project from the CLI)."""
    doc_dir.mkdir(parents=True, exist_ok=True)
    meta = load_document_meta(doc_dir)
    created = not meta
    if created:
        meta = {"slug": slug, "title": None, "author": [], "edition": None, "publisher": None, "year": None,
                "isbn": None, "citation": None, "citation_key": None, "kind": "textbook", "tags": [],
                "project": None, "physical": None, "reading_direction": "ltr"}
    changed = created
    for k, v in seed.items():
        if v in (None, [], ""):
            continue
        if k == "tags":
            merged = sorted(set(meta.get("tags") or []) | set(v))
            if merged != meta.get("tags"):
                meta["tags"] = merged; changed = True
        elif meta.get(k) != v:
            meta[k] = v; changed = True
    if changed:
        document_meta_path(doc_dir).write_text(
            "# Document-level metadata. Inherited into every page's `context` on `ocrmd build`;\n"
            "# a page's own value wins. Edit freely — the pipeline never overwrites this file.\n"
            + yaml.dump(meta, Dumper=_Dumper, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return meta


# --- build ---------------------------------------------------------------------------------------
def _roman_or_int(label: str | None) -> int | None:
    if not label:
        return None
    s = str(label).strip().lower()
    if s.isdigit():
        return int(s)
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
    if s and all(ch in vals for ch in s):
        total, prev = 0, 0
        for ch in reversed(s):
            v = vals[ch]
            total += -v if v < prev else v
            prev = max(prev, v)
        return -total            # negative so front matter sorts before arabic pages
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else None


def label_gaps(labels: list[str | None]) -> list[tuple[int, int]]:
    """Consecutive printed page numbers that skip (last, next): a page was probably missed."""
    nums = [n for n in (_roman_or_int(x) for x in labels) if n is not None and n > 0]
    gaps = []
    for a, b in zip(nums, nums[1:], strict=False):
        if 1 < b - a <= 20:      # bigger jumps are chapter skips the user meant
            gaps.append((a, b))
    return gaps


def build_document(doc_dir: Path, slug: str) -> Path:
    meta = load_document_meta(doc_dir)
    pages = sorted((doc_dir / "pages").glob("*.md"))
    parts, toc, agreements, flagged, unresolved = [], [], [], 0, 0
    statuses: dict[str, int] = {}
    labels: list[str | None] = []
    reshoot: list[int] = []
    prev_ctx: dict = {}
    for i, p in enumerate(pages):
        fm, body = read_page(p)
        prev = f"pages/{pages[i-1].name}" if i else None
        nxt = f"pages/{pages[i+1].name}" if i + 1 < len(pages) else None
        fm["nav"] = {"prev": prev, "next": nxt, "document": f"../{slug}.md"}
        ctx = fm.setdefault("context", {})
        for k in INHERIT_FIELDS:                      # document defaults fill blanks, never override
            if ctx.get(k) in (None, [], "") and meta.get(k) not in (None, [], ""):
                ctx[k] = meta[k]
        for k in ("chapter", "section"):              # carry forward across heading-less pages
            if ctx.get(k) in (None, "") and prev_ctx.get(k):
                ctx[k] = prev_ctx[k]
        if ctx.get("chapter") != prev_ctx.get("chapter") and ctx.get("chapter"):
            toc.append((ctx["chapter"], fm["sequence"], fm.get("page_label")))
        prev_ctx = ctx
        p.write_text(page_frontmatter(**fm) + "\n" + body, encoding="utf-8")
        v = fm.get("verification", {}) or {}
        if v.get("agreement") is not None:
            agreements.append(v["agreement"])
        status = v.get("status", "unverified")
        flagged += status == "flagged"
        unresolved += review.unresolved(body)
        statuses[status] = statuses.get(status, 0) + 1
        if status in ("illegible", "suspect"):
            reshoot.append(fm["sequence"])
        labels.append(fm.get("page_label"))
        anchor = f"<!-- page: {fm['sequence']:04d} label: {fm.get('page_label')} id: {fm['id']} status: {status} -->"
        parts.append(f"{anchor}\n\n{body.rstrip()}\n")
    known = [x for x in labels if x]
    doc_fm = {"ocrmd": 1, "document": slug,
              **{k: meta.get(k) for k in DOCUMENT_FIELDS if k in meta},
              "pages": len(pages), "page_labels": f"{known[0]}–{known[-1]}" if known else None,
              "statuses": dict(sorted(statuses.items())), "flagged": flagged, "unresolved_flags": unresolved,
              "needs_reshoot": reshoot, "label_gaps": [f"{a}→{b}" for a, b in label_gaps(labels)],
              "agreement": round(sum(agreements) / len(agreements), 4) if agreements else None}
    out = doc_dir / f"{slug}.md"
    title = meta.get("title") or slug
    toc_md = "\n".join(f"- {c} (page {s:04d}{f', p. {lbl}' if lbl else ''})" for c, s, lbl in toc)
    out.write_text(page_frontmatter(**doc_fm) + f"\n# {title}\n\n" + (toc_md + "\n\n" if toc_md else "")
                   + "\n".join(parts), encoding="utf-8")
    return out
