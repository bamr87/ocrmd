"""The inventory: what has been scanned, what is known about each document, and search across all of it.

`ocrmd.db` already records every page and every stage; `document.yaml` holds what a person knows about
the book — title, author, year, ISBN, tags. This module joins the two and keeps the join queryable
in SQL. `catalog` mirrors each document.yaml onto its `documents` row whenever the file has changed,
so any tool that can read SQLite can answer "what has been scanned here". The page bodies are indexed
into `page_text` (plus an FTS5 table when SQLite has one) **lazily, by file mtime**: no write path has
to remember to update an index, a page edited by hand in an editor is picked up the next time anyone
searches, and deleting the index costs nothing but the next search. `search` returns hits with the
page they sit on — a hit you cannot take back to the page is a rumour, like every other kind of
answer in this repository.
"""
from __future__ import annotations
import logging, re
from dataclasses import dataclass, field, asdict
from .config import Settings
from .store import Store
from . import render

log = logging.getLogger("ocrmd.library")
MIRRORED = ("title", "author", "kind", "year", "isbn", "publisher", "tags", "project", "citation_key")


@dataclass
class DocumentInfo:
    """One row of the inventory: the catalogue half from document.yaml, the progress half from the store."""
    slug: str
    title: str | None = None
    author: list[str] = field(default_factory=list)
    kind: str | None = None
    year: int | None = None
    isbn: str | None = None
    publisher: str | None = None
    tags: list[str] = field(default_factory=list)
    project: str | None = None
    citation_key: str | None = None
    pages: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    created_at: str | None = None
    last_captured_at: str | None = None
    page_labels: str | None = None            # first–last printed page label
    built: bool = False                       # documents/<slug>/<slug>.md exists
    exports: dict[str, str] = field(default_factory=dict)   # format -> path of the last export
    path: str = ""

    @property
    def done(self) -> int:
        return self.counts.get("verified", 0) + self.counts.get("reviewed", 0)

    @property
    def needs_attention(self) -> int:
        return sum(n for k, n in self.counts.items() if k in ("flagged", "suspect", "illegible", "unverified", "error"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["done"] = self.done
        d["needs_attention"] = self.needs_attention
        return d

    def describe(self) -> str:
        who = ", ".join(self.author) if self.author else ""
        what = " · ".join(x for x in (self.kind or "", str(self.year) if self.year else "", who) if x)
        return (f"{self.slug:<22} {(self.title or '—')[:38]:<38} {self.pages:>5} page(s)  {self.done:>4} done"
                + (f"  [{what}]" if what else ""))


@dataclass
class Hit:
    """One search result, with the page it sits on."""
    document: str
    sequence: int
    page_id: str
    page_label: str | None
    status: str | None
    snippet: str
    title: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _as_list(v) -> list[str]:
    if v in (None, "", []):
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [str(x).strip() for x in v if str(x).strip()]


def sync_document(s: Settings, st: Store, slug: str) -> bool:
    """Mirror document.yaml onto the `documents` row when the file is newer than the last mirror.
    Returns whether anything was written. Cheap enough to call on every catalogue read: one stat."""
    st.ensure_document(slug)
    doc_dir = s.documents / slug
    p = render.document_meta_path(doc_dir)
    mtime = p.stat().st_mtime if p.exists() else 0.0
    row = st.document_row(slug)
    if row is not None and row["meta_mtime"] is not None and row["meta_mtime"] >= mtime:
        return False
    try:
        meta = render.load_document_meta(doc_dir) if p.exists() else {}
    except RuntimeError as e:                  # a hand-edited yaml with a typo must not break the inventory
        log.warning("%s: %s", slug, e)
        meta = {}
    year = meta.get("year")
    m = re.match(r"^\s*(\d{3,4})", str(year)) if year not in (None, "") else None
    st.set_document_meta(slug, mtime, title=meta.get("title") or None, author=", ".join(_as_list(meta.get("author"))) or None,
                         kind=meta.get("kind") or None, year=int(m.group(1)) if m else None, isbn=str(meta["isbn"]) if meta.get("isbn") else None,
                         publisher=meta.get("publisher") or None, tags=",".join(_as_list(meta.get("tags"))) or None,
                         project=meta.get("project") or None, citation_key=meta.get("citation_key") or None)
    return True


def info(s: Settings, st: Store, slug: str) -> DocumentInfo:
    if slug not in st.documents():
        raise KeyError(f"no such document: {slug}")
    sync_document(s, st, slug)
    row = st.document_row(slug)
    pages = st.pages(slug)
    labels = [p["page_label"] for p in pages if p["page_label"]]
    doc_dir = s.documents / slug
    return DocumentInfo(
        slug=slug, title=row["title"], author=_as_list(row["author"]), kind=row["kind"], year=row["year"],
        isbn=row["isbn"], publisher=row["publisher"], tags=_as_list(row["tags"]), project=row["project"],
        citation_key=row["citation_key"], pages=len(pages), counts=st.counts(slug), created_at=row["created_at"],
        last_captured_at=st.last_captured(slug), page_labels=f"{labels[0]}–{labels[-1]}" if labels else None,
        built=(doc_dir / f"{slug}.md").exists(), exports={r["format"]: r["path"] for r in st.doc_exports(slug)},
        path=str(doc_dir))


def catalog(s: Settings, st: Store) -> list[DocumentInfo]:
    """Every document, newest first."""
    return sorted((info(s, st, slug) for slug in st.documents()),
                  key=lambda d: (d.last_captured_at or d.created_at or ""), reverse=True)


def reindex(s: Settings, st: Store, slug: str | None = None, force: bool = False) -> int:
    """Bring the page-text index up to date with the page files. Returns how many pages were (re)read.

    By mtime, not by write path: a page can be rewritten by the pipeline, the web UI, `ocrmd review`,
    an annotation sync, an extraction write-back or a text editor, and none of them has to know that
    a search index exists."""
    n = 0
    for doc in ([slug] if slug else st.documents()):
        have = st.indexed(doc)
        for page in st.pages(doc):
            p = render.page_path(s.documents / doc, page["sequence"])
            if not p.exists():
                if page["id"] in have:
                    st.drop_index(page["id"])
                continue
            mtime = p.stat().st_mtime
            if not force and have.get(page["id"]) == mtime:
                continue
            try:
                fm, body = render.read_page(p)
            except Exception as e:                # a page with broken YAML must not block search
                log.warning("not indexed: %s (%s)", p, e)
                continue
            st.index_page(page["id"], doc, page["sequence"], fm.get("page_label") or page["page_label"],
                          (fm.get("verification") or {}).get("status") or page["status"], body, mtime)
            n += 1
    return n


def _excerpt(body: str, query: str, width: int = 90) -> str:
    """The substring fallback's snippet: the match with some context, the match marked like FTS does."""
    text = " ".join(body.split())
    i = text.lower().find(query.lower().strip())
    if i < 0:
        return text[:2 * width] + ("…" if len(text) > 2 * width else "")
    j = i + len(query.strip())
    pre, post = text[max(0, i - width):i], text[j:j + width]
    return ("…" if i > width else "") + pre + "[" + text[i:j] + "]" + post + ("…" if j + width < len(text) else "")


def search(s: Settings, st: Store, query: str, doc: str | None = None, limit: int = 50) -> list[Hit]:
    query = query.strip()
    if not query:
        return []
    reindex(s, st, doc)
    titles = {r["slug"]: r["title"] for r in st.document_rows()}
    out = []
    for r in st.search(query, doc, limit):
        snippet = r["snippet"] if r["rank"] != 0 or st.fts else _excerpt(r["snippet"], query)
        out.append(Hit(document=r["document"], sequence=r["sequence"], page_id=r["page_id"], page_label=r["page_label"],
                       status=r["status"], snippet=" ".join(str(snippet).split()), title=titles.get(r["document"])))
    return out
