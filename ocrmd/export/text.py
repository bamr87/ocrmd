"""Plain text: the document as prose, for anything that reads neither Markdown nor YAML.

The one decision here is what to do with a disagreement mark. `==[ambitious]==[^ocrmd-1]` is the
pipeline refusing to choose between two readings, and a text export has no footnotes to keep the
refusal in. The default keeps it visible in-line — `ambitious[?ambiguous]`, the model's reading first
and the OCR reading after the question mark — because the whole repository exists to not silently
pick. `flags="llm"` or `flags="ocr"` picks one side, for a consumer that would rather have clean prose
and knows what it is giving up."""
from __future__ import annotations
import re
from pathlib import Path

from ..config import Settings
from ..store import Store
from .. import render, review
from . import default_path

FORMAT = "txt"
SEPARATOR = "-" * 12
_HEADING = re.compile(r"^#{1,6}\s+", re.M)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_EMPH = re.compile(r"(\*\*|__|`)(.+?)\1", re.S)
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")


def page_text(body: str, flags: str = "mark") -> str:
    """The page body as plain text. Flags are rendered as `flags` says; footnote lines go."""
    def flag(m: re.Match) -> str:
        n = int(m.group("n"))
        ocr = next((f.ocr for f in sites if f.n == n), "")
        llm = m.group("llm")
        if flags == "ocr":
            return ocr or llm
        if flags == "llm":
            return llm
        return f"{llm}[?{ocr}]" if ocr else llm
    if flags not in ("mark", "llm", "ocr"):
        raise ValueError("flags must be mark | llm | ocr")
    sites = review.parse_flags(body)
    text = review.MARK_RE.sub(flag, body)
    text = review.NOTE_RE.sub("", text)
    text = _IMAGE.sub(lambda m: f"[figure: {m.group(1)}]" if m.group(1) else "[figure]", text)
    text = _HEADING.sub("", text)
    text = _EMPH.sub(r"\2", text)
    text = _ITALIC.sub(r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def document_text(s: Settings, st: Store, slug: str, flags: str = "mark") -> str:
    meta = render.load_document_meta(s.documents / slug)
    head = [meta.get("title") or slug]
    if meta.get("author"):
        head.append(", ".join(meta["author"]) if isinstance(meta["author"], list) else str(meta["author"]))
    for k in ("edition", "publisher", "year", "isbn"):
        if meta.get(k):
            head.append(f"{k}: {meta[k]}")
    parts = ["\n".join(head) + "\n"]
    for page in st.pages(slug):
        p = render.page_path(s.documents / slug, page["sequence"])
        if not p.exists():
            continue
        fm, body = render.read_page(p)
        label = fm.get("page_label") or page["page_label"]
        parts.append(f"{SEPARATOR} page {page['sequence']:04d}" + (f" (p. {label})" if label else "")
                     + f" [{(fm.get('verification') or {}).get('status') or page['status']}] {SEPARATOR}\n\n"
                     + page_text(body, flags))
    return "\n".join(parts)


def export_document(s: Settings, st: Store, slug: str, out: Path | None = None, flags: str = "mark") -> Path:
    out = out or default_path(s, slug, FORMAT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(document_text(s, st, slug, flags), encoding="utf-8")
    tmp.replace(out)
    return out
