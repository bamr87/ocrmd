"""Measure local OCR against the LLM transcriptions already in the store.

There is no hand-made ground truth for a scanned book, but every verified page carries a model
transcription that read the image independently of the OCR engine. Scoring OCR text against it
(bag-of-words recall / precision / F1, order-insensitive because engines read columns and contents
pages in different orders) is what chose the engines, the upscaling and the merge rules in `ocr.py`;
`ocrmd ocr-bench` repeats that measurement for the current configuration, so a new engine, a new
camera or a changed threshold is a number rather than an impression.
"""
from __future__ import annotations
import collections, re, statistics, time
from dataclasses import dataclass, field
import cv2
from .config import Settings
from .store import Store
from . import ocr
from .llm import LLMResult

_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "­": ""})


def words(s: str) -> list[str]:
    s = s.translate(_FOLD)
    s = re.sub(r"(\w)-\s*\n\s*(?=[a-z])", r"\1", s)                       # re-join line-end hyphenation
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) >= 2]


def reference_text(md: str) -> str:
    md = re.sub(r"\[(uncertain|illegible)[^\]]*\]", " ", md, flags=re.I)
    return re.sub(r"^#{1,6}\s+|^\s*[-*>]\s+|\[\^[^\]]+\]:?|[*_`|=☐]", "", md, flags=re.M)


def score(ref: str, hyp: str) -> tuple[float, float, float]:
    r, h = collections.Counter(words(ref)), collections.Counter(words(hyp))
    inter = sum((r & h).values())
    rec, prec = inter / max(1, sum(r.values())), inter / max(1, sum(h.values()))
    return rec, prec, 2 * rec * prec / max(1e-9, rec + prec)


@dataclass
class Row:
    page_id: str
    document: str
    sequence: int
    width: int
    recall: float
    precision: float
    f1: float
    seconds: float
    agreement: float | None


@dataclass
class Report:
    version: str
    rows: list[Row] = field(default_factory=list)
    skipped: int = 0

    def mean(self, attr: str, rows=None) -> float | None:
        rows = self.rows if rows is None else rows
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        return statistics.mean(vals) if vals else None

    def summary(self) -> dict:
        small = [r for r in self.rows if r.width and r.width < 600]
        big = [r for r in self.rows if not r.width or r.width >= 600]
        return {"version": self.version, "pages": len(self.rows), "skipped": self.skipped,
                "recall": self.mean("recall"), "precision": self.mean("precision"), "f1": self.mean("f1"),
                "f1_under_600px": self.mean("f1", small), "f1_600px_plus": self.mean("f1", big),
                "seconds_per_page": self.mean("seconds"), "engine_agreement": self.mean("agreement")}


def reference_pages(s: Settings, st: Store, doc: str | None = None, min_words: int = 30) -> list[tuple[dict, str]]:
    """Pages with a legible LLM transcription of at least `min_words` words, and their reference text."""
    out = []
    for page in st.pages(doc):
        if page["status"] in ("suspect", "illegible", "blank"):
            continue
        art = s.llm_dir / f"{page['id']}.{s.provider}.json"
        if not art.exists():
            continue
        try:
            tr = LLMResult.model_validate_json(art.read_text()).transcription
        except ValueError:
            continue
        if not tr.legible:
            continue
        ref = reference_text(tr.markdown)
        if len(ref.split()) >= min_words:
            out.append((page, ref))
    return out


def run(s: Settings, st: Store, doc: str | None = None, engines: list[str] | None = None,
        limit: int | None = None, on_page=None) -> Report:
    """Recognise every reference page with the given engines (default: the configured set) and score it."""
    rec = ocr.Recognizer(engines, list(s.ocr_languages) or None, s.ocr_min_side) if engines else ocr.recognizer(s)
    report = Report(version=rec.version)
    refs = reference_pages(s, st, doc)
    if limit:
        refs = refs[:limit]
    for page, ref in refs:
        img = cv2.imread(str(s.processed / f"{page['id']}.png"))
        if img is None:
            report.skipped += 1
            continue
        t0 = time.monotonic()
        res = rec.run(img)
        r, p, f = score(ref, res.text)
        row = Row(page["id"], page["document"], page["sequence"], page["width_px"] or 0, r, p, f,
                  time.monotonic() - t0, res.agreement)
        report.rows.append(row)
        if on_page:
            on_page(row)
    return report
