"""Where the words actually sat on the page.

The pipeline produces two things that know different halves of the truth. Local OCR knows *where*
every line was — `Block.bbox`, normalised to the crop — but reads it imperfectly. The LLM pass knows
*what* the page says, reconciled and corrected, but returns reflowed Markdown with no geometry at all.
Neither alone can put the page back together.

This module joins them: each OCR line is matched to the span of the finished Markdown that
corresponds to it, so a line carries the *reconciled* words at the *measured* position. The matching
is the same `SequenceMatcher` walk `reconcile` uses to diff the two sources — one alignment, two
uses, and a line the diff cannot place keeps its own OCR text rather than borrowing someone else's.

It also separates ink from words. A region that carries ink but that no OCR engine read is a figure,
a diagram, a stamp or a photograph, and `figures()` finds those by subtracting the text boxes from
the page's ink and keeping what is left over and large enough to matter.
"""
from __future__ import annotations
import logging, re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np

from .ocr import Block, OcrResult

log = logging.getLogger("ocrmd.layout")

MIN_FIGURE = 0.010      # a leftover ink region smaller than this share of the page is noise, not a figure
MAX_FIGURE = 0.35       # ...and one larger than this is not a figure either, it is text nobody read
FIGURE_PAD = 0.004      # grown by this much of the page before it is cut out, so nothing is clipped
TEXT_OVERLAP = 0.25     # a candidate sitting this much inside the text block is a gap between lines
MIN_MATCH = 0.30        # below this the Markdown is not a reading of these lines at all


@dataclass
class Line:
    """One line of the page, at the place it was photographed."""
    text: str                       # the best available reading: reconciled where it could be matched
    bbox: list[float]               # x, y, w, h normalised to the crop
    confidence: float = 0.0
    source: str = "ocr"             # ocr | reconciled
    heading: int = 0                # 1-6 when the Markdown says this line is a heading

    @property
    def height(self) -> float:
        return self.bbox[3]


@dataclass
class Figure:
    """A region carrying ink that no engine read as text."""
    bbox: list[float]
    kind: str = "raster"            # raster | vector — how it will be reconstructed
    caption: str | None = None


@dataclass
class Layout:
    """A page as geometry: its size, its lines, and everything on it that is not words."""
    width: int
    height: int
    lines: list[Line] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)

    @property
    def body_size(self) -> float:
        """Median line height in page fractions — the type size everything else is judged against."""
        hs = sorted(x.height for x in self.lines)
        return hs[len(hs) // 2] if hs else 0.02


_WORD = re.compile(r"\S+")
_MD_PREFIX = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)")


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def markdown_words(md: str) -> list[tuple[str, int]]:
    """Body words with the heading level of the line they came from, footnote blocks dropped."""
    out: list[tuple[str, int]] = []
    for raw in md.splitlines():
        if raw.lstrip().startswith("[^"):        # ocrmd's own reconciliation footnotes
            continue
        h = len(m.group(1).strip()) if (m := re.match(r"^\s{0,3}(#{1,6})\s+", raw)) else 0
        line = _MD_PREFIX.sub("", raw)
        line = re.sub(r"\[\^[^\]]+\]", "", line)             # footnote refs
        line = re.sub(r"==\[([^\]]*)\]==", r"\1", line)      # unresolved marks: keep the reading
        line = re.sub(r"[*_`|]", "", line)
        out += [(w, h) for w in _WORD.findall(line)]
    return out


def build(res: OcrResult, markdown: str, size: tuple[int, int]) -> Layout:
    """Join the OCR geometry to the reconciled text. `size` is (width, height) of the page crop."""
    lay = Layout(width=size[0], height=size[1])
    blocks = [b for b in res.blocks if b.text.strip()]
    md = markdown_words(markdown)
    a = [_norm(w) for b in blocks for w in b.text.split()]
    b_ = [_norm(w) for w, _ in md]
    # One alignment for the whole page, then sliced per line: matching line by line would let a line
    # steal words from further down the page whenever OCR dropped one. `replace` spans are adopted
    # too — a corrected word differs from the OCR word by definition, so taking only exact matches
    # would throw away the corrections that are the point of reconciling.
    mapping: dict[int, int] = {}
    if a and b_:
        sm = SequenceMatcher(None, a, b_, autojunk=False)
        # ...but only when the two really are readings of the same page. Positional pairing inside a
        # `replace` span is meaningless otherwise, and would quietly stamp an unrelated document's
        # words onto this page's line boxes.
        if sm.quick_ratio() >= MIN_MATCH and sm.ratio() >= MIN_MATCH:
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op in ("equal", "replace"):
                    for k in range(min(i2 - i1, j2 - j1)):
                        mapping[i1 + k] = j1 + k
        else:
            log.debug("markdown and OCR share too little (%.2f) to place lines; keeping OCR text",
                      sm.ratio())
    i = 0
    for blk in blocks:
        n = len(blk.text.split())
        idx = [mapping[i + k] for k in range(n) if i + k in mapping]
        text, source, heading = blk.text, "ocr", 0
        if len(idx) >= max(2, n // 2):            # enough of the line was placed to trust the span
            lo, hi = min(idx), max(idx) + 1
            text = " ".join(w for w, _ in md[lo:hi])
            heading = md[lo][1]
            source = "reconciled"
        lay.lines.append(Line(text=text, bbox=list(blk.bbox), confidence=blk.confidence,
                              source=source, heading=heading))
        i += n
    lay.lines = dedupe(lay.lines)
    return lay


def dedupe(lines: list[Line]) -> list[Line]:
    """Drop lines that are really the same line seen twice.

    The cross-engine merge fills in lines one engine missed, and two engines box the same row of text
    slightly differently, so a page can carry two entries for one line. On a reflowed document that is
    invisible; on a facsimile they are drawn on top of each other. The longer reading wins, because a
    fill is usually the shorter fragment.
    """
    out: list[Line] = []
    for ln in sorted(lines, key=lambda x: (round(x.bbox[1], 3), x.bbox[0])):
        hit = None
        for kept in out:
            v = min(kept.bbox[1] + kept.bbox[3], ln.bbox[1] + ln.bbox[3]) - max(kept.bbox[1], ln.bbox[1])
            hx = min(kept.bbox[0] + kept.bbox[2], ln.bbox[0] + ln.bbox[2]) - max(kept.bbox[0], ln.bbox[0])
            if v > 0.5 * min(kept.bbox[3], ln.bbox[3]) and hx > 0.5 * min(kept.bbox[2], ln.bbox[2]):
                hit = kept
                break
        if hit is None:
            out.append(ln)
        elif len(ln.text) > len(hit.text):
            out[out.index(hit)] = ln
    return sorted(out, key=lambda x: (round(x.bbox[1], 2), x.bbox[0]))


def figures(img: np.ndarray, blocks: list[Block], min_area: float = MIN_FIGURE) -> list[Figure]:
    """Ink regions no engine read as text: diagrams, photographs, stamps, handwriting.

    Text boxes are painted out of the page's ink mask and whatever survives — once it has been closed
    up into blobs and filtered by area — is something the transcription does not contain. Getting this
    wrong in the safe direction matters more than precision: a figure wrongly kept is an extra picture
    in the reconstruction, a figure missed is a hole in it.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    detail = np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 12))
    ink = (detail > 12).astype(np.uint8) * 255
    for b in blocks:                              # erase the text, generously
        x0, y0 = int((b.bbox[0] - 0.004) * w), int((b.bbox[1] - 0.006) * h)
        x1, y1 = int((b.bbox[0] + b.bbox[2] + 0.004) * w), int((b.bbox[1] + b.bbox[3] + 0.006) * h)
        ink[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 60) | 1, max(3, h // 60) | 1))
    ink = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
                           cv2.MORPH_CLOSE, k)
    # Where the text *block* is, as opposed to where individual lines are. Erasing the line boxes
    # leaves the gaps between them, and closing those gaps up joins them into one blob covering the
    # whole column — which then gets pasted back over the reconstruction as a picture of the page.
    text_rows = np.zeros(h, dtype=bool)
    for b in blocks:
        y0 = int(max(0.0, b.bbox[1] - b.bbox[3] * 0.7) * h)
        y1 = int(min(1.0, b.bbox[1] + b.bbox[3] * 1.7) * h)
        text_rows[y0:y1] = True
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Figure] = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = (cw * ch) / float(w * h)
        if area < min_area or area > MAX_FIGURE:
            continue
        if ch and float(text_rows[y:y + ch].mean()) > 1 - TEXT_OVERLAP:
            continue                    # lives entirely among the text lines: it is text, not a figure
        out.append(Figure(bbox=[max(0.0, x / w - FIGURE_PAD), max(0.0, y / h - FIGURE_PAD),
                                min(1.0, (cw + 2 * FIGURE_PAD * w) / w),
                                min(1.0, (ch + 2 * FIGURE_PAD * h) / h)]))
    out.sort(key=lambda f: (round(f.bbox[1], 2), f.bbox[0]))
    log.debug("found %d figure region(s)", len(out))
    return out


def crop(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = int(max(0, bbox[0]) * w), int(max(0, bbox[1]) * h)
    x1, y1 = int(min(1.0, bbox[0] + bbox[2]) * w), int(min(1.0, bbox[1] + bbox[3]) * h)
    return img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
