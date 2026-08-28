"""Where the words actually sat on the page — and what each of them was.

The pipeline produces two things that know different halves of the truth. Local OCR knows *where*
every line was — `Block.bbox`, normalised to the crop — but reads it imperfectly and knows nothing
about structure. The LLM pass knows *what* the page says, reconciled and corrected, and *how it is
organised* — headings, lists, tables, captions — but returns reflowed Markdown with no geometry at
all. Neither alone can put the page back together, and neither alone can be exported as anything
richer than a string.

This module joins them into the **page model**: a list of `Block`s, each with a role (`heading`,
`paragraph`, `table`, `figure`, `caption`, `math`, …), a reading order, a column, the measured
bounding box, and the lines inside it. The join is the same `SequenceMatcher` walk `reconcile` uses
to diff the two sources — one alignment, two uses — applied once for the whole page and then sliced
per line; a line the diff cannot place keeps its own OCR text rather than borrowing someone else's.
Lines are then grouped by the Markdown block their words came from, so a heading is one block, a
paragraph is one block, a table is one block with its rows as lines.

It also separates ink from words. A region that carries ink but that no OCR engine read is a figure,
a diagram, a stamp or a photograph, and `figures()` finds those by subtracting the text boxes from
the page's ink and keeping what is left over and large enough to matter. A caption the Markdown
names is joined to the nearest such region, which is the only way a caption ever meets its picture.

The model is written to `layout/<pid>.json` by the pipeline's `layout` stage and read by every
exporter (`ocrmd.export`), the web UI's overlays, and the extraction layer. Nothing downstream
re-derives the join.
"""
from __future__ import annotations
import logging, re
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .ocr import Block as OcrBlock, OcrResult

log = logging.getLogger("ocrmd.layout")

LAYOUT_VERSION = "layout@1"     # bump when the block parser, the join, or the figure detector changes
MIN_FIGURE = 0.010      # a leftover ink region smaller than this share of the page is noise, not a figure
MAX_FIGURE = 0.35       # ...and one larger than this is not a figure either, it is text nobody read
FIGURE_PAD = 0.004      # grown by this much of the page before it is cut out, so nothing is clipped
TEXT_OVERLAP = 0.25     # a candidate sitting this much inside the text block is a gap between lines
MIN_MATCH = 0.30        # below this the Markdown is not a reading of these lines at all
CAPTION_REACH = 3.0     # a caption within this many body-line heights of a figure belongs to it
FULL_WIDTH = 0.60       # a line wider than this share of the page spans the columns
GUTTER_MIN_SHARE = 0.30  # a column needs at least this share of the narrow lines on its side
GUTTER_MAX_CROSS = 0.10  # ...and at most this share may cross the gutter for it to count as one

ROLES = ("heading", "paragraph", "list", "quote", "table", "figure", "caption", "math", "code",
         "footnote", "running_head", "page_number")


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


class Block(BaseModel):
    """One structural unit of the page: a paragraph, a heading, a table, a figure…"""
    id: str
    role: str = "paragraph"
    bbox: list[float] = [0.0, 0.0, 0.0, 0.0]   # x, y, w, h normalised to the crop
    order: int = 0                  # reading order, 0-based
    column: int = 0                 # 0-based column; full-width blocks are column 0
    level: int = 0                  # heading level, 1-6
    text: str = ""                  # Markdown for text roles, LaTeX for math, alt text for figures
    lines: list[Line] = []
    confidence: float = 0.0         # mean line confidence
    source: str = "ocr"             # ocr | llm | pdf | human — who decided the role and text
    kind: str = ""                  # figure: raster | vector; how it will be reconstructed
    asset: str | None = None        # figure: file name under documents/<slug>/figures/
    caption: str | None = None      # figure: the caption the Markdown gave it
    caption_of: str | None = None   # caption: the figure block it belongs to
    style: dict = {}                # bold / italic / size_px / align when something measured them

    @property
    def height(self) -> float:
        return self.bbox[3]


class Layout(BaseModel):
    """A page as geometry: its size, its blocks in reading order, and everything on it."""
    page_id: str = ""
    width: int
    height: int
    columns: int = 1
    version: str = LAYOUT_VERSION
    blocks: list[Block] = Field(default_factory=list)

    def __init__(self, **data):
        # `Layout(width=…, height=…, lines=[…], figures=[…])` is the shape the facsimile writer and its
        # tests have always used; a flat list of lines is one paragraph block per line.
        lines = data.pop("lines", None)
        figs = data.pop("figures", None)
        super().__init__(**data)
        if lines:
            self.blocks += [Block(id=f"l{i}", role="heading" if ln.heading else "paragraph", level=ln.heading,
                                  bbox=list(ln.bbox), text=ln.text, lines=[ln], order=i, confidence=ln.confidence,
                                  source="llm" if ln.source == "reconciled" else "ocr")
                            for i, ln in enumerate(lines)]
        if figs:
            self.figures = figs

    @property
    def lines(self) -> list[Line]:
        """Every text line in reading order — what the facsimile draws."""
        return [ln for b in sorted(self.blocks, key=lambda b: b.order) for ln in b.lines]

    @property
    def figures(self) -> list[Block]:
        return [b for b in self.blocks if b.role == "figure"]

    @figures.setter
    def figures(self, figs: list) -> None:
        self.blocks = [b for b in self.blocks if b.role != "figure"]
        for i, f in enumerate(figs, 1):
            self.blocks.append(f if isinstance(f, Block) else Block(id=f"f{i}", role="figure", bbox=list(f)))
        order(self.blocks, self.columns)

    @property
    def body_size(self) -> float:
        """Median line height in page fractions — the type size everything else is judged against."""
        hs = sorted(x.height for x in self.lines)
        return hs[len(hs) // 2] if hs else 0.02

    def block(self, bid: str) -> Block | None:
        return next((b for b in self.blocks if b.id == bid), None)


# --- the Markdown side ----------------------------------------------------------------------------
@dataclass
class MdBlock:
    """One block of the reconciled Markdown, with the words a line can be aligned to."""
    role: str
    text: str                       # the block's Markdown, as written
    words: list[str]                # normalised alignable words
    level: int = 0
    raw_words: list[str] = None     # the words as they appear, for the reconciled line text

    def __post_init__(self):
        if self.raw_words is None:
            self.raw_words = []


_WORD = re.compile(r"\S+")
_MD_PREFIX = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)")
_IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$")
_CAPTION = re.compile(r"^\s*(>\s*)?(\*\*)?(fig(ure|\.)?|table|plate|chart|diagram)\b", re.I)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def _inline_words(line: str) -> list[str]:
    """The words of one Markdown line with the markup stripped — footnote refs, marks, emphasis."""
    line = re.sub(r"\[\^[^\]]+\]", "", line)             # footnote refs
    line = re.sub(r"==\[([^\]]*)\]==", r"\1", line)      # unresolved marks: keep the reading
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)     # images carry no words
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # links: the text, not the target
    line = re.sub(r"[*_`|~=]", "", line)
    return _WORD.findall(line)


def parse_markdown(md: str) -> list[MdBlock]:
    """Split a page body into structural blocks.

    Deliberately a small parser and not a Markdown library: the dialect is the one `llm.DIALECT`
    asks for (ATX headings, lists, blockquotes, GFM tables, `$$` math, images, `[^n]` footnotes)
    plus ocrmd's own marks, and what matters here is which *words* belong to which block, which a
    renderer's AST does not give back cheaply.
    """
    out: list[MdBlock] = []
    lines = md.splitlines()
    i, n = 0, len(lines)

    def flush(role: str, buf: list[str], level: int = 0) -> None:
        if not buf:
            return
        text = "\n".join(buf).strip()
        if not text:
            return
        if role in ("math", "code"):
            words = []
        else:
            words = [w for ln in buf for w in _inline_words(_MD_PREFIX.sub("", ln))]
        out.append(MdBlock(role=role, text=text, words=[_norm(w) for w in words], level=level, raw_words=words))

    para: list[str] = []
    while i < n:
        raw = lines[i]
        s = raw.strip()
        if not s:
            flush("paragraph", para); para = []; i += 1; continue
        if s.startswith("[^ocrmd-"):                    # the pipeline's own reconciliation footnotes
            flush("paragraph", para); para = []; i += 1; continue
        if s.startswith("[^") and "]:" in s:
            flush("paragraph", para); para = []
            flush("footnote", [raw]); i += 1; continue
        if s.startswith("```") or s.startswith("$$"):
            flush("paragraph", para); para = []
            fence = "```" if s.startswith("```") else "$$"
            buf = [raw]; i += 1
            closed = s.count(fence) >= 2 and len(s) > len(fence)
            while i < n and not closed:
                buf.append(lines[i]); closed = lines[i].strip().startswith(fence) or lines[i].strip().endswith(fence); i += 1
            flush("math" if fence == "$$" else "code", buf); continue
        if m := _IMAGE.match(raw):
            flush("paragraph", para); para = []
            out.append(MdBlock(role="figure", text=raw.strip(), words=[], raw_words=[]))
            out[-1].level = 0
            i += 1; continue
        if m := re.match(r"^\s{0,3}(#{1,6})\s+", raw):
            flush("paragraph", para); para = []
            flush("heading", [raw], level=len(m.group(1))); i += 1; continue
        if _TABLE_ROW.match(raw) and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            flush("paragraph", para); para = []
            buf = [raw, lines[i + 1]]; i += 2
            while i < n and _TABLE_ROW.match(lines[i]):
                buf.append(lines[i]); i += 1
            flush("table", [b for b in buf if not _TABLE_SEP.match(b)]); continue
        if re.match(r"^\s{0,3}>", raw):
            flush("paragraph", para); para = []
            buf = [raw]; i += 1
            while i < n and re.match(r"^\s{0,3}>", lines[i]):
                buf.append(lines[i]); i += 1
            role = "caption" if _CAPTION.match(buf[0]) else "quote"
            flush(role, buf); continue
        if re.match(r"^\s{0,3}([-*+]|\d+[.)])\s+", raw):
            flush("paragraph", para); para = []
            buf = [raw]; i += 1
            while i < n and (re.match(r"^\s{0,3}([-*+]|\d+[.)])\s+", lines[i]) or lines[i].startswith("  ")):
                buf.append(lines[i]); i += 1
            flush("list", buf); continue
        if s.startswith("$") and s.endswith("$") and len(s) > 2:
            flush("paragraph", para); para = []
            flush("math", [raw]); i += 1; continue
        para.append(raw); i += 1
    flush("paragraph", para)
    for b in out:                                   # a paragraph that reads like a caption is one
        if b.role == "paragraph" and _CAPTION.match(b.text) and len(b.words) <= 40:
            b.role = "caption"
    return out


def markdown_words(md: str) -> list[tuple[str, int]]:
    """Body words with the heading level of the line they came from, footnote blocks dropped."""
    return [(w, b.level) for b in parse_markdown(md) if b.role != "footnote" for w in b.raw_words]


# --- the join --------------------------------------------------------------------------------------
def build(res: OcrResult, markdown: str, size: tuple[int, int], page_id: str = "") -> Layout:
    """Join the OCR geometry to the reconciled text and group it into blocks.

    `size` is (width, height) of the page crop. Figures are not detected here — they need the image
    (`figures()`); `assemble()` does both.
    """
    lay = Layout(page_id=page_id, width=size[0], height=size[1])
    ocr_lines = [b for b in res.blocks if b.text.strip()]
    md = [b for b in parse_markdown(markdown) if b.role != "footnote"]
    # every alignable md word, tagged with the block it came from
    md_words: list[tuple[str, str, int]] = [(w, rw, bi) for bi, b in enumerate(md)
                                            for w, rw in zip(b.words, b.raw_words, strict=True)]
    a = [_norm(w) for b in ocr_lines for w in b.text.split()]
    b_ = [w for w, _, _ in md_words]
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
            log.debug("markdown and OCR share too little (%.2f) to place lines; keeping OCR text", sm.ratio())
    placed: list[tuple[Line, int | None]] = []       # (line, md block index or None)
    i = 0
    for blk in ocr_lines:
        n = len(blk.text.split())
        idx = [mapping[i + k] for k in range(n) if i + k in mapping]
        text, source, heading, owner = blk.text, "ocr", 0, None
        if len(idx) >= max(2, n // 2):            # enough of the line was placed to trust the span
            lo, hi = min(idx), max(idx) + 1
            text = " ".join(rw for _, rw, _ in md_words[lo:hi])
            owners = [bi for _, _, bi in md_words[lo:hi]]
            owner = max(set(owners), key=owners.count)
            heading = md[owner].level if md[owner].role == "heading" else 0
            source = "reconciled"
        placed.append((Line(text=text, bbox=list(blk.bbox), confidence=blk.confidence, source=source,
                            heading=heading), owner))
        i += n
    placed = _dedupe_placed(placed)
    lay.columns = detect_columns([ln for ln, _ in placed])
    lay.blocks = group(placed, md, lay.columns)
    order(lay.blocks, lay.columns)
    return lay


def _dedupe_placed(placed: list[tuple[Line, int | None]]) -> list[tuple[Line, int | None]]:
    keep = dedupe([ln for ln, _ in placed])
    owner = {id(ln): o for ln, o in placed}
    return [(ln, owner[id(ln)]) for ln in keep]


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


def union(boxes: list[list[float]]) -> list[float]:
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def detect_columns(lines: list[Line]) -> int:
    """Two columns when a persistent gutter splits the narrow lines, otherwise one.

    Counted on the lines narrower than `FULL_WIDTH` of the page: a heading or a full-measure paragraph
    says nothing about columns. The gutter is the position that the fewest narrow lines cross while
    both sides keep a real share of them — a page with an indented block quote has lines on one side
    only, and does not count.
    """
    narrow = [ln for ln in lines if ln.bbox[2] < FULL_WIDTH]
    if len(narrow) < 6:
        return 1
    best = None
    for g in np.linspace(0.35, 0.65, 31):
        left = sum(1 for ln in narrow if ln.bbox[0] + ln.bbox[2] <= g)
        right = sum(1 for ln in narrow if ln.bbox[0] >= g)
        cross = len(narrow) - left - right
        if (left >= GUTTER_MIN_SHARE * len(narrow) and right >= GUTTER_MIN_SHARE * len(narrow)
                and cross <= GUTTER_MAX_CROSS * len(narrow)):
            if best is None or cross < best[0]:
                best = (cross, g)
    return 2 if best else 1


def _column_of(bbox: list[float], columns: int) -> int:
    if columns < 2 or bbox[2] >= FULL_WIDTH:
        return 0
    return 0 if bbox[0] + bbox[2] / 2 < 0.5 else 1


def group(placed: list[tuple[Line, int | None]], md: list[MdBlock], columns: int) -> list[Block]:
    """Lines into blocks: consecutive lines of one Markdown block are one block; a run of unplaced
    lines is split into paragraphs at vertical gaps, the way the OCR-only renderer paragraphs them."""
    blocks: list[Block] = []
    # sort lines the way they will be read, so "consecutive" means consecutive in reading order
    seq = sorted(placed, key=lambda p: (round(p[0].bbox[1], 3), p[0].bbox[0]))
    heights = sorted(ln.bbox[3] for ln, _ in seq)
    med_h = heights[len(heights) // 2] if heights else 0.02
    used: set[int] = set()
    cur: list[Line] = []; cur_owner: int | None = None; cur_col = 0; prev_bottom = None

    def close():
        nonlocal cur, cur_owner
        if not cur:
            return
        bid = f"b{len(blocks) + 1}"
        if cur_owner is not None:
            m = md[cur_owner]
            used.add(cur_owner)
            blocks.append(Block(id=bid, role=m.role, level=m.level, text=m.text, lines=list(cur),
                                bbox=union([ln.bbox for ln in cur]), source="llm", column=cur_col,
                                confidence=round(sum(ln.confidence for ln in cur) / len(cur), 3)))
        else:
            blocks.append(Block(id=bid, role="paragraph", text="\n".join(ln.text for ln in cur), lines=list(cur),
                                bbox=union([ln.bbox for ln in cur]), source="ocr", column=cur_col,
                                confidence=round(sum(ln.confidence for ln in cur) / len(cur), 3)))
        cur, cur_owner = [], None

    for ln, owner in seq:
        col = _column_of(ln.bbox, columns)
        gap = prev_bottom is not None and ln.bbox[1] - prev_bottom > 0.6 * med_h
        same = cur and owner == cur_owner and col == cur_col and (owner is not None or not gap)
        if not same:
            close()
            cur_owner, cur_col = owner, col
        cur.append(ln)
        prev_bottom = ln.bbox[1] + ln.bbox[3]
    close()
    # Markdown blocks no line was placed on still exist on the page — a table OCR read as noise, a
    # caption under a figure — and are kept without geometry so exporters do not lose their text.
    for bi, m in enumerate(md):
        if bi in used or m.role in ("figure",) or not m.text.strip():
            continue
        blocks.append(Block(id=f"b{len(blocks) + 1}", role=m.role, level=m.level, text=m.text, source="llm"))
    return blocks


def order(blocks: list[Block], columns: int) -> None:
    """Reading order in place: top to bottom, and inside a band of columns left column first.

    A full-width block (a heading over two columns, a figure across the page) closes the band above
    it, so the order is: everything above the heading column by column, the heading, then the next
    band column by column. Blocks with no geometry go last, in the order the Markdown had them.
    """
    placed = [b for b in blocks if b.bbox[2] > 0 and b.bbox[3] > 0]
    loose = [b for b in blocks if not (b.bbox[2] > 0 and b.bbox[3] > 0)]
    placed.sort(key=lambda b: (round(b.bbox[1], 3), b.bbox[0]))
    band, keyed = 0, []
    for b in placed:
        col = _column_of(b.bbox, columns)
        b.column = col
        full = columns < 2 or b.bbox[2] >= FULL_WIDTH
        if full:
            band += 1
            keyed.append(((band, 0, b.bbox[1]), b))
            band += 1
        else:
            keyed.append(((band, col, b.bbox[1]), b))
    keyed.sort(key=lambda kb: kb[0])
    for i, (_, b) in enumerate(keyed):
        b.order = i
    for j, b in enumerate(loose):
        b.order = len(keyed) + j


# --- figures ---------------------------------------------------------------------------------------
def figures(img: np.ndarray, blocks: list[OcrBlock], min_area: float = MIN_FIGURE) -> list[Block]:
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
    found: list[list[float]] = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = (cw * ch) / float(w * h)
        if area < min_area or area > MAX_FIGURE:
            continue
        if ch and float(text_rows[y:y + ch].mean()) > 1 - TEXT_OVERLAP:
            continue                    # lives entirely among the text lines: it is text, not a figure
        found.append([max(0.0, x / w - FIGURE_PAD), max(0.0, y / h - FIGURE_PAD),
                      min(1.0, (cw + 2 * FIGURE_PAD * w) / w), min(1.0, (ch + 2 * FIGURE_PAD * h) / h)])
    found.sort(key=lambda f: (round(f[1], 2), f[0]))
    log.debug("found %d figure region(s)", len(found))
    return [Block(id=f"f{i}", role="figure", bbox=bb, kind="raster") for i, bb in enumerate(found, 1)]


def crop(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = int(max(0, bbox[0]) * w), int(max(0, bbox[1]) * h)
    x1, y1 = int(min(1.0, bbox[0] + bbox[2]) * w), int(min(1.0, bbox[1] + bbox[3]) * h)
    return img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]


def _caption_text(text: str) -> str:
    t = re.sub(r"^\s*>\s*", "", text, flags=re.M)
    t = re.sub(r"\*\*(figure|fig\.?|table|plate|chart|diagram)\s*:?\*\*\s*:?\s*", r"\1: ", t, flags=re.I)
    return " ".join(t.split())


def join_captions(lay: Layout) -> int:
    """Give each figure the caption nearest to it, and each caption its figure. Returns the joins made.

    Two channels that never met: the detector finds ink, the Markdown names captions. A caption's
    words were placed by the join, so it has a box; the figure it belongs to is the nearest one within
    `CAPTION_REACH` body lines above or below that overlaps it horizontally. A caption with no box
    (OCR never read it) is joined to the nearest unclaimed figure in reading order instead.
    """
    figs = [b for b in lay.figures if b.caption is None]
    caps = [b for b in lay.blocks if b.role == "caption" and b.caption_of is None]
    if not figs or not caps:
        return 0
    reach = lay.body_size * CAPTION_REACH
    n = 0
    for c in caps:
        best, best_d = None, None
        for f in figs:
            if f.caption is not None:
                continue
            if c.bbox[2] > 0 and c.bbox[3] > 0:
                hx = min(c.bbox[0] + c.bbox[2], f.bbox[0] + f.bbox[2]) - max(c.bbox[0], f.bbox[0])
                if hx <= 0:
                    continue
                d = max(c.bbox[1] - (f.bbox[1] + f.bbox[3]), f.bbox[1] - (c.bbox[1] + c.bbox[3]), 0.0)
                if d > reach:
                    continue
            else:
                d = 1.0 + f.order
            if best is None or d < best_d:
                best, best_d = f, d
        if best is not None:
            best.caption = _caption_text(c.text)
            c.caption_of = best.id
            n += 1
    return n


def assemble(res: OcrResult, markdown: str, img: np.ndarray, page_id: str = "",
             figure_boxes: list[list[float]] | None = None) -> Layout:
    """The whole page model: the join, the figures, the captions, the order.

    `figure_boxes` are figures the source already knew about (a PDF's placed images); when given, the
    ink detector is not run — a file that says where its pictures are does not need them guessed.
    """
    lay = build(res, markdown, (img.shape[1], img.shape[0]), page_id)
    if figure_boxes is not None:
        figs = [Block(id=f"f{i}", role="figure", bbox=[float(v) for v in bb], kind="raster", source="pdf")
                for i, bb in enumerate(figure_boxes, 1)]
    else:
        figs = figures(img, [b for b in res.blocks if b.text.strip()])
    lay.blocks += figs
    order(lay.blocks, lay.columns)
    join_captions(lay)
    # a Markdown image reference that is already in the body names an existing asset; keep it on the figure
    refs = [b for b in parse_markdown(markdown) if b.role == "figure"]
    for ref, f in zip(refs, lay.figures, strict=False):
        m = _IMAGE.match(ref.text)
        if m:
            f.asset = m.group(2).split("/")[-1]
            if not f.caption and m.group(1).strip():
                f.caption = m.group(1).strip()
    return lay


def merge_human_layout(new: Layout, old: Layout | None) -> list[str]:
    """Carry a person's layout decisions forward onto a fresh build. Returns what was kept.

    A block someone marked `source: human` — a figure box dragged into place, a role corrected, a
    caption typed in — or that an extraction pipeline wrote (`source: etl`, a caption the model
    read off the figure) replaces the block at the same position, or is added when the rebuild found
    nothing there. Same rule as the page body: the pipeline's output is never allowed to erase what a
    human decided.
    """
    if old is None:
        return []
    kept: list[str] = []
    for hb in [b for b in old.blocks if b.source in ("human", "etl")]:
        hit = None
        for nb in new.blocks:
            if nb.role == hb.role or (nb.role in ("figure", "caption") and hb.role in ("figure", "caption")):
                v = min(nb.bbox[1] + nb.bbox[3], hb.bbox[1] + hb.bbox[3]) - max(nb.bbox[1], hb.bbox[1])
                hx = min(nb.bbox[0] + nb.bbox[2], hb.bbox[0] + hb.bbox[2]) - max(nb.bbox[0], hb.bbox[0])
                if v > 0 and hx > 0 and v * hx > 0.5 * min(nb.bbox[2] * nb.bbox[3], hb.bbox[2] * hb.bbox[3]):
                    hit = nb; break
        if hit is not None:
            new.blocks[new.blocks.index(hit)] = hb.model_copy()
        else:
            new.blocks.append(hb.model_copy())
        kept.append(hb.id)
    if kept:
        order(new.blocks, new.columns)
    return kept
