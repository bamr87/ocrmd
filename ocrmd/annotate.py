"""Keep the Markdown in sync with what someone writes on the paper.

Point the camera at a page that has already been scanned, underline a phrase, run a highlighter over
a sentence, scribble a note in the margin — and the page file picks all of it up. The book stays the
document of record; the Markdown follows it.

The design constraint that shapes everything here is the one the operator asked for: **OCR and the
LLM run only when something actually changed.** That is achievable because by this point the page has
already been read once and every line's position is on disk in `ocr/<id>.json`. So:

    identify  →  align  →  diff  →  classify  →  apply

Only the last step ever costs anything, and only for the marks that need it:

- an **underline** or a **highlight** costs nothing at all. The words under the mark are found by
  intersecting it with line boxes the store already has, so the Markdown is edited directly. No
  recognition of any kind runs.
- a **handwritten note** costs one LLM call, because handwriting is the one thing on the page that
  has never been transcribed. Local OCR is not asked: it is trained on print and reads cursive as
  noise, which is worse than not guessing.
- **nothing new on the page** costs nothing. The diff is a subtraction on a 900-px greyscale, and a
  page with no new ink stops there.

Two channels, because pens and highlighters do opposite things. A pen *darkens* paper. A highlighter
*colours* it while leaving it bright — subtracting luminance finds a pen and misses a highlighter
almost entirely, which is why saturation is looked at separately.

What this deliberately does not do is re-transcribe the printed text. If the print appears to have
changed, the page has been mis-identified or the alignment has failed, and inventing an edit from
that would be exactly the silent corruption the rest of the pipeline is built to avoid — so a diff
that is too large is rejected as a misalignment rather than applied.
"""
from __future__ import annotations
import logging, re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .ocr import Block

log = logging.getLogger("ocrmd.annotate")

ALIGN_WIDTH = 900          # the scale everything is compared at
MIN_MARK_AREA = 0.00015    # a blob smaller than this share of the page is sensor noise, not a pen
UNDERLINE_RATIO = 4.0      # a mark this much wider than it is tall is a rule, not handwriting
MAX_CHANGE = 0.12          # more of the page than this changed: a misalignment, not an annotation
ALIGN_MIN_CORRELATION = 0.5  # below this the two views are not the same page, whatever ECC returned
BORDER = 0.012             # blobs this close to the page edge are warp artifacts, not annotations
MIN_NOTE_AREA = 0.0015     # ...and a note has to be big enough to be handwriting worth a model call
RULE_JOIN = 21             # px at ALIGN_WIDTH: how far apart two fragments of one rule may sit
MARK_STYLE = {"highlight": ("==", "=="), "underline": ("**", "**"), "strike": ("~~", "~~")}


@dataclass
class Mark:
    """Something on the paper that was not there when the page was scanned."""
    kind: str                       # highlight | underline | strike | note
    bbox: list[float]               # x, y, w, h normalised to the page crop
    colour: str = ""                # a rough name for a highlighter, for the note the operator sees
    words: list[str] = field(default_factory=list)   # the printed words it covers, in reading order
    text: str = ""                  # for a note: the transcription, once something has read it

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]

    def describe(self) -> str:
        what = " ".join(self.words)[:60] or self.text[:60] or "—"
        return f"{self.kind}{f' ({self.colour})' if self.colour else ''}: {what}"


@dataclass
class Diff:
    """Everything new on one page, and whether it can be trusted."""
    marks: list[Mark] = field(default_factory=list)
    changed: float = 0.0            # share of the page that differs
    aligned: bool = True            # False when the two views could not be registered
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.aligned and self.changed <= MAX_CHANGE

    def needs_llm(self) -> list[Mark]:
        return [m for m in self.marks if m.kind == "note" and not m.text]


# --- alignment ------------------------------------------------------------------------------------
def _grey(img: np.ndarray, width: int = ALIGN_WIDTH) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = max(1, int(g.shape[0] * width / g.shape[1]))
    return cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA)


def align(new: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Warp `new` onto `reference`'s frame. Returns (aligned image, correlation) or (None, score).

    Both are crops of the same page produced by the same preprocessing, so they are already close;
    what is left is a few pixels of translation and a fraction of a degree of rotation from the book
    having been touched. ECC handles exactly that, and refuses when it cannot — an unconverged warp
    would put a highlight over the wrong line.
    """
    ref, mov = _grey(reference), _grey(new)
    if mov.shape != ref.shape:
        mov = cv2.resize(mov, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    # Plain greyscale, not a high pass. High-passing first was tried and is measurably wrong: ECC's
    # correlation coefficient is already invariant to brightness and contrast, which is the whole
    # point of it, and removing the low frequencies takes away the structure it converges on. On a
    # real page nudged 5 px and 0.6 degrees, greyscale registers at cc 0.99 and high-passed input at
    # 0.38 with the shift recovered as 21 px instead of 3.
    fa, fb = np.float32(ref), np.float32(mov)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            fa, fb, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 5)
    except cv2.error as e:
        log.debug("ECC did not converge: %s", e)
        return None, 0.0
    if cc < ALIGN_MIN_CORRELATION:
        log.debug("alignment too weak (cc=%.3f)", cc)
        return None, float(cc)
    colour = cv2.warpAffine(cv2.resize(new, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA),
                            warp, (ref.shape[1], ref.shape[0]),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return colour, float(cc)


# --- what is new ----------------------------------------------------------------------------------
def _colour_name(bgr_mean) -> str:
    h = float(cv2.cvtColor(np.uint8([[bgr_mean]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    for lo, hi, name in ((20, 40, "yellow"), (40, 85, "green"), (85, 130, "blue"),
                         (130, 165, "pink"), (165, 180, "red"), (0, 20, "orange")):
        if lo <= h < hi:
            return name
    return "colour"


def new_ink(reference: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pen mask, highlighter mask) at alignment scale.

    A pen leaves ink where there was paper: look for pixels that got markedly *darker*. A highlighter
    leaves colour while keeping the paper bright, so it barely registers in luminance at all — it is
    found as a rise in saturation instead. Running only the first test finds every pen stroke and
    almost no highlighter, which is the mistake this pair of masks exists to avoid.
    """
    ref_g, new_g = _grey(reference), _grey(aligned)
    darker = cv2.subtract(ref_g, new_g)                     # positive where the new view is darker
    pen = (darker > 40).astype(np.uint8) * 255
    ref_s = cv2.cvtColor(cv2.resize(reference, (ref_g.shape[1], ref_g.shape[0])), cv2.COLOR_BGR2HSV)[:, :, 1]
    new_s = cv2.cvtColor(cv2.resize(aligned, (ref_g.shape[1], ref_g.shape[0])), cv2.COLOR_BGR2HSV)[:, :, 1]
    hl = ((cv2.subtract(new_s, ref_s) > 45) & (new_g > 90)).astype(np.uint8) * 255
    pen = cv2.morphologyEx(pen, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Join a rule back together. Where an underline passes under existing print it adds no darkness —
    # the pixels were already dark — so the stroke arrives as a row of fragments, and a rule drawn
    # close under a line of text can come back as a single word's worth of mark.
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (RULE_JOIN, 3)))
    hl = cv2.morphologyEx(hl, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return pen, hl


def _blobs(mask: np.ndarray, min_area: float) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if (cw * ch) / float(w * h) >= min_area:
            out.append((x, y, cw, ch))
    return out


def word_boxes(block: Block, img: np.ndarray) -> list[tuple[str, float, float]]:
    """(word, x0, x1) within a line, normalised to the page.

    The stored boxes are per *line*, so the words inside one have to be placed. The gaps between them
    are visible in the line's own ink profile, so they are measured rather than assumed — a run of
    blank columns wide enough to be a space splits the line, and the pieces are handed to the words in
    order. When the count of gaps does not match the count of words (a stray mark, a joined pair) the
    line falls back to sharing its width out by word length, which is never worse than refusing.
    """
    words = block.text.split()
    x, y, w_, h_ = block.bbox
    if not words:
        return []
    H, W = img.shape[:2]
    x0, x1 = int(max(0, x) * W), int(min(1.0, x + w_) * W)
    y0, y1 = int(max(0, y) * H), int(min(1.0, y + h_) * H)
    strip = img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    cuts: list[float] = []
    if strip.size and len(words) > 1:
        g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
        prof = np.abs(g.astype(np.float32) - cv2.GaussianBlur(g.astype(np.float32), (0, 0), 6)).mean(axis=0)
        blank = prof < max(prof.max() * 0.12, 1.0)
        runs, start = [], None
        for i, b in enumerate(list(blank) + [False]):
            if b and start is None:
                start = i
            elif not b and start is not None:
                if i - start >= max(2, strip.shape[0] * 0.18):     # a space, not a letter gap
                    runs.append((start + i) / 2)
                start = None
        if len(runs) == len(words) - 1:
            cuts = [x + (c / max(1, strip.shape[1])) * w_ for c in runs]
    if not cuts:
        total = sum(len(t) + 1 for t in words)
        acc, cuts = 0, []
        for t in words[:-1]:
            acc += len(t) + 1
            cuts.append(x + w_ * acc / total)
    edges = [x] + cuts + [x + w_]
    return [(t, edges[i], edges[i + 1]) for i, t in enumerate(words)]


def _covered(mark: list[float], blocks: list[Block], img: np.ndarray, band_up: float = 0.0,
             coverage: float = 0.5) -> list[str]:
    """Printed words a mark applies to. `band_up` extends the search *upward*, for underlines: an
    underline is drawn below the words it marks, so the text it refers to is above the rule.

    `coverage` is the share of a line's height the mark must cover to count. For a highlighter it is
    half: any overlap at all was tried first and picks up the neighbours, because a stroke is about as
    tall as the line it covers, OCR boxes include the descender space, and on a tight page that
    reaches into the rows above and below — a highlight came back carrying words from three lines.
    For a rule it is zero, because a rule is thin by construction and could never cover half of
    anything; there, the mark's own position is what places it.
    """
    mx0, my0 = mark[0], mark[1] - band_up
    mx1, my1 = mark[0] + mark[2], mark[1] + mark[3]
    out: list[str] = []
    for b in blocks:
        bx, by, bw, bh = b.bbox
        overlap = min(by + bh, my1) - max(by, my0)
        if bh <= 0 or overlap <= 0 or overlap < coverage * bh or bx > mx1 or bx + bw < mx0:
            continue
        for word, wx0, wx1 in word_boxes(b, img):
            if min(wx1, mx1) - max(wx0, mx0) > 0.35 * max(wx1 - wx0, 1e-6):
                out.append(word)
    return out


def _at_border(bbox: list[float]) -> bool:
    """A blob hard against the page edge. Warping one view onto the other replicates the border, and
    a fraction of a degree of rotation turns that into a dark rim — which is not something anybody
    wrote on the paper."""
    return (bbox[0] <= BORDER or bbox[1] <= BORDER
            or bbox[0] + bbox[2] >= 1 - BORDER or bbox[1] + bbox[3] >= 1 - BORDER)


def classify(pen: np.ndarray, hl: np.ndarray, blocks: list[Block], img: np.ndarray) -> list[Mark]:
    """Turn the two masks into marks with the words they apply to."""
    H, W = pen.shape[:2]
    marks: list[Mark] = []
    line_h = float(np.median([b.bbox[3] for b in blocks])) if blocks else 0.02
    for x, y, w_, h_ in _blobs(hl, MIN_MARK_AREA * 4):
        bbox = [x / W, y / H, w_ / W, h_ / H]
        if _at_border(bbox):
            continue
        region = cv2.resize(img, (W, H))[y:y + h_, x:x + w_]
        m = Mark("highlight", bbox, colour=_colour_name(region.reshape(-1, 3).mean(axis=0)))
        m.words = _covered(bbox, blocks, img)
        if m.words:
            marks.append(m)
    for x, y, w_, h_ in _blobs(pen, MIN_MARK_AREA):
        bbox = [x / W, y / H, w_ / W, h_ / H]
        if _at_border(bbox):
            continue
        if w_ >= UNDERLINE_RATIO * h_ and h_ / H < line_h * 0.6:
            # A long thin rule. Through the middle of a line it strikes it out; at or below the
            # baseline it underlines it. "Anywhere inside the line box" is not the test — OCR boxes
            # carry descender space, so a rule drawn just under the text still falls inside one.
            centre = bbox[1] + bbox[3] / 2
            through = [b for b in blocks
                       if b.bbox[1] + 0.15 * b.bbox[3] < centre < b.bbox[1] + 0.70 * b.bbox[3]]
            kind = "strike" if through else "underline"
            m = Mark(kind, bbox)
            m.words = _covered(bbox, blocks, img, 0.0 if through else line_h * 1.05, coverage=0.0)
            if m.words:
                marks.append(m)
                continue
        if (w_ * h_) / float(W * H) >= MIN_NOTE_AREA:
            marks.append(Mark("note", bbox))      # handwriting: only the LLM can read it
    marks.sort(key=lambda m: (round(m.bbox[1], 2), m.bbox[0]))
    return marks


def compare(reference: np.ndarray, new: np.ndarray, blocks: list[Block]) -> Diff:
    """The whole read side: align, subtract, classify. Costs no recognition of any kind."""
    aligned, cc = align(new, reference)
    if aligned is None:
        return Diff(aligned=False, reason="could not register the two views of the page")
    pen, hl = new_ink(reference, aligned)
    changed = float(((pen > 0) | (hl > 0)).mean())
    d = Diff(changed=round(changed, 4))
    if changed > MAX_CHANGE:
        d.reason = (f"{changed:.0%} of the page differs — that is a different page or a failed "
                    "alignment, not an annotation")
        return d
    d.marks = classify(pen, hl, blocks, reference)
    return d


# --- writing it back --------------------------------------------------------------------------------
def _phrase_re(words: list[str]) -> re.Pattern:
    """Match the phrase in the Markdown allowing for whatever punctuation and wrapping sit between."""
    return re.compile(r"(?<!\w)" + r"[^A-Za-z0-9]{0,4}".join(re.escape(w) for w in words) + r"(?!\w)")


def apply_marks(body: str, marks: list[Mark], notes_heading: str = "## Notes") -> tuple[str, list[str]]:
    """Apply marks to a page body. Returns (new body, what changed, in words).

    Emphasis is written around the phrase the mark covered, and a phrase that is already marked is
    left alone so that re-seeing the same page does not wrap it again and again. Handwritten notes go
    under their own heading at the end, where they cannot be confused with the printed page.
    """
    changes: list[str] = []
    for m in marks:
        if m.kind not in MARK_STYLE or not m.words:
            continue
        open_, close = MARK_STYLE[m.kind]
        pat = _phrase_re(m.words)
        hit = pat.search(body)
        if not hit:
            log.debug("mark %s: phrase %r not found in the body", m.kind, " ".join(m.words))
            continue
        before, after = body[:hit.start()], body[hit.end():]
        if before.endswith(open_) and after.startswith(close):
            continue                                    # already marked: leave it be
        body = before + open_ + hit.group(0) + close + after
        changes.append(m.describe())
    notes = [m for m in marks if m.kind == "note" and m.text.strip()]
    if notes:
        if notes_heading not in body:
            body = body.rstrip() + f"\n\n{notes_heading}\n"
        existing = body.split(notes_heading, 1)[1]
        for m in notes:
            line = f"\n- {m.text.strip()}"
            if m.text.strip() not in existing:
                body = body.rstrip() + line
                changes.append(m.describe())
    return body, changes


# --- the whole loop for one frame --------------------------------------------------------------------
@dataclass
class Sync:
    """What one look at a page came to."""
    page_id: str | None = None
    sequence: int | None = None
    diff: Diff | None = None
    changes: list[str] = field(default_factory=list)
    llm_calls: int = 0
    reason: str = ""

    def describe(self) -> str:
        if self.page_id is None:
            return self.reason or "no scanned page recognised"
        head = f"page {self.sequence:04d}" if self.sequence is not None else self.page_id[:8]
        if not self.changes:
            return f"{head}: unchanged" + (f" ({self.reason})" if self.reason else "")
        return f"{head}: " + "; ".join(self.changes)


def crop_mark(img: np.ndarray, bbox: list[float], pad: float = 0.02) -> np.ndarray:
    """The region a mark covers, with a margin, at full resolution — what a model is shown."""
    h, w = img.shape[:2]
    x0 = int(max(0.0, bbox[0] - pad) * w); y0 = int(max(0.0, bbox[1] - pad) * h)
    x1 = int(min(1.0, bbox[0] + bbox[2] + pad) * w); y1 = int(min(1.0, bbox[1] + bbox[3] + pad) * h)
    return img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
