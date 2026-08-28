"""Focus bracketing resolved at the line level: read the page again at a different focus.

The capture side (`capture.focus_bracket`) shoots the same still page at several lens positions and
keeps the losers as *variants*. This module decides what they are worth.

The premise is that whole-page sharpness — the number the auto-shutter and autofocus both optimise —
is an average, and a page is not uniformly in focus. A curled leaf, a book that does not lie flat, a
camera a few degrees off square: the frame that wins on average can be the worse frame for the top
third of the page. So instead of choosing one frame, the primary's *low-confidence lines only* are
looked up in every variant, and a line another focus read markedly better replaces it.

The rules deliberately mirror `ocr.merge`, which does the same thing across engines rather than
across focus, and for the same reason: a swap is only allowed when the challenger is confident, of
similar length, and beating a reading the primary was not sure of anyway. Two differences, both
because variant crops are cropped independently and so carry a little more geometric jitter than two
engines reading one crop:

- a line only a variant saw must reach `ocr.CONFIRMED` (0.9), not `ocr.FILL_CONFIDENCE` (0.85);
- the primary's box is kept after a swap, so reading order can never be disturbed by the swap itself;
- one variant line can be spent on only one of the page's lines.

And one rule `merge` does not have: a *longer* reading may replace a short one when it is certain. Two
engines reading one image disagree word by word; two focus settings do not. An out-of-focus band comes
back as a fragment — measured on a half-blurred page, "milies" where the page reads "is so well fixed
in the minds of the surrounding families" — so a symmetric length window would reject exactly the
captures this feature exists to rescue. Shorter readings are still refused: that is text being lost.

What is left over is the point of the exercise. A line no focus could read stays in the result at its
own low confidence *and* is recorded as an `Unresolved` region with every candidate reading and where
it came from. That analysis, and the variant images themselves, go to the LLM — which is the only
stage that can look at three exposures of the same line and say "I still cannot read this", instead
of the pipeline silently picking the highest number.
"""
from __future__ import annotations
import logging
from pydantic import BaseModel

from .ocr import Block, OcrResult, CONFIRMED, _same_line

log = logging.getLogger("ocrmd.multifocus")

LENGTH_RATIO = (0.7, 1.4)      # a challenger reading of a line must be about as long as the primary's


class Candidate(BaseModel):
    source: str                # "focus" for the captured frame, or the variant's label
    text: str
    confidence: float


class Unresolved(BaseModel):
    """A line no focus in the bracket read confidently. Carried to the LLM, never resolved by voting."""
    bbox: list[float]          # x, y, w, h normalised on the primary crop
    candidates: list[Candidate]

    @property
    def best(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.confidence)


class FocusReading(BaseModel):
    """One bracketed exposure as the OCR stage saw it. The lens position it was shot at stays on the
    capture side (the scan log and the HUD) — by the time a page is read, what matters about a variant
    is what it read, not where the lens was."""
    label: str                 # "v1", "v2": the variant's suffix in raw/ and processed/
    mean_confidence: float = 0.0
    lines: int = 0


class MultiFocus(BaseModel):
    """What the bracket bought for this page. Stored as `ocr/<page>.focus.json`."""
    variants: list[FocusReading] = []
    improved: int = 0                    # lines replaced by a better-focused reading
    filled: int = 0                      # lines only another focus saw at all
    unresolved: list[Unresolved] = []

    @property
    def useful(self) -> bool:
        return bool(self.improved or self.filled or self.unresolved)

    def summary(self) -> str:
        return (f"{len(self.variants)} focus variant(s): {self.improved} line(s) improved, "
                f"{self.filled} filled, {len(self.unresolved)} unresolved")


def _words_by_block(res: OcrResult) -> list[list[tuple[str, float]]]:
    """Split `res.words` back into one list per block.

    `ocr.merge` emits the flat list in block order with exactly one entry per whitespace token, so the
    split is exact — but a hand-written or older artifact need not honour that, and getting it wrong
    would silently reassign confidences to the wrong words. Check, and fall back to the block's own
    confidence when the counts do not line up.
    """
    counts = [len(b.text.split()) for b in res.blocks]
    if sum(counts) != len(res.words):
        return [[(w, b.confidence) for w in b.text.split()] for b in res.blocks]
    out, i = [], 0
    for n in counts:
        out.append(list(res.words[i:i + n])); i += n
    return out


def _line_reading(block: Block, other: list[Block]) -> tuple[list[int], str, float, str] | None:
    """How another focus read the line at `block`: its overlapping blocks joined left to right, with
    their indices so a block can never be spent on two different lines.

    The confidence is the *minimum* over those blocks, not the mean: a line assembled from a confident
    fragment and a doubtful one is only as trustworthy as the doubtful part, and taking the mean is
    how a half-read line talks its way past the swap threshold.
    """
    same = sorted(((j, o) for j, o in enumerate(other) if _same_line(block, o)), key=lambda t: t[1].bbox[0])
    if not same:
        return None
    return ([j for j, _ in same], " ".join(o.text for _, o in same),
            min(o.confidence for _, o in same), same[0][1].engine)


def _acceptable(primary_words: int, text: str, conf: float) -> bool:
    """May this reading stand in for a line the page read badly?

    Two shapes qualify. A reading of *about the same length* is the ordinary case — the same line,
    read differently — and is the rule `ocr.merge` uses between engines. A reading that is
    substantially *longer* qualifies too, but only if it is certain, because that is what an
    out-of-focus band actually produces: measured on a half-blurred 520-px page, every doubtful line
    came back as a fragment ("milies" where the page reads "is so well fixed in the minds of the
    surrounding families"), not as a word-for-word misreading. A symmetric length window rejected all
    of them — the exact captures a focus bracket exists to rescue.

    A *shorter* reading never qualifies, whatever its confidence: that is text being dropped.
    """
    ratio = len(text.split()) / max(1, primary_words)
    return LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1] or (ratio > LENGTH_RATIO[1] and conf >= CONFIRMED)


def resolve(primary: OcrResult, variants: list[tuple[FocusReading, OcrResult]], trust: float = 0.90,
            margin: float = 0.10) -> tuple[OcrResult, MultiFocus]:
    """Re-read the primary's doubtful lines in every variant. Returns (result, analysis).

    `trust` is `Settings.ocr_trust` — the confidence at which reconcile is willing to let OCR overrule
    the model, and therefore the right line between "read" and "not read". `margin` is how much better
    another focus has to be before its reading replaces one this page already has.
    """
    mf = MultiFocus(variants=[fr for fr, _ in variants])
    if not variants:
        return primary, mf
    blocks = list(primary.blocks)
    words = _words_by_block(primary)
    used: dict[str, set[int]] = {fr.label: set() for fr, _ in variants}      # blocks any line overlapped
    spent: dict[str, set[int]] = {fr.label: set() for fr, _ in variants}     # blocks a swap has consumed
    for i, b in enumerate(blocks):
        readings: list[tuple[FocusReading, list[int], str, float, str]] = []
        for fr, res in variants:
            r = _line_reading(b, res.blocks)
            if r is not None:
                used[fr.label].update(r[0])
                readings.append((fr, *r))
        if b.confidence >= trust or not readings:
            continue
        n = len(b.text.split())
        # A variant line already given to an earlier (higher) page line may not be handed out again:
        # variant crops carry a little geometric jitter, so one variant line can overlap two of the
        # page's, and without this the same text would be written twice.
        fit = [r for r in readings if _acceptable(n, r[2], r[3]) and not (set(r[1]) & spent[r[0].label])]
        best = max(fit, key=lambda r: r[3]) if fit else None
        if best is not None and best[3] >= b.confidence + margin:
            fr, idx, text, conf, eng = best
            blocks[i] = b.model_copy(update={"text": text, "confidence": conf, "engine": eng, "focus": fr.label})
            words[i] = [(w, conf) for w in text.split()]
            spent[fr.label].update(idx)
            mf.improved += 1
            log.debug("focus %s improved a line: %.2f -> %.2f (%r -> %r)", fr.label, b.confidence, conf,
                      b.text[:40], text[:40])
        final = blocks[i]
        if final.confidence < trust:
            cands = [Candidate(source="focus", text=b.text, confidence=round(b.confidence, 3))]
            cands += [Candidate(source=fr.label, text=text, confidence=round(conf, 3))
                      for fr, _idx, text, conf, _eng in readings]
            mf.unresolved.append(Unresolved(bbox=[round(v, 4) for v in final.bbox], candidates=cands))
    # lines the captured frame missed entirely: only taken from a focus that is sure of them
    for fr, res in variants:
        for j, o in enumerate(res.blocks):
            if j in used[fr.label] or o.confidence < CONFIRMED or len(o.text.split()) < 2:
                continue
            if any(_same_line(q, o) for q in blocks):
                continue
            blocks.append(o.model_copy(update={"focus": fr.label, "agreed": None}))
            words.append([(w, o.confidence) for w in o.text.split()])
            used[fr.label].add(j)
            mf.filled += 1
    order = sorted(range(len(blocks)), key=lambda k: (round(blocks[k].bbox[1], 2), blocks[k].bbox[0]))
    blocks, words = [blocks[k] for k in order], [words[k] for k in order]
    conf = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0
    out = primary.model_copy(update={
        "blocks": blocks, "text": "\n".join(b.text for b in blocks), "mean_confidence": conf,
        "words": [w for ws in words for w in ws]})
    return out, mf


# --- what the LLM is told -------------------------------------------------------------------------
def variant_ranking(mf: MultiFocus) -> list[str]:
    """Variant labels worth sending to the model, best first: the ones holding the most confident
    candidate for an unresolved region, then the rest in bracket order."""
    score: dict[str, float] = {fr.label: 0.0 for fr in mf.variants}
    for u in mf.unresolved:
        best = u.best
        if best.source in score:
            score[best.source] += best.confidence
    return sorted(score, key=lambda label: (-score[label], [fr.label for fr in mf.variants].index(label)))


def describe(mf: MultiFocus, images: list[str], limit: int = 15) -> str:
    """The prompt fragment: which extra images are attached, and what could not be read from any of them.

    Only the regions matter to the model — a confidence table for lines that came out fine would be
    noise. Positions are given as fractions of the page so they can be found in the image, and the
    candidates are labelled as machine guesses, because they are exactly the kind of plausible wrong
    text the anti-fabrication rules exist to keep out of the transcription.
    """
    if not images and not mf.unresolved:
        return ""
    lines = []
    if images:
        lines.append(f"Attached after the page image: {len(images)} more exposure(s) of the SAME page taken at "
                     f"different focus settings ({', '.join(images)}), in that order. They are alternative views of "
                     "one page, not further pages — never transcribe them as separate pages, and never merge their "
                     "content as if it were extra text. Use them only to read what the first image does not resolve.")
    if mf.unresolved:
        lines.append(f"Local OCR could not read {len(mf.unresolved)} region(s) confidently at any focus. "
                     "Positions are fractions of the page (x, y from the top-left). The readings are "
                     "low-confidence machine guesses, NOT ground truth — take them as hints about where to "
                     "look. If you can read the region, transcribe what you see even if it matches none of "
                     "them; if you cannot, leave it out and say so in `uncertain` rather than picking one.")
        for u in mf.unresolved[:limit]:
            cands = "; ".join(f'{c.source}: "{c.text[:80]}" ({c.confidence:.2f})' for c in u.candidates)
            lines.append(f"  - x={u.bbox[0]:.2f} y={u.bbox[1]:.2f} w={u.bbox[2]:.2f} — {cands}")
        if len(mf.unresolved) > limit:
            lines.append(f"  - ... and {len(mf.unresolved) - limit} more")
    return "\n".join(lines)
