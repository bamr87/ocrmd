"""Should this capture become a page at all?

Every capture used to become a page. On the store in this repository that produced 139 pages of which
the model called 88 of the 120 it looked at illegible — 73 % — and each of those cost an LLM call to
discover. The operator's complaint is the right one: several of them were never going to be readable
and should have been thrown away at the door.

**What does not work.** The obvious approach is to score the image. It was tried first, on the real
crops, against the model's own `legible` verdict, and it does not separate:

    signal                 legible median   illegible median   AUC
    page width px                 627              526        0.63
    x-height px                    10               20        0.46
    edge sharpness (p90)          109              104        0.58
    ink/paper contrast           0.55             0.49        0.57
    laplacian variance            156              133        0.62
    line-spacing gradient        0.75             0.60        0.64

Nothing above 0.64. A picture that looks fine by every cheap measure can still be unreadable, and the
corpus is uniformly marginal (median page 570 px wide) so there is little contrast to learn from.
Do not reintroduce an image-quality threshold here on the strength of intuition; it was measured.

**What does work** is to ask the OCR engines, which have already read the page by this point and are
free. On the same 120 pages:

    rule                    rejects   illegible   legible lost
    ocr words < 30               15    14 (93 %)       1 (3 %)
    ocr words < 60               21    20 (95 %)       1 (3 %)
    ocr words < 100              25    23 (92 %)       2 (6 %)
    mean confidence < 0.5        10     9 (90 %)       1 (3 %)
    mean confidence < 0.8        30    26 (87 %)      4 (12 %)

against a 73 % base rate. `words < 60 or confidence < 0.5` is the shipped default: on that corpus it
refuses 21 captures, 20 of which the model independently called illegible, and costs one legible page.

**Rejection is not deletion.** The raw frame stays on disk, the capture is recorded with its reason,
and no page, no page file and no LLM call are created. The invariant that nothing is removed
automatically holds — this only declines to *add*. A capture rejected wrongly is re-admitted with
`ocrmd ingest raw/<sha>.jpg --doc <slug>`, because file ingest never applies this test: a file the
operator handed over on purpose is not a guess the shutter made.

That escape hatch is load-bearing. The corpus has no genuinely empty pages to calibrate against, so
there is no measured way to tell "a blank page" from "a page we could not read" — both read as no
words. Rather than invent a threshold for that, a blank page is rejected like any other unreadable
one and costs the operator one command to put back.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

from .ocr import OcrResult

log = logging.getLogger("ocrmd.legibility")


@dataclass
class Verdict:
    """Whether a capture is worth making a page of, and why."""
    admit: bool
    reason: str = ""                            # why not, in words the operator can act on
    metrics: dict = field(default_factory=dict)

    def describe(self) -> str:
        return "admitted" if self.admit else self.reason


def assess(res: OcrResult | None, min_words: int = 60, min_confidence: float = 0.5) -> Verdict:
    """Judge a capture by what the local OCR engines made of it.

    `res` is None when no OCR engine is installed, in which case there is nothing to judge on and the
    capture is admitted: refusing pages on a machine that cannot read them would be silent data loss.
    """
    if res is None:
        return Verdict(True, metrics={"ocr": "unavailable"})
    words = len(res.words)
    conf = round(res.mean_confidence, 3)
    m = {"words": words, "confidence": conf, "lines": len(res.blocks)}
    if words < min_words:
        return Verdict(False, f"only {words} word(s) legible (needs {min_words})", m)
    if res.mean_confidence < min_confidence:
        return Verdict(False, f"OCR confidence {conf:.2f} (needs {min_confidence:.2f})", m)
    return Verdict(True, metrics=m)


def advice(reasons: list[str]) -> str | None:
    """One line for the operator after a run that rejected captures. Framing and focus are what these
    rejections are actually about — the two causes behind 73 % and 69 % of the illegible pages in the
    store — so say that rather than repeating the counts."""
    if not reasons:
        return None
    return (f"{len(reasons)} capture(s) refused as unreadable and left as raw frames — move the camera "
            "closer so one page fills the frame, and let the lens settle before turning the page. "
            "`ocrmd status` lists them; `ocrmd ingest raw/<sha>.jpg --doc <slug>` forces one in.")
