"""Hand-in-frame detection for the auto-shutter.

The gates in `capture.py` ask whether the scene is still, new and sharp. A hand holding a textbook
open passes all three: it stops moving, it changes the frame hash, and it is in focus. The result was
captures with a hand across the text. This module answers a fourth question — is something covering
paper that was visible a moment ago? — by comparing the current *paper mask* against a reference:

    missing = reference & ~current      paper that has disappeared (a hand, a pen, a curled page)
    added   = current & ~reference      paper that has appeared (the book was moved, a page unfolded)

A hand removes paper and adds none; moving the book does both. That distinction matters on a fixed
rig: a book nudged into a new position must re-baseline immediately, while a hand must hold the
shutter for as long as it stays. Skin colour (YCrCb) inside the missing region only *labels* the
intrusion; it is not the detector — a wooden desk is skin-toned in YCrCb, which is why measuring
skin inside the page outline (the first attempt) flagged half of the clean captures.

Masks are compared at analysis scale after a morphological clean-up wide enough to swallow text, so
two different pages of the same book produce near-identical masks and only real intrusions differ.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2, numpy as np


@dataclass
class Occlusion:
    missing: float       # fraction of the reference paper area no longer visible as paper
    added: float         # fraction of the reference paper area that is *new* paper
    skin: float          # fraction of the missing region that is skin-toned
    moved: bool          # the paper changed shape rather than being covered
    occluded: bool

    def describe(self) -> str:
        return f"cover={self.missing:.2f} skin={self.skin:.2f}" + (" moved" if self.moved else "")


def _small(bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = width / max(1, w)
    return cv2.resize(bgr, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else bgr


def paper_mask(bgr: np.ndarray, width: int = 320) -> np.ndarray:
    """Bright, unsaturated sheet(s) in the frame at analysis scale, with text and thin shadows removed.
    Skin is excluded explicitly: under a desk lamp a hand can be as bright as paper in grey."""
    small = _small(bgr, width)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thr[skin_mask(small, width)] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thr = cv2.morphologyEx(cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return thr > 0


def skin_mask(bgr: np.ndarray, width: int = 320) -> np.ndarray:
    small = _small(bgr, width)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    return (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y >= 40)


def compare(reference: np.ndarray | None, current: np.ndarray, skin: np.ndarray | None = None,
            cover_threshold: float = 0.06) -> Occlusion:
    """Occlusion of `current` relative to `reference` (both boolean paper masks of the same shape)."""
    if reference is None or reference.shape != current.shape:
        return Occlusion(0.0, 0.0, 0.0, False, False)
    ref_area = int(reference.sum())
    if ref_area < 0.05 * reference.size:          # no meaningful sheet in the reference
        return Occlusion(0.0, 0.0, 0.0, False, False)
    missing = reference & ~current
    added = current & ~reference
    mf = float(missing.sum()) / ref_area
    af = float(added.sum()) / ref_area
    sk = float((missing & skin).sum()) / max(1, int(missing.sum())) if skin is not None else 0.0
    # paper appeared as well as disappeared: the book moved. Unless what disappeared is skin-toned
    # *and* little appeared — a hand arriving while a page unfolds looks like a move otherwise.
    moved = af > 0.02 and af >= 0.5 * mf and not (sk >= 0.35 and af < mf)
    occluded = (not moved) and mf >= cover_threshold
    return Occlusion(round(mf, 4), round(af, 4), round(sk, 4), moved, occluded)


class Tracker:
    """Keeps the reference paper mask for a session.

    The reference follows the scene whenever nothing is covering it, never shrinks to what a hand
    leaves uncovered, and re-baselines when the paper changed shape (book moved) or when a
    non-skin intrusion has sat still long enough to be furniture rather than a hand.
    """

    def __init__(self, cover_threshold: float = 0.06, skin_threshold: float = 0.3, rebaseline_after: int = 30):
        self.cover_threshold, self.skin_threshold, self.rebaseline_after = cover_threshold, skin_threshold, rebaseline_after
        self.reference: np.ndarray | None = None
        self.stuck = 0

    def reset(self) -> None:
        self.reference, self.stuck = None, 0

    def update(self, frame: np.ndarray, width: int = 320) -> Occlusion:
        cur = paper_mask(frame, width)
        occ = compare(self.reference, cur, skin_mask(frame, width), self.cover_threshold)
        if self.reference is None or occ.moved:
            self.reference, self.stuck = cur, 0
        elif not occ.occluded:
            self.stuck = 0
            if cur.sum() >= 0.9 * self.reference.sum():   # same sheet, possibly a new page: follow it
                self.reference = cur
        else:
            self.stuck += 1
            # Not going away: accept the scene. A non-skin intrusion (bookmark, shadow, a book moved
            # over a dark desk) re-baselines quickly; something skin-toned waits three times longer,
            # because a wooden desk is skin-toned too and a stuck gate is worse than a late capture.
            limit = self.rebaseline_after * (3 if occ.skin >= self.skin_threshold else 1)
            if self.stuck >= limit:
                self.reference, self.stuck = cur, 0
                return compare(None, cur)
        return occ
