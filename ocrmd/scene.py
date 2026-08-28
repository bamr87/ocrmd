"""What the shutter is looking at, and whether it has stopped changing.

The auto-shutter used to be a memoryless chain of per-frame tests: still, focused, unobstructed, new,
sharp. Every one of them can be true *in the middle of a page turn*. A hand lets go, the page is
mid-fall, the pixels go quiet for a fraction of a second, the lens is still where it was, and the
frame is certainly novel — so the shutter fires on a page that is not where it will end up.

Measured on the real store in this repository: one two-minute session produced 21 captures, and
between consecutive ones the detected page outline moved by 100 to 700 px and its shape alternated
between a portrait page and a near-square region. Every one of those 21 became a page. That is what
this module exists to stop.

Two changes to the model:

**The page outline is a gate.** Pixel stillness says the *image* stopped changing; it goes quiet
between the frames of a slow movement. The outline of the detected sheet says the *page* stopped
moving, which is the thing that actually has to be true. On an unchanged scene the detector is very
steady: re-detecting 30 real captures from this store under sensor noise (450 draws) moved the outline
by less than 0.001 of the frame 95 % of the time, and only 0.89 % of draws moved it past 0.03 at all.
So a tolerance of 0.03 sits ~30x above the detector's own noise and far below a page turn, which
measured 0.02-0.36 between consecutive captures of a real session.

Those few unstable draws are worth knowing about: *all* of them came from one frame of the thirty.
The detector is not randomly jittery — particular scenes are genuinely ambiguous about where the
sheet is, and on those the crop would be wrong however still the desk is. Holding the shutter there
is the right answer rather than a false negative, which is why the test is the strict one ("the last
`window` detections all agree") with no tolerance for an outlier.

**The scene remembers what just happened.** A settled desk that twitches needs only to go quiet
again; a desk where a hand just moved a page needs to prove it has finished. So the quiet period
required before the shutter can fire depends on the *kind* of disturbance last seen: `stable_frames`
after ordinary noise, `settle_frames` after a hand or a moved page. That single distinction is the
difference between capturing a page turn and capturing the page it turned to.

Kept free of I/O — no camera, no OpenCV window — so the whole decision is unit-testable frame by
frame, which is how the page-turn sequences in `tests/test_scene.py` are written.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass

# Blocking reasons, most urgent first. The order is the diagnosis the operator sees, so it has to be
# the *cause* rather than a symptom: a moving scene says nothing about focus, a hunting lens makes
# sharpness meaningless, and a hand in frame is what makes a frame novel in the first place.
BLOCKERS = ("moving", "focusing", "hand", "page moved", "settling")
# Disturbances that mean the scene was *handled* rather than merely noisy. These demand the long
# settle: something moved the book, and where it ends up is not where it is halfway through.
MAJOR = ("hand", "page moved")


@dataclass(frozen=True)
class Outline:
    """The detected page as fractions of the frame, so one tolerance works at every resolution."""
    cx: float
    cy: float
    w: float
    h: float

    @classmethod
    def from_rect(cls, rect, shape) -> "Outline":
        """From an OpenCV rotated rect ((cx, cy), (w, h), angle) and the frame's shape."""
        (cx, cy), (w, h), _ang = rect
        fh, fw = shape[:2]
        # The two sides are an unordered pair scaled by the *same* number. minAreaRect describes one
        # sheet as either (w, h, -90) or (h, w, 0), so a gate that told those apart would never settle
        # on a page that never moved — and dividing the sides by the frame's width and height
        # separately reintroduces exactly that, because the swap then changes both values.
        side = max(fw, fh, 1)
        lo, hi = sorted((abs(w) / side, abs(h) / side))
        return cls(cx / max(fw, 1), cy / max(fh, 1), lo, hi)

    def drift(self, other: "Outline") -> float:
        """How far this outline is from another, as a fraction of the frame: the worst of centre
        movement and size change. One number, because any of them moving means the page moved."""
        return max(abs(self.cx - other.cx), abs(self.cy - other.cy),
                   abs(self.w - other.w), abs(self.h - other.h))


class OutlineTracker:
    """Has the detected page held still for the last `window` detections?

    Fed only when the outline is actually re-detected (the capture loop does that a few times a
    second, not every frame), so `window` counts detections rather than frames.
    """

    def __init__(self, window: int = 3, tolerance: float = 0.03):
        self.window, self.tolerance = max(1, window), tolerance
        self.seen: deque[Outline] = deque(maxlen=max(1, window))

    def reset(self) -> None:
        self.seen.clear()

    def update(self, outline: Outline | None) -> None:
        """`None` — no page found — is recorded rather than discarded. A detector that consistently
        finds nothing is a *stable* observation about a rig where the page fills the frame edge to
        edge; one that flickers between a page and nothing is an ambiguous scene, and those are
        exactly the frames worth not capturing."""
        self.seen.append(outline)

    @staticmethod
    def _drift(a: Outline | None, b: Outline | None) -> float:
        if a is None and b is None:
            return 0.0
        if a is None or b is None:
            return 1.0          # a page appeared or vanished: the largest change there is
        return a.drift(b)

    @property
    def drift(self) -> float:
        """Worst disagreement in the window; 0 when there is nothing to compare yet."""
        if len(self.seen) < 2:
            return 0.0
        return max(self._drift(a, b) for a in self.seen for b in self.seen)

    @property
    def settled(self) -> bool:
        """A full window of detections that all agree.

        Empty is settled: a caller that never reports an outline has said nothing about the page, and
        absence of evidence must not hold the shutter shut forever. But once outlines *are* being
        reported, a single one is not yet evidence of stillness — the frame after a page lands looks
        exactly like the frame before it starts to move.
        """
        return not self.seen or (len(self.seen) >= self.window and self.drift <= self.tolerance)


@dataclass
class Verdict:
    """Why the shutter is or is not firing, this frame."""
    blocker: str = ""                 # "" when nothing is holding it; otherwise one of BLOCKERS
    quiet: int = 0                    # consecutive undisturbed frames
    needed: int = 0                   # how many this scene has to show before it may fire
    disturbance: str = ""             # what it is still settling from ("" once it has settled)
    drift: float = 0.0                # worst page-outline disagreement in the window

    @property
    def settled(self) -> bool:
        return not self.blocker

    def describe(self) -> str:
        if self.settled:
            return "settled"
        if self.blocker == "settling":
            return f"settling after {self.disturbance} ({self.quiet}/{self.needed})"
        return self.blocker


class Scene:
    """The shutter's model of the desk: disturbed, settling, or settled.

    `observe` is the whole decision and takes only booleans, so the caller owns every measurement and
    this stays testable. It answers one question — has the scene finished changing? — and leaves
    "is this page new" and "is it sharp enough" to the caller, which are properties of the *page*
    rather than of the scene.
    """

    def __init__(self, stable_frames: int = 8, settle_frames: int = 24, window: int = 3,
                 tolerance: float = 0.03):
        self.stable_frames, self.settle_frames = max(1, stable_frames), max(1, settle_frames)
        self.outlines = OutlineTracker(window, tolerance)
        self.quiet = 0
        self.disturbance = ""

    def reset(self) -> None:
        """Forget the scene: a new camera view, a reopened device, a fresh run."""
        self.outlines.reset()
        self.quiet, self.disturbance = 0, ""

    def note_outline(self, outline: Outline | None) -> None:
        """Feed a *freshly detected* page outline (None when no page was found)."""
        self.outlines.update(outline)

    @property
    def needed(self) -> int:
        """Frames of quiet this scene must show. A page turn has to prove it has finished; ordinary
        sensor noise only has to stop."""
        return self.settle_frames if self.disturbance in MAJOR else self.stable_frames

    def observe(self, *, moving: bool, adjusting: bool, occluded: bool) -> Verdict:
        """Advance one frame and report what, if anything, is holding the shutter."""
        blocker = ("moving" if moving else
                   "focusing" if adjusting else
                   "hand" if occluded else
                   "page moved" if not self.outlines.settled else "")
        if blocker:
            self.quiet = 0
            # Remember the *worst* thing seen since the scene was last settled. A page turn that ends
            # with a moment of ordinary jitter is still a page turn, and must not be let off with the
            # short wait because the last frame before the quiet happened to be merely "moving".
            if not self.disturbance or (blocker in MAJOR and self.disturbance not in MAJOR):
                self.disturbance = blocker
            return Verdict(blocker, self.quiet, self.needed, self.disturbance, self.outlines.drift)
        self.quiet += 1
        needed = self.needed
        if self.quiet < needed:
            return Verdict("settling", self.quiet, needed, self.disturbance, self.outlines.drift)
        self.disturbance = ""
        return Verdict("", self.quiet, needed, "", self.outlines.drift)

    def captured(self) -> None:
        """A frame was taken. The next one has to earn its own quiet period, so a single settled
        moment cannot produce a burst of captures of a scene that is still being handled."""
        self.quiet = 0
