"""Keeping the lens on the page, on cameras that will not say where the lens is.

Everything ocrmd knew about focus came from AVFoundation: `isAdjustingFocus` gated the shutter, and
`lensPosition` drove the focus bracket. Measured on this repository's own rig — a Logitech C525, the
camera the tool was built on — both are useless:

- `isAdjustingFocus` was **false on all 599 frames** of a session that included two forced mode
  changes. The focus gate has never fired on this camera, and the HUD's `FOCUSING` state has never
  been shown.
- `lensPosition` reads 0.0 and never changes, in `continuous` or `locked`. That is also why the focus
  bracket reports "unavailable" here.

And the failure that matters is real: the same book, the same lamp, minutes apart, measured page
sharpness of **18 and then 1030**. In the bad state the camera reported `focus=continuous` and sat
there for the twenty seconds it was watched, through a `locked`→`continuous` toggle. A scan started
in that state would have photographed the whole book out of focus, because the shutter's sharpness
floor is relative to the best *this session* has shown and decays (`sharp_decay`) — a session that
starts soft quietly lowers its own bar. In the store that shows up as crop blur of 149 median for
pages that ended up merely flagged, 101 for `suspect` and 66 for `illegible`.

So focus is watched in the image instead, where every camera can be watched the same way:

- **Is the lens moving?** The spread of page sharpness over a short window. On a still, in-focus
  scene the rolling 12-frame spread measured p50 0.012, p99 0.026, max 0.027 (598 frames), so
  `HUNT_BAND = 0.06` is a bit over twice the worst quiet frame and anything past it is real movement.
- **Is the page soft?** Sharpness against the best this *rig* has ever produced, remembered across
  sessions next to the camera's view (`.ocrmd-camera.json`). Over 20 s of a working rig the ratio
  never fell below 0.939; the broken state was 0.017. `SOFT_RATIO = 0.45` sits between them with room
  for a genuinely low-contrast page.
- **Then do something about it.** `Watch.observe` asks for a retrigger (drop the focus mode and put
  it back, which is all a camera with no lens control offers) and escalates to reopening the stream.
  A retrigger is cheap and harmless, which is the point: a mostly blank page can trip the softness
  test, and the worst that costs is one re-hunt.

The remembered best is allowed to fall. A rig that is moved further from the page is genuinely less
sharp than it used to be, and a memory that refuses to believe that would block every capture
forever; after `attempts_before_accepting` recoveries have failed to bring it back, the watch accepts
the new normal, says so, and carries on.
"""
from __future__ import annotations
import json, logging, statistics, time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .camview import _path as _camera_file

log = logging.getLogger("ocrmd.focus")

WINDOW = 12                  # frames the spread is measured over (~0.4 s at 30 fps)
HUNT_BAND = 0.06             # relative spread above which the lens is moving, not the sensor noising
SOFT_RATIO = 0.45            # of the rig's remembered best: below this the page is out of focus
SOFT_FRAMES = 45             # ...for this many consecutive frames before anything is done (~1.5 s)
RECOVERY_GAP = 6.0           # seconds between attempts, so a retrigger gets time to work
ATTEMPTS_BEFORE_REOPEN = 2   # retriggers before the stream itself is restarted
ATTEMPTS_BEFORE_ACCEPTING = 4  # ...and before the remembered best is believed to be out of date


@dataclass
class Verdict:
    """What the sharpness trace says about the lens right now."""
    hunting: bool = False            # sharpness still moving: the lens has not settled
    soft: bool = False               # far below what this rig can do, and holding there
    ratio: float = 1.0               # sharpness / remembered best
    spread: float = 0.0              # relative spread over the window
    action: str | None = None        # None | "retrigger" | "reopen" | "accept"
    note: str = ""

    def describe(self) -> str:
        if self.action:
            return f"{self.action}: {self.note}"
        if self.hunting:
            return f"focusing (spread {self.spread:.3f})"
        if self.soft:
            return f"soft ({self.ratio:.0%} of this rig's best)"
        return f"focus ok ({self.ratio:.0%})"


@dataclass
class Watch:
    """Per-session focus state: the sharpness history, the rig's memory, and what to do about it."""
    best: float = 0.0                        # the rig's remembered best page sharpness
    soft_ratio: float = SOFT_RATIO
    soft_frames: int = SOFT_FRAMES
    hunt_band: float = HUNT_BAND
    window: int = WINDOW
    recovery_gap: float = RECOVERY_GAP
    attempts_before_reopen: int = ATTEMPTS_BEFORE_REOPEN
    attempts_before_accepting: int = ATTEMPTS_BEFORE_ACCEPTING
    samples: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    session_best: float = 0.0
    soft_run: int = 0
    attempts: int = 0
    last_attempt: float = float("-inf")   # so the first recovery is not made to wait out a gap
                                          # that has not started yet
    accepted: bool = False                   # the remembered best has been written down to this rig

    def __post_init__(self):
        self.samples = deque(maxlen=self.window)

    def observe(self, sharp: float, occluded: bool = False, adjusting: bool | None = None,
                now: float | None = None) -> Verdict:
        """One frame's page sharpness. `adjusting` is the camera's own opinion when it has one — kept
        as an *additional* reason to wait, never as the only one, because most cameras never say so."""
        now = time.monotonic() if now is None else now
        if occluded:                          # a hand is sharper than paper and is not the lens moving
            self.samples.clear()
            self.soft_run = 0
            return Verdict(hunting=bool(adjusting), ratio=self._ratio(sharp), note="occluded")
        self.samples.append(float(sharp))
        self.session_best = max(self.session_best, float(sharp))
        spread = self.spread()
        hunting = bool(adjusting) or (spread > self.hunt_band if len(self.samples) >= self.window else False)
        ratio = self._ratio(sharp)
        v = Verdict(hunting=hunting, ratio=ratio, spread=spread)
        if self.best <= 0:                    # nothing to compare against yet: this rig is the baseline
            self.best = max(self.best, float(sharp))
            return v
        if hunting or ratio >= self.soft_ratio:
            self.soft_run = 0
            if sharp > self.best:
                self.best = float(sharp)
            return v
        self.soft_run += 1
        if self.soft_run < self.soft_frames:
            return v
        v.soft = True
        if now - self.last_attempt < self.recovery_gap:
            return v
        self.attempts += 1
        self.last_attempt = now
        if self.attempts <= self.attempts_before_reopen:
            v.action, v.note = "retrigger", (f"page sharpness is {ratio:.0%} of this rig's best "
                                             f"({sharp:.0f} against {self.best:.0f}); refocusing")
        elif self.attempts <= self.attempts_before_accepting:
            v.action, v.note = "reopen", (f"still soft after {self.attempts - 1} refocus attempt(s); "
                                          "restarting the camera stream")
        else:
            v.action = "accept"
            v.note = (f"refocusing did not help; taking {max(self.session_best, sharp):.0f} as what this "
                      f"rig now manages (it reached {self.best:.0f} before). "
                      "Check the focus, the distance and the light.")
            self.best, self.accepted = max(self.session_best, sharp), True
            self.soft_run = 0
        return v

    def acted(self) -> None:
        """Called after a recovery action: give the lens a clean window to be judged on."""
        self.samples.clear()
        self.soft_run = 0

    def spread(self) -> float:
        if len(self.samples) < max(3, self.window // 2):
            return 0.0
        mid = statistics.median(self.samples)
        return (max(self.samples) - min(self.samples)) / max(1.0, mid)

    def _ratio(self, sharp: float) -> float:
        return float(sharp) / self.best if self.best > 0 else 1.0

    def floor(self, absolute: float, session_best: float, ratio: float) -> float:
        """The sharpness floor the shutter should use: the session's own bar, but never far below what
        this rig is known to manage. Without the second term a session that starts out of focus
        photographs the whole book that way and never notices."""
        anchored = self.best * ratio if self.best > 0 and not self.accepted else 0.0
        return max(absolute, session_best * ratio, anchored)


# --- the rig's memory, beside the camera's view ---------------------------------------------------
_KEY = "_focus"


def load_best(root: Path, camera: str) -> float:
    try:
        data = json.loads(_camera_file(root).read_text())
        return float((data.get(_KEY) or {}).get(camera, {}).get("best_sharp") or 0.0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0


def save_best(root: Path, camera: str, best: float) -> None:
    """Kept next to the camera view, under a key no camera name can collide with."""
    p = _camera_file(root)
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        data = {}
    if not isinstance(data.get(_KEY), dict):
        data[_KEY] = {}
    data[_KEY][camera] = {"best_sharp": round(float(best), 1), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)
