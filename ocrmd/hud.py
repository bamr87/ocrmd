"""Preview overlay for `ocrmd scan`.

Kept separate from the capture loop: drawing is pure (frame in, frame out) so it can be tuned
without touching the gate logic, and the capture loop stays readable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import cv2, numpy as np

# BGR
GREEN, AMBER, RED, BLUE, WHITE, GREY = (90, 220, 120), (60, 190, 250), (80, 80, 240), (220, 160, 90), (245, 245, 245), (150, 150, 150)
PANEL = (28, 26, 24)
STATUS_COLOR = {"verified": GREEN, "flagged": AMBER, "suspect": RED, "illegible": RED,
                "blank": BLUE, "unverified": GREY, "error": RED, "working": WHITE}
GATE_COLOR = {"READY": GREEN, "moving": AMBER, "focusing": AMBER, "hand": RED, "same page": GREY,
              "blurry": RED, "page moved": AMBER, "settling": AMBER}
GATE_LABEL = {"hand": "HAND IN FRAME", "page moved": "PAGE MOVING"}


@dataclass
class ScanState:
    """Shared between the capture loop and the worker thread. Written by both, read by the HUD."""
    queued: int = 0
    working: str | None = None          # e.g. "page 3: llm"
    recent: list[tuple[str, str]] = field(default_factory=list)   # (label, status), newest last
    captured: int = 0
    advice: str | None = None
    error: str | None = None
    active: dict[str, str] = field(default_factory=dict)          # label -> stage, for the LLM pool
    degraded: bool = False                                        # LLM disabled for this session

    def set_active(self, label: str, stage: str | None) -> None:
        if stage is None:
            self.active.pop(label, None)
        else:
            self.active[label] = stage
        self.working = ", ".join(f"page {k}: {v}" for k, v in list(self.active.items())[:3]) or None

    def note(self, label: str, status: str) -> None:
        self.recent = (self.recent + [(label, status)])[-6:]


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.72) -> None:
    if w <= 0 or h <= 0:
        return
    y0, y1, x0, x1 = max(0, y), min(img.shape[0], y + h), max(0, x), min(img.shape[1], x + w)
    if y1 <= y0 or x1 <= x0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, PANEL), alpha, roi, 1 - alpha, 0, roi)


def _text(img, s, x, y, color=WHITE, scale=0.5, thick=1) -> None:
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _meter(img, x, y, w, value: float, color, label: str, detail: str = "") -> None:
    """Horizontal bar, value clamped to 0..1."""
    cv2.rectangle(img, (x, y), (x + w, y + 7), (70, 68, 66), -1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill:
        cv2.rectangle(img, (x, y), (x + fill, y + 7), color, -1)
    _text(img, label, x, y - 4, GREY, 0.38)
    if detail:
        _text(img, detail, x + w - 8 * len(detail), y - 4, GREY, 0.38)


def draw(frame: np.ndarray, *, gate: str, motion: float, still: int, stable_frames: int,
         settling_from: str = "", drift: float = 0.0,
         novelty: int, novelty_threshold: int, sharp: float, blur_threshold: float,
         view_desc: str, focus: str, adjusting: bool, resolution: tuple[int, int],
         page_quad: np.ndarray | None, state: ScanState, help_text: str,
         flash: float = 0.0, width: int = 1000, page_px: int | None = None,
         min_page_px: int = 900, clipped: list[str] | None = None,
         occlusion: str | None = None, focus_note: str = "") -> np.ndarray:
    """Return an annotated copy of `frame`, scaled to `width`."""
    scale = width / max(1, frame.shape[1])
    disp = cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h, w = disp.shape[:2]

    # detected page outline: the single most useful framing aid
    if page_quad is not None:
        cv2.polylines(disp, [(page_quad * scale).astype(np.int32)], True, GREEN if gate == "READY" else GREY, 2)

    if flash > 0:                                  # capture confirmation
        overlay = np.full_like(disp, (255, 255, 255))
        cv2.addWeighted(overlay, min(0.55, flash), disp, 1 - min(0.55, flash), 0, disp)

    # ---- top bar: gate state + meters
    _panel(disp, 0, 0, w, 78)
    colour = GATE_COLOR.get(gate, WHITE)
    label = GATE_LABEL.get(gate, gate.upper())
    _text(disp, label, 14, 30, colour, 0.82 if len(label) < 10 else 0.62, 2)
    bar_x, bar_w = 190, 200
    # "still" is the scene's settle progress, and its target grows after a page turn — showing the
    # bar against a fixed target would say "nearly there" through the whole of the longer wait.
    _meter(disp, bar_x, 20, bar_w, still / max(1, stable_frames), colour,
           f"settling after {settling_from}" if settling_from else "still",
           f"{still}/{stable_frames}" if settling_from else f"{motion:.1f}")
    _meter(disp, bar_x, 46, bar_w, min(1.0, novelty / max(1, novelty_threshold)), colour, "new page", str(novelty))
    _meter(disp, bar_x + bar_w + 40, 20, bar_w, min(1.0, sharp / max(1.0, blur_threshold * 3)),
           GREEN if sharp >= blur_threshold else RED, "sharpness", f"{sharp:.0f}")
    _text(disp, f"{resolution[0]}x{resolution[1]}  focus={focus}{'*' if adjusting else ''}"
          + (f"  {occlusion}" if occlusion else ""), bar_x + bar_w + 40, 60, GREY, 0.44)
    if focus_note and not focus_note.startswith("focus ok"):
        # the lens is the one thing the operator cannot see going wrong until the pages come out soft
        _text(disp, focus_note, bar_x + bar_w + 40, 76, AMBER if "soft" not in focus_note else RED, 0.44)
    _text(disp, view_desc, w - 8 * len(view_desc) - 14, 30, GREY, 0.44)
    _text(disp, f"captured {state.captured}", w - 130, 60, WHITE, 0.44)

    # px across the detected page: the single number that predicts whether OCR will work
    if clipped:
        _text(disp, f"CUT OFF: {' + '.join(clipped)}", 14, 62, RED, 0.52, 2)
        for edge in clipped:                       # mark the offending frame edge
            if edge == "left":   cv2.line(disp, (2, 0), (2, h), RED, 4)
            if edge == "right":  cv2.line(disp, (w - 3, 0), (w - 3, h), RED, 4)
            if edge == "top":    cv2.line(disp, (0, 2), (w, 2), RED, 4)
            if edge == "bottom": cv2.line(disp, (0, h - 3), (w, h - 3), RED, 4)
    elif page_px:
        ok = page_px >= min_page_px
        col = GREEN if ok else (AMBER if page_px >= min_page_px * 0.7 else RED)
        _text(disp, f"{page_px}px/page", 14, 62, col, 0.52, 2)
        if not ok:
            _text(disp, f"need {min_page_px}+ — move closer", 130, 62, col, 0.42)

    # ---- right panel: queue + recent pages
    if state.working or state.queued or state.recent:
        pw, px = 250, w - 250
        rows = len(state.recent)
        ph = 46 + rows * 20
        _panel(disp, px, 90, pw, ph)
        y = 112
        if state.working:
            _text(disp, state.working, px + 12, y, WHITE, 0.46); y += 20
        else:
            _text(disp, "idle", px + 12, y, GREY, 0.46); y += 20
        _text(disp, f"{state.queued} queued" + ("   LLM OFF" if state.degraded else ""), px + 12, y,
              RED if state.degraded else (AMBER if state.queued else GREY), 0.44)
        y += 22
        for label, status in state.recent:
            _text(disp, f"{label}", px + 12, y, GREY, 0.42)
            _text(disp, status, px + 100, y, STATUS_COLOR.get(status, WHITE), 0.42)
            y += 20

    # ---- bottom: advice / errors / help
    bottom = h - 52
    if state.error:
        _panel(disp, 0, bottom - 26, w, 26)
        _text(disp, state.error[:110], 14, bottom - 8, RED, 0.44)
    elif state.advice:
        _panel(disp, 0, bottom - 26, w, 26)
        _text(disp, state.advice[:110], 14, bottom - 8, AMBER, 0.44)
    _panel(disp, 0, h - 30, w, 30)
    _text(disp, help_text, 14, h - 10, GREY, 0.44)
    return disp
