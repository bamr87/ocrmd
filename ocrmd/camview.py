"""Camera view: rotation + digital zoom/pan applied identically to preview and captured frames.
Persisted per camera name so a rig set up once stays set up."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2, numpy as np

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class CameraView:
    rotation: int = 0       # degrees clockwise, 0/90/180/270
    zoom: float = 1.0       # 1.0 = full frame; 2.0 = centre half, etc.
    cx: float = 0.5         # zoom centre, fraction of width
    cy: float = 0.5         # zoom centre, fraction of height

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.zoom > 1.0:
            h, w = frame.shape[:2]
            cw, ch = int(w / self.zoom), int(h / self.zoom)
            x0 = int(np.clip(self.cx * w - cw / 2, 0, w - cw))
            y0 = int(np.clip(self.cy * h - ch / 2, 0, h - ch))
            frame = frame[y0:y0 + ch, x0:x0 + cw]
        code = ROTATIONS[self.rotation % 360]
        return cv2.rotate(frame, code) if code is not None else frame

    # --- interactive edits (return True if changed) -----------------------
    def rotate(self, delta: int = 90) -> None:
        self.rotation = (self.rotation + delta) % 360

    def zoom_by(self, factor: float) -> None:
        self.zoom = float(np.clip(round(self.zoom * factor, 3), 1.0, 6.0))
        if self.zoom == 1.0:
            self.cx = self.cy = 0.5

    def pan(self, dx: float, dy: float) -> None:
        """Pan in *view* coordinates; rotation maps it back to sensor coordinates."""
        if self.zoom <= 1.0:
            return
        r = self.rotation % 360
        sx, sy = {0: (dx, dy), 90: (-dy, dx), 180: (-dx, -dy), 270: (dy, -dx)}[r]
        half_w, half_h = 0.5 / self.zoom, 0.5 / self.zoom
        self.cx = float(np.clip(self.cx + sx, half_w, 1 - half_w))
        self.cy = float(np.clip(self.cy + sy, half_h, 1 - half_h))

    def reset(self) -> None:
        self.rotation, self.zoom, self.cx, self.cy = 0, 1.0, 0.5, 0.5

    def describe(self) -> str:
        return f"rot={self.rotation} zoom={self.zoom:.2f}x centre=({self.cx:.2f},{self.cy:.2f})"


# key handling is separate from the OpenCV loop so it can be unit-tested
KEYS = {
    "r": lambda v: v.rotate(90), "R": lambda v: v.rotate(-90),
    "+": lambda v: v.zoom_by(1.25), "=": lambda v: v.zoom_by(1.25), "-": lambda v: v.zoom_by(1 / 1.25),
    "w": lambda v: v.pan(0, -0.05), "s": lambda v: v.pan(0, 0.05),
    "a": lambda v: v.pan(-0.05, 0), "d": lambda v: v.pan(0.05, 0),
    "0": lambda v: v.reset(),
}
ARROWS = {0: "w", 1: "s", 2: "a", 3: "d"}  # OpenCV arrow keycodes (macOS): up=0 down=1 left=2 right=3


def handle_key(view: CameraView, key: int) -> bool:
    """Apply a keypress to the view. Returns True if the view changed."""
    ch = chr(key & 0xFF) if key & 0xFF else ""
    if key in ARROWS:
        ch = ARROWS[key]
    fn = KEYS.get(ch)
    if fn is None:
        return False
    before = asdict(view)
    fn(view)
    return asdict(view) != before


HELP = "r/R rotate  +/- zoom  wasd/arrows pan  f/F focus lock/auto  0 reset  space capture  q quit"


# --- persistence ------------------------------------------------------------
def _path(root: Path) -> Path:
    return root / ".ocrmd-camera.json"


def load_view(root: Path, camera_name: str) -> CameraView:
    try:
        data = json.loads(_path(root).read_text()).get(camera_name)
        return CameraView(**data) if data else CameraView()
    except (OSError, ValueError, TypeError):
        return CameraView()


def save_view(root: Path, camera_name: str, view: CameraView) -> None:
    p = _path(root)
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        data = {}
    data[camera_name] = asdict(view)
    p.write_text(json.dumps(data, indent=1))
