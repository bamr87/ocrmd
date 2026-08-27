"""Webcam auto-shutter: motion gate -> focus gate -> hand gate -> novelty gate -> quality gate -> capture."""
from __future__ import annotations
import logging, queue, time
from dataclasses import dataclass, field
import cv2, numpy as np
from .config import Settings
from .fingerprint import dhash, hamming
from .camview import CameraView, handle_key, HELP, load_view, save_view
from . import camctl, preprocess, occlusion
from .hud import ScanState, draw as draw_hud

log = logging.getLogger("ocrmd.capture")
WINDOW = "ocrmd scan"


@dataclass
class Frame:
    image: np.ndarray
    phash: str
    reason: str                      # auto | forced
    meta: dict = field(default_factory=dict)   # sharpness, occlusion, page_px at capture time


class CameraLost(RuntimeError):
    """The camera stopped delivering frames and could not be reopened."""


# Commands a driver without a keyboard (the web UI) can send; they map onto the preview's keys.
COMMAND_KEYS = {"capture": ord(" "), "focus_lock": ord("f"), "focus_auto": ord("F"),
                "rotate": ord("r"), "rotate_ccw": ord("R"), "zoom_in": ord("+"), "zoom_out": ord("-"),
                "pan_up": ord("w"), "pan_down": ord("s"), "pan_left": ord("a"), "pan_right": ord("d"),
                "reset": ord("0")}


def resolve_camera(spec: str | int) -> int:
    """An OpenCV index, or a case-insensitive substring of the device name (e.g. 'c525').

    Names are matched through the probed index map, not AVFoundation's enumeration order: the two
    orders differ, so matching on the enumeration order opens the wrong camera.
    """
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    mapping = camctl.map_cv2_indices()
    for idx, name in sorted(mapping.items()):
        if str(spec).lower() in name.lower():
            return idx
    raise RuntimeError(f"no camera matching {spec!r}; available: "
                       + ", ".join(f"{i}={n}" for i, n in sorted(mapping.items())))


def open_camera(index: int, resolution: tuple[int, int] | None = None, focus: str = "continuous") -> cv2.VideoCapture:
    """Open a camera at its best resolution with focus/exposure running.

    Two macOS specifics are handled here: OpenCV cannot read or write any camera property through the
    AVFoundation backend (see camctl), and webcams frequently start with focus and exposure LOCKED —
    a C525 sits at lens position 0.0 and every frame is blurry until the mode is changed.
    """
    name = camctl.name_for_cv2_index(index)
    if focus in ("continuous", "settle", "locked"):
        camctl.set_focus("continuous" if focus == "settle" else focus, name=name, index=None if name else index)
        camctl.set_exposure("continuous", name=name, index=None if name else index)
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index} (macOS: grant Camera permission to your terminal/IDE)")
    w, h = resolution or camctl.best_native_format(name=name, index=None if name else index) or (1280, 960)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def sharpness(frame: np.ndarray, crop: float = 0.6, width: int = 640,
              quad: np.ndarray | None = None) -> float:
    """Laplacian variance of the page, at a fixed scale so the number is comparable run to run.

    Measured inside the detected page when there is one, because that is what has to be readable —
    a frame-centre crop scores the desk, a hand, or whichever part of the page happens to be central,
    and let visibly soft captures through the gate. Measured on the *unmodified* frame: the smoothing
    applied for motion detection destroys exactly this signal.
    """
    h, w = frame.shape[:2]
    if quad is not None:
        x0, y0 = int(max(0, quad[:, 0].min())), int(max(0, quad[:, 1].min()))
        x1, y1 = int(min(w, quad[:, 0].max())), int(min(h, quad[:, 1].max()))
        if x1 - x0 > 80 and y1 - y0 > 80:
            centre = frame[y0:y1, x0:x1]
        else:
            quad = None
    if quad is None:
        cw, ch = int(w * crop), int(h * crop)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        centre = frame[y0:y0 + ch, x0:x0 + cw]
    scale = width / max(1, centre.shape[1])
    if scale < 1:
        centre = cv2.resize(centre, (width, int(centre.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(centre, cv2.COLOR_BGR2GRAY) if centre.ndim == 3 else centre
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def camera_name(index: int) -> str:
    return camctl.name_for_cv2_index(index) or f"camera{index}"


def list_devices() -> list[str]:
    """Camera names in AVFoundation order."""
    return camctl.device_names()


def camera_authorized() -> bool | None:
    try:
        import AVFoundation as AV
        return AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo) == 3  # authorized
    except Exception:
        return None


def list_cameras(max_index: int = 4) -> list[int]:
    found = []
    for i in range(max_index):
        c = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if c.isOpened():
            found.append(i)
        c.release()
    return found


class Gate:
    """The per-frame decision, kept separate from I/O so it can be unit-tested.

    Order matters: a moving scene says nothing about the rest; a hunting lens makes sharpness
    meaningless; a hand in frame must beat "new page" (the hand is what makes the frame new).
    """

    def __init__(self, s: Settings):
        self.s = s
        self.prev_small: np.ndarray | None = None
        self.still = 0
        self.last_hash: str | None = None
        self.best_sharp = 0.0
        self.tracker = occlusion.Tracker(cover_threshold=s.occlusion_cover, rebaseline_after=s.occlusion_rebaseline)

    def reset_scene(self) -> None:
        self.prev_small, self.still, self.last_hash = None, 0, None
        self.tracker.reset()

    def evaluate(self, frame: np.ndarray, quad: np.ndarray | None, adjusting: bool) -> dict:
        s = self.s
        blur = sharpness(frame, quad=quad)   # on the raw frame, before any smoothing
        small = cv2.cvtColor(cv2.resize(frame, (s.analysis_width, int(frame.shape[0] * s.analysis_width / frame.shape[1]))),
                             cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0)  # suppress sensor noise before differencing
        motion = (float(np.abs(small.astype(np.int16) - self.prev_small.astype(np.int16)).mean())
                  if self.prev_small is not None else 999.0)
        self.prev_small = small
        self.still = self.still + 1 if motion < s.motion_threshold else 0
        h = dhash(small)
        novelty = hamming(h, self.last_hash) if self.last_hash else 999
        occ = self.tracker.update(frame, s.analysis_width)
        # Adaptive floor: what counts as sharp depends on the lens, the lighting and how far away
        # the page is, so a single absolute number is either too strict for one rig or too loose
        # for another. Track the best this scene has produced and refuse anything well below it.
        # A hand in frame is excluded from that tracking: skin is sharper than paper.
        if not occ.occluded:
            self.best_sharp = max(blur, self.best_sharp * s.sharp_decay)
        floor = max(s.blur_threshold, self.best_sharp * s.sharp_ratio)
        gate = ("moving" if self.still < s.stable_frames else
                "focusing" if adjusting else
                "hand" if occ.occluded else
                "same page" if novelty < s.novelty_threshold else
                "blurry" if blur < floor else "READY")
        return {"gate": gate, "motion": motion, "novelty": novelty, "sharp": blur, "floor": floor,
                "hash": h, "occlusion": occ}

    def captured(self, h: str) -> None:
        self.last_hash = h


def sharpest_burst(cap, view, frame: np.ndarray, sharp: float, quad, n: int) -> tuple[np.ndarray, float]:
    """Read up to `n - 1` more frames right after the gate opens and keep the sharpest of the burst.

    The scene is still by construction (the motion gate passed), so the frames differ only in sensor
    noise, compression and the lens's last micro-adjustment — and Laplacian variance between consecutive
    webcam frames of the same page varies by 10-20 %. Choosing the best one is free sharpness.
    """
    best, best_sharp = frame, sharp
    for _ in range(max(0, n - 1)):
        ok, f = cap.read()
        if not ok or f is None:
            break
        f = view.apply(f)
        sh = sharpness(f, quad=quad)
        if sh > best_sharp:
            best, best_sharp = f, sh
    if best is not frame:
        log.debug("burst: kept a sharper frame (%.0f -> %.0f)", sharp, best_sharp)
    return best, best_sharp


def auto_shutter(s: Settings, preview: bool = True, on_status=print, view: CameraView | None = None,
                 resolution: tuple[int, int] | None = None, focus: str = "continuous",
                 state: ScanState | None = None, duration: float | None = None,
                 max_captures: int | None = None, on_frame=None, commands: "queue.Queue[str] | None" = None):
    """Generator yielding a Frame whenever a new, still, sharp, unobstructed page is in view.
    Stops on its own after `duration` seconds or `max_captures` captures (unattended runs, smoke tests).

    `preview=True` opens the OpenCV window and reads its keys. `on_frame(hud_image, info)` receives the
    same annotated frame without a window (the web UI streams it), and `commands` is a queue of
    COMMAND_KEYS names (plus "quit") standing in for key presses. Both may be combined.

    The caller must not block on a yielded frame: while this generator is suspended no frames are
    read and no key events are serviced, so the preview window stops redrawing and macOS marks it
    unresponsive. `ocrmd scan` hands each frame to a worker thread and comes straight back.
    """
    cap = open_camera(s.camera_index, resolution=resolution, focus=focus)
    name = camera_name(s.camera_index)
    view = view or load_view(s.root, name)
    st = camctl.focus_state(name=name)
    on_status(f"camera '{name}' {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
              f"  focus={st.get('focus', 'n/a')} exposure={st.get('exposure', 'n/a')}  view: {view.describe()}")
    state = state or ScanState()
    gate = Gate(s)
    last_emit, last_status = 0.0, 0.0
    quad, quad_age, flash, page_px, clipped = None, 0, 0.0, None, []
    failed_reads, reopens, captures = 0, 0, 0
    settled = focus != "settle"
    t_start = time.time()
    try:
        while True:
            if duration is not None and time.time() - t_start > duration:
                on_status(f"duration {duration:.0f}s reached")
                return
            if max_captures is not None and captures >= max_captures:
                on_status(f"{max_captures} capture(s) reached")
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= s.camera_reopen_after:
                    reopens += 1
                    if reopens > 5:
                        raise CameraLost(f"camera '{name}' stopped delivering frames")
                    on_status(f"camera '{name}' not responding; reopening ({reopens}/5)")
                    log.warning("reopening camera %s after %d failed reads", name, failed_reads)
                    cap.release(); time.sleep(1.0)
                    cap = open_camera(s.camera_index, resolution=resolution, focus=focus)
                    failed_reads = 0
                    gate.reset_scene()
                time.sleep(0.05); continue
            failed_reads = 0
            frame = view.apply(frame)
            fstate = camctl.focus_state(name=name)
            quad_age -= 1
            if quad_age <= 0:                      # page outline is the main framing aid; ~3/s is plenty
                rect = preprocess.find_page_rect(frame)
                quad_age = 6
                quad = cv2.boxPoints(rect) if rect is not None else None
                page_px = preprocess.estimated_page_width(frame, rect) if rect is not None else None
                clipped = preprocess.clipped_edges(quad, frame.shape, img=frame) if quad is not None else []
            g = gate.evaluate(frame, quad, adjusting=bool(fstate.get("adjusting")))
            force = False
            keys: list[int] = []
            if preview or on_frame is not None:
                disp = draw_hud(frame, gate=g["gate"], motion=g["motion"], still=gate.still, stable_frames=s.stable_frames,
                                novelty=g["novelty"], novelty_threshold=s.novelty_threshold, sharp=g["sharp"],
                                blur_threshold=g["floor"], view_desc=view.describe(),
                                focus=fstate.get("focus", "n/a"), adjusting=bool(fstate.get("adjusting")),
                                resolution=(frame.shape[1], frame.shape[0]), page_quad=quad, state=state,
                                help_text=HELP, flash=flash, page_px=page_px,
                                min_page_px=s.min_page_width, clipped=clipped, occlusion=g["occlusion"].describe())
                flash = max(0.0, flash - 0.12)
                if preview:
                    cv2.imshow(WINDOW, disp)
                    k = cv2.waitKeyEx(1)
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        return                   # user closed the window
                    if k != -1:
                        keys.append(k)
                if on_frame is not None:
                    on_frame(disp, {"gate": g["gate"], "motion": round(g["motion"], 2), "novelty": g["novelty"],
                                    "sharp": round(g["sharp"], 1), "floor": round(g["floor"], 1),
                                    "still": gate.still, "stable_frames": s.stable_frames,
                                    "novelty_threshold": s.novelty_threshold,
                                    "occlusion": g["occlusion"].describe(), "cover": g["occlusion"].missing,
                                    "page_px": page_px, "min_page_px": s.min_page_width, "clipped": list(clipped),
                                    "focus": fstate.get("focus"), "adjusting": bool(fstate.get("adjusting")),
                                    "resolution": [frame.shape[1], frame.shape[0]], "view": view.describe(),
                                    "captures": captures})
            if commands is not None:
                while True:
                    try:
                        cmd = commands.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "quit":
                        return
                    if cmd in COMMAND_KEYS:
                        keys.append(COMMAND_KEYS[cmd])
            for k in keys:
                ch = k & 0xFF
                if ch == ord("q"):
                    return
                if ch == ord(" "):
                    force = True
                elif ch in (ord("f"), ord("F")):
                    mode = "locked" if ch == ord("f") else "continuous"
                    ok_f = camctl.set_focus(mode, name=name)
                    on_status(f"focus -> {mode}" + ("" if ok_f else " (not supported by this camera)"))
                    settled = True               # the user took over focus
                elif handle_key(view, k):
                    save_view(s.root, name, view)
                    gate.reset_scene()           # the scene changed by definition
                    on_status(f"view: {view.describe()}")
            if (g["gate"] == "READY" and time.time() - last_emit > s.capture_cooldown) or force:
                gate.captured(g["hash"])
                last_emit, flash = time.time(), 1.0
                state.captured += 1
                captures += 1
                frame, g["sharp"] = sharpest_burst(cap, view, frame, g["sharp"], quad, s.burst_frames)
                meta = {"sharpness": round(g["sharp"], 1), "occlusion": g["occlusion"].describe(),
                        "page_px": page_px, "clipped": list(clipped), "forced": force}
                log.info("capture #%d %s %s", captures, "forced" if force else "auto", meta)
                if not settled and captures >= s.settle_after:
                    # fixed rig: the lens has found the page; stop it hunting every time a hand comes in
                    if camctl.set_focus("locked", name=name):
                        on_status("focus settled -> locked (F returns to continuous)")
                    settled = True
                yield Frame(frame.copy(), g["hash"], "forced" if force else "auto", meta)
            elif not preview and on_frame is None and time.time() - last_status > 2.0:
                last_status = time.time()
                on_status(f"{g['gate']}: motion={g['motion']:.1f} novelty={g['novelty']} "
                          f"sharpness={g['sharp']:.0f}/{g['floor']:.0f} {g['occlusion'].describe()}")
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()
            for _ in range(4):      # macOS needs a few event-loop turns to actually close the window
                cv2.waitKey(1)
