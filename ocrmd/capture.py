"""Webcam auto-shutter: motion gate -> focus gate -> hand gate -> novelty gate -> quality gate -> capture.

Once the gates open the shutter does not take one frame: it takes a *burst* at the current focus and
keeps the sharpest, and — when the camera can be driven to an explicit lens position — a *focus
bracket* around it. The extra exposures travel with the capture as variants, so a line the sharpest
frame cannot resolve can be re-read at another focus (`multifocus.py`) instead of being guessed at.
"""
from __future__ import annotations
import logging, queue, time
from dataclasses import dataclass, field
import cv2, numpy as np
from .config import Settings
from .fingerprint import dhash, hamming
from .camview import CameraView, handle_key, HELP, load_view, save_view
from . import camctl, preprocess, occlusion, scene
from . import focus as focus_mod   # `focus` is a parameter name in this module
from .hud import ScanState, draw as draw_hud

log = logging.getLogger("ocrmd.capture")
WINDOW = "ocrmd scan"


@dataclass
class Shot:
    """One exposure of the bracket: the frame, the lens position it was taken at, its page sharpness."""
    image: np.ndarray
    lens: float | None
    sharpness: float


@dataclass
class Frame:
    image: np.ndarray
    phash: str
    reason: str                      # auto | forced
    meta: dict = field(default_factory=dict)   # sharpness, occlusion, page_px at capture time
    variants: list[Shot] = field(default_factory=list)   # other focus positions of the same page


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

    Two questions, deliberately separated. `scene.Scene` answers *has the desk stopped changing?* —
    stillness, focus, a hand, and whether the detected page outline is holding still — and remembers
    what kind of disturbance it is settling from, so a page turn has to prove it has finished. This
    class answers the rest, which are properties of the page rather than of the scene: is it new, and
    is it sharp enough.

    That split is the fix for the failure this was rebuilt around. Every test in the old chain can be
    true in the middle of a page turn, and on the real store one two-minute session captured 21 times
    while the detected page outline swung 100-700 px between consecutive shots.
    """

    def __init__(self, s: Settings, rig_best: float = 0.0):
        self.s = s
        self.prev_small: np.ndarray | None = None
        self.last_hash: str | None = None
        self.best_sharp = 0.0
        self.tracker = occlusion.Tracker(cover_threshold=s.occlusion_cover, rebaseline_after=s.occlusion_rebaseline)
        self.scene = scene.Scene(stable_frames=s.stable_frames, settle_frames=s.settle_frames,
                                 window=s.geometry_frames, tolerance=s.geometry_tolerance)
        # `rig_best` is what this camera has managed before, remembered across sessions. Without it a
        # session that starts out of focus lowers its own bar to match (see focus.py).
        self.focus = focus_mod.Watch(best=rig_best, soft_ratio=s.focus_soft_ratio,
                                 soft_frames=s.focus_soft_frames, hunt_band=s.focus_hunt_band,
                                 recovery_gap=s.focus_recovery_gap)

    @property
    def still(self) -> int:
        """Consecutive undisturbed frames — what the HUD's stillness meter shows."""
        return self.scene.quiet

    def reset_scene(self) -> None:
        self.prev_small, self.last_hash = None, None
        self.tracker.reset()
        self.scene.reset()

    def note_page(self, rect, shape) -> None:
        """Feed a freshly detected page outline (`None` when no page was found). Called only when the
        outline is actually re-detected, which the capture loop does a few times a second."""
        self.scene.note_outline(scene.Outline.from_rect(rect, shape) if rect is not None else None)

    def evaluate(self, frame: np.ndarray, quad: np.ndarray | None, adjusting: bool,
                 require_new: bool = True) -> dict:
        s = self.s
        blur = sharpness(frame, quad=quad)   # on the raw frame, before any smoothing
        small = cv2.cvtColor(cv2.resize(frame, (s.analysis_width, int(frame.shape[0] * s.analysis_width / frame.shape[1]))),
                             cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0)  # suppress sensor noise before differencing
        motion = (float(np.abs(small.astype(np.int16) - self.prev_small.astype(np.int16)).mean())
                  if self.prev_small is not None else 999.0)
        self.prev_small = small
        h = dhash(small)
        novelty = hamming(h, self.last_hash) if self.last_hash else 999
        occ = self.tracker.update(frame, s.analysis_width)
        # The lens is watched in the image, not in the driver: `adjusting` is false on every frame of
        # every camera measured so far, and is kept only as an extra reason to wait (focus.py).
        f = self.focus.observe(blur, occluded=occ.occluded, adjusting=adjusting)
        v = self.scene.observe(moving=motion >= s.motion_threshold, adjusting=f.hunting,
                               occluded=occ.occluded)
        # Adaptive floor: what counts as sharp depends on the lens, the lighting and how far away
        # the page is, so a single absolute number is either too strict for one rig or too loose
        # for another. Track the best this scene has produced and refuse anything well below it.
        # A hand in frame is excluded from that tracking: skin is sharper than paper.
        if not occ.occluded:
            self.best_sharp = max(blur, self.best_sharp * s.sharp_decay)
        floor = self.focus.floor(s.blur_threshold, self.best_sharp, s.sharp_ratio)
        # `require_new` is off when watching a page for annotations: the point is to look at the same
        # page again and again, and "this is the page you just captured" is the answer, not a blocker.
        gate = (v.blocker or
                ("same page" if require_new and novelty < s.novelty_threshold else
                 "blurry" if blur < floor else "READY"))
        return {"gate": gate, "motion": motion, "novelty": novelty, "sharp": blur, "floor": floor,
                "hash": h, "occlusion": occ, "scene": v, "still": v.quiet, "needed": v.needed,
                "settling_from": v.disturbance, "drift": round(v.drift, 4), "focus": f}

    def captured(self, h: str) -> None:
        self.last_hash = h
        self.scene.captured()


def sharpest_burst(cap, view, frame: np.ndarray, sharp: float, quad, n: int
                   ) -> tuple[np.ndarray, float, list[Shot]]:
    """Read up to `n - 1` more frames right after the gate opens and keep the sharpest of the burst.

    The scene is still by construction (the motion gate passed), so the frames differ only in sensor
    noise, compression and the lens's last micro-adjustment — and Laplacian variance between consecutive
    webcam frames of the same page varies by 10-20 %. Choosing the best one is free sharpness.

    The runners-up are returned rather than dropped, sharpest first. On a camera that *can* be locked
    to a lens position they are ignored in favour of a real focus bracket; on one that cannot — the
    C525 on the development rig supports `locked` and `continuous` but not a custom lens position, and
    neither camera attached to it can be bracketed — they are the only second opinion available, and
    they have already been paid for.
    """
    shots = [Shot(frame, None, sharp)]
    for _ in range(max(0, n - 1)):
        ok, f = cap.read()
        if not ok or f is None:
            break
        f = view.apply(f)
        shots.append(Shot(f, None, sharpness(f, quad=quad)))
    shots.sort(key=lambda sh: -sh.sharpness)
    best = shots[0]
    if best.image is not frame:
        log.debug("burst: kept a sharper frame (%.0f -> %.0f)", sharp, best.sharpness)
    return best.image, best.sharpness, shots[1:]


def _read_settled(cap, view, settle: float, max_reads: int = 30) -> np.ndarray | None:
    """Read frames for `settle` seconds and return the last one.

    Both halves matter. The lens needs time to travel — AVFoundation reports the move as complete
    through a completion handler this thread has no run loop to pump, so the wait is what stands in
    for it. And the frames already in the camera's buffer were exposed *before* the move, so reading
    the next frame after a plain sleep can hand back the old focus; reading continuously flushes them.
    """
    t0, last = time.monotonic(), None
    for _ in range(max_reads):
        ok, f = cap.read()
        if not ok or f is None:
            break
        last = f
        if time.monotonic() - t0 >= settle:
            break
    return view.apply(last) if last is not None else None


def bracket_offsets(n: int, step: float) -> list[float]:
    """Lens offsets for `n` shots, nearest first: [-1, +1, -2, +2, ...] x step, excluding the shot the
    lens is already at. Alternating outwards means an interrupted bracket is still symmetric."""
    out = []
    for i in range(1, n):
        out.append((-1 if i % 2 else 1) * ((i + 1) // 2) * step)
    return out


def focus_bracket(cap, view, frame: np.ndarray, sharp: float, quad, s: Settings, name: str | None
                  ) -> tuple[np.ndarray, float, list[Shot]]:
    """Shoot the same still page at several lens positions; return (sharpest frame, its sharpness, the rest).

    Autofocus optimises one number over the whole frame, and on a book that number is dominated by
    whatever has the most contrast — a page edge, a gutter shadow, the sharper of two leaves of a
    curled spread. The body text it misses is not recoverable by any amount of preprocessing, but it
    is usually in focus a few hundredths of a lens position away. So the shutter also brackets: the
    frames are seconds apart on a scene the motion gate has already declared still, which makes them
    directly comparable, and the ones that lose on whole-page sharpness are kept anyway because
    sharpness is a *page* average and a variant can still win a line (see multifocus.resolve).

    Returns no variants — and costs nothing — when the camera cannot be locked to a lens position.
    """
    st = camctl.focus_state(name=name)
    start, mode = st.get("lens"), st.get("focus")
    if start is None or s.focus_bracket_shots < 2 or not camctl.supports_lens_position(name=name):
        return frame, sharp, []
    shots = [Shot(frame, start, sharp)]
    try:
        for off in bracket_offsets(s.focus_bracket_shots, s.focus_bracket_step):
            pos = camctl.set_lens_position(start + off, name=name)
            if pos is None:
                break
            if any(abs(pos - sh.lens) < 1e-3 for sh in shots):   # clamped onto a position already shot
                continue
            f = _read_settled(cap, view, s.focus_bracket_settle)
            if f is None:
                break
            shots.append(Shot(f, pos, sharpness(f, quad=quad)))
    finally:
        # Put the lens back. Leaving the scan locked at the last bracket position would quietly
        # defocus every page after this one — the one failure this feature must not introduce.
        if mode in ("continuous", "auto"):
            camctl.set_focus(mode, name=name)
        else:
            camctl.set_lens_position(start, name=name)
    shots.sort(key=lambda sh: -sh.sharpness)
    best, rest = shots[0], shots[1:]
    log.info("focus bracket: %s (kept %d variant(s))",
             " ".join(f"{sh.lens:.3f}={sh.sharpness:.0f}" for sh in shots), min(len(rest), max(0, s.focus_keep)))
    return best.image, best.sharpness, rest[:max(0, s.focus_keep)]


def auto_shutter(s: Settings, preview: bool = True, on_status=print, view: CameraView | None = None,
                 resolution: tuple[int, int] | None = None, focus: str = "continuous",
                 state: ScanState | None = None, duration: float | None = None,
                 max_captures: int | None = None, on_frame=None, commands: "queue.Queue[str] | None" = None,
                 require_new: bool = True):
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
    # The mode this session runs in, read after `open_camera` has applied it. Restored on the way out
    # so that `focus=settle`'s lock does not outlive the scan: measured on this rig, a session that
    # ends locked comes back up locked, and a lens parked somewhere useless is the failure focus.py
    # exists to catch.
    entry_focus = camctl.focus_state(name=name).get("focus")
    rig_best = focus_mod.load_best(s.root, name or "")
    st = camctl.focus_state(name=name)
    on_status(f"camera '{name}' {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
              f"  focus={st.get('focus', 'n/a')} exposure={st.get('exposure', 'n/a')}  view: {view.describe()}")
    state = state or ScanState()
    gate = Gate(s, rig_best=rig_best)
    if rig_best:
        on_status(f"this rig has focused to {rig_best:.0f} before; anything under "
                  f"{rig_best * s.sharp_ratio:.0f} will be treated as out of focus")
    last_emit, last_status, last_fstate = 0.0, 0.0, 0.0
    fstate: dict = camctl.focus_state(name=name)
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
            if time.time() - last_fstate > 0.2:      # the driver's opinion changes slowly and is rarely useful
                fstate, last_fstate = camctl.focus_state(name=name), time.time()
            quad_age -= 1
            if quad_age <= 0:                      # page outline is the main framing aid; ~3/s is plenty
                rect = preprocess.find_page_rect(frame)
                quad_age = 6
                gate.note_page(rect, frame.shape)   # the outline is a gate, not only a framing aid
                quad = cv2.boxPoints(rect) if rect is not None else None
                page_px = preprocess.estimated_page_width(frame, rect) if rect is not None else None
                clipped = preprocess.clipped_edges(quad, frame.shape, img=frame) if quad is not None else []
            g = gate.evaluate(frame, quad, adjusting=bool(fstate.get("adjusting")), require_new=require_new)
            act = g["focus"].action
            if act and s.focus_recovery:
                on_status(g["focus"].note, "warn" if act != "accept" else "info")
                log.warning("focus %s: %s", act, g["focus"].note)
                if act == "retrigger":
                    mode = camctl.retrigger_focus(name=name)
                    if mode is None:
                        on_status("this camera offers no focus control; check the lens and the distance", "warn")
                elif act == "reopen":
                    cap.release(); time.sleep(0.4)
                    cap = open_camera(s.camera_index, resolution=resolution, focus=focus)
                    gate.reset_scene()
                gate.focus.acted()
            force = False
            keys: list[int] = []
            if preview or on_frame is not None:
                disp = draw_hud(frame, gate=g["gate"], motion=g["motion"], still=g["still"], stable_frames=g["needed"],
                                settling_from=g["settling_from"], drift=g["drift"],
                                novelty=g["novelty"], novelty_threshold=s.novelty_threshold, sharp=g["sharp"],
                                blur_threshold=g["floor"], view_desc=view.describe(),
                                focus=fstate.get("focus", "n/a"), adjusting=g["focus"].hunting,
                                focus_note=g["focus"].describe(),
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
                                    "still": g["still"], "stable_frames": g["needed"],
                                    "settling_from": g["settling_from"], "drift": g["drift"],
                                    "novelty_threshold": s.novelty_threshold,
                                    "occlusion": g["occlusion"].describe(), "cover": g["occlusion"].missing,
                                    "page_px": page_px, "min_page_px": s.min_page_width, "clipped": list(clipped),
                                    "focus": fstate.get("focus"), "adjusting": g["focus"].hunting,
                                    "focus_note": g["focus"].describe(), "focus_ratio": round(g["focus"].ratio, 3),
                                    "rig_best": round(gate.focus.best, 1),
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
                frame, g["sharp"], spare = sharpest_burst(cap, view, frame, g["sharp"], quad, s.burst_frames)
                variants: list[Shot] = []
                # `auto` only brackets when this frame is not the best the scene has shown: when the
                # lens is already on the page a bracket costs ~0.5 s of frozen preview for nothing.
                if s.focus_bracket == "always" or (s.focus_bracket == "auto"
                                                   and g["sharp"] < gate.best_sharp * s.focus_bracket_ratio):
                    frame, g["sharp"], variants = focus_bracket(cap, view, frame, g["sharp"], quad, s, name)
                    if not variants:
                        # fixed-lens camera: the burst's runners-up are the only other exposures there
                        # are. They differ only by noise and the lens's own micro-adjustment, so they
                        # buy far less than a real bracket — but they are already in hand.
                        variants = spare[:max(0, s.focus_keep)]
                meta = {"sharpness": round(g["sharp"], 1), "occlusion": g["occlusion"].describe(),
                        "page_px": page_px, "clipped": list(clipped), "forced": force}
                if variants:
                    meta["focus_bracket"] = [{"lens": v.lens, "sharpness": round(v.sharpness, 1)} for v in variants]
                log.info("capture #%d %s %s", captures, "forced" if force else "auto", meta)
                if not settled and captures >= s.settle_after:
                    # fixed rig: the lens has found the page; stop it hunting every time a hand comes in
                    if camctl.set_focus("locked", name=name):
                        on_status("focus settled -> locked (F returns to continuous)")
                    settled = True
                yield Frame(frame.copy(), g["hash"], "forced" if force else "auto", meta,
                            [Shot(v.image.copy(), v.lens, v.sharpness) for v in variants])
            elif not preview and on_frame is None and time.time() - last_status > 2.0:
                last_status = time.time()
                on_status(f"{g['scene'].describe() if not g['scene'].settled else g['gate']}: "
                          f"motion={g['motion']:.1f} novelty={g['novelty']} drift={g['drift']:.3f} "
                          f"sharpness={g['sharp']:.0f}/{g['floor']:.0f} {g['occlusion'].describe()}")
    finally:
        cap.release()
        # A scan that ends with the lens locked leaves the next one starting from wherever it parked —
        # measured: this rig comes back up `locked` after every session that locked it.
        if entry_focus and camctl.focus_state(name=name).get("focus") != entry_focus:
            camctl.set_focus(entry_focus, name=name)
        if name and gate.focus.best > rig_best:
            focus_mod.save_best(s.root, name, gate.focus.best)
        elif name and gate.focus.accepted:
            focus_mod.save_best(s.root, name, gate.focus.best)
        if preview:
            cv2.destroyAllWindows()
            for _ in range(4):      # macOS needs a few event-loop turns to actually close the window
                cv2.waitKey(1)
