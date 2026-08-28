"""macOS camera control via AVFoundation.

OpenCV's AVFoundation backend reports -1 for every CAP_PROP on macOS and ignores writes, so focus,
exposure and the real list of sensor formats are unreachable through cv2. AVCaptureDevice exposes
them, and device-level configuration applies to the stream OpenCV is already reading.
"""
from __future__ import annotations
from dataclasses import dataclass

try:
    import AVFoundation as AV
    import CoreMedia as CM
    AVAILABLE = True
except ImportError:  # non-macOS, or pyobjc not installed
    AVAILABLE = False


@dataclass
class Capabilities:
    name: str
    focus_modes: list[str]
    exposure_modes: list[str]
    formats: list[tuple[int, int]]      # native sensor modes, ascending by pixel count
    custom_lens_position: bool


def _device(name_substr: str | None = None, index: int | None = None):
    if not AVAILABLE:
        return None
    sess = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        [AV.AVCaptureDeviceTypeBuiltInWideAngleCamera, AV.AVCaptureDeviceTypeExternal], AV.AVMediaTypeVideo, 0)
    devices = list(sess.devices())
    if index is not None and index < len(devices):
        return devices[index]
    if name_substr:
        for d in devices:
            if name_substr.lower() in str(d.localizedName()).lower():
                return d
    return None


_FOCUS = {"locked": 0, "auto": 1, "continuous": 2}          # AVCaptureFocusMode values
_EXPOSURE = {"locked": 0, "auto": 1, "continuous": 2}       # AVCaptureExposureMode values


def capabilities(index: int | None = None, name: str | None = None) -> Capabilities | None:
    d = _device(name, index)
    if d is None:
        return None
    fmts = []
    for f in d.formats():
        dims = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
        fmts.append((int(dims.width), int(dims.height)))
    return Capabilities(
        name=str(d.localizedName()),
        focus_modes=[k for k, v in _FOCUS.items() if d.isFocusModeSupported_(v)],
        exposure_modes=[k for k, v in _EXPOSURE.items() if d.isExposureModeSupported_(v)],
        formats=sorted(set(fmts), key=lambda wh: wh[0] * wh[1]),
        custom_lens_position=bool(d.isLockingFocusWithCustomLensPositionSupported()),
    )


def _configure(d, fn) -> bool:
    res = d.lockForConfiguration_(None)
    ok = res[0] if isinstance(res, tuple) else res
    if not ok:
        return False
    try:
        fn(d)
    finally:
        d.unlockForConfiguration()
    return True


def set_focus(mode: str, index: int | None = None, name: str | None = None) -> bool:
    """mode: 'continuous' (hunt continuously), 'auto' (one-shot), 'locked' (freeze where it is).
    Webcams often power up in `locked` at lens position 0 — every frame blurry until this is set."""
    d = _device(name, index)
    if d is None or mode not in _FOCUS or not d.isFocusModeSupported_(_FOCUS[mode]):
        return False
    return _configure(d, lambda dev: dev.setFocusMode_(_FOCUS[mode]))


def set_exposure(mode: str, index: int | None = None, name: str | None = None) -> bool:
    d = _device(name, index)
    if d is None or mode not in _EXPOSURE or not d.isExposureModeSupported_(_EXPOSURE[mode]):
        return False
    return _configure(d, lambda dev: dev.setExposureMode_(_EXPOSURE[mode]))


def set_lens_position(pos: float, index: int | None = None, name: str | None = None) -> float | None:
    """Lock the lens at an explicit position (0.0 = nearest, 1.0 = infinity). Returns the position asked
    for, or None when the device cannot be driven that way.

    This is the only way to shoot the same scene at several focus distances: `auto` focus is one-shot
    and lands wherever its own metric says, which is exactly the decision a focus bracket exists to
    second-guess. The completion handler is not used — the caller waits (`focus_bracket_settle`),
    because a handler would have to be pumped from a run loop the capture thread does not run.
    """
    d = _device(name, index)
    if d is None or not d.isLockingFocusWithCustomLensPositionSupported():
        return None
    pos = max(0.0, min(1.0, float(pos)))
    if not _configure(d, lambda dev: dev.setFocusModeLockedWithLensPosition_completionHandler_(pos, None)):
        return None
    return pos


def supports_lens_position(index: int | None = None, name: str | None = None) -> bool:
    d = _device(name, index)
    return bool(d is not None and d.isLockingFocusWithCustomLensPositionSupported())


def focus_state(index: int | None = None, name: str | None = None) -> dict:
    d = _device(name, index)
    if d is None:
        return {}
    inv = {v: k for k, v in _FOCUS.items()}
    return {"focus": inv.get(int(d.focusMode()), "?"), "adjusting": bool(d.isAdjustingFocus()),
            "lens": round(float(d.lensPosition()), 3),
            "exposure": {v: k for k, v in _EXPOSURE.items()}.get(int(d.exposureMode()), "?")}


# --- OpenCV index <-> AVFoundation device ------------------------------------
# OpenCV enumerates capture devices in its own order, which does NOT match AVFoundation's
# discovery-session order (on the development Mac the two are exactly reversed). Getting this wrong
# points focus/exposure commands at a different camera than the one being read, silently. There is no
# API that maps them, so identify each OpenCV index by what it actually delivers: every camera has
# resolutions no other attached camera supports, so one probe per index is enough.
_INDEX_MAP: dict[int, str] | None = None
_MAP_KEY: tuple | None = None


def _unique_formats() -> dict[str, tuple[int, int]]:
    """A resolution that identifies each device: supported by it and by no other attached camera."""
    devs = {}
    for i in range(8):
        c = capabilities(index=i)
        if c is None:
            break
        devs[c.name] = set(c.formats)
    out = {}
    for name, fmts in devs.items():
        others = set().union(*(f for n, f in devs.items() if n != name)) if len(devs) > 1 else set()
        unique = sorted(fmts - others, key=lambda wh: -wh[0] * wh[1])
        if unique:
            out[name] = unique[0]
    return out


def map_cv2_indices(max_index: int = 4) -> dict[int, str]:
    """{OpenCV index: AVFoundation device name}, probed once and cached for the process."""
    global _INDEX_MAP, _MAP_KEY
    names = device_names()
    key = tuple(names)
    if _INDEX_MAP is not None and _MAP_KEY == key:
        return _INDEX_MAP
    import cv2
    probes = _unique_formats()
    mapping: dict[int, str] = {}
    for idx in range(min(max_index, len(names))):     # asking beyond the device count logs an error
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            break
        try:
            for name, (w, h) in probes.items():
                if name in mapping.values():
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, frame = cap.read()
                if ok and (frame.shape[1], frame.shape[0]) == (w, h):
                    mapping[idx] = name
                    break
        finally:
            cap.release()
    _INDEX_MAP, _MAP_KEY = mapping, key
    return mapping


def device_names() -> list[str]:
    if not AVAILABLE:
        return []
    sess = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        [AV.AVCaptureDeviceTypeBuiltInWideAngleCamera, AV.AVCaptureDeviceTypeExternal], AV.AVMediaTypeVideo, 0)
    return [str(d.localizedName()) for d in sess.devices()]


def name_for_cv2_index(index: int) -> str | None:
    return map_cv2_indices().get(index)


def best_format(index: int | None = None, name: str | None = None) -> tuple[int, int] | None:
    """The largest mode the camera offers.

    Cameras advertise modes above their sensor size (the C525 lists 1920x1080 over a 1280x960
    sensor) and the obvious assumption is that those add nothing. Measured on a real page, the
    opposite holds: 1920x1080 produced 146 correctly-read words against 98 at 1280x960, because the
    page lands on ~1.7x as many pixels and the camera's own upscaling beats ours. What matters for
    OCR is pixels on the page, so take the biggest mode and let framing do the rest.
    """
    caps = capabilities(index, name)
    if not caps or not caps.formats:
        return None
    return max(caps.formats, key=lambda wh: wh[0] * wh[1])


best_native_format = best_format   # kept: earlier name
