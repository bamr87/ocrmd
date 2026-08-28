"""The hand gate and the auto-shutter decision, on synthetic frames."""
from __future__ import annotations
import cv2
import numpy as np

from conftest import on_desk, text_page, with_hand
from ocrmd import occlusion
from ocrmd.capture import Gate
from ocrmd.config import Settings


def _rig():
    page = text_page()
    clean = on_desk(np.hstack([page, np.full((page.shape[0], 26, 3), 90, np.uint8), page]), margin=120)
    return clean


def test_hand_over_the_page_is_detected_and_a_clean_page_is_not():
    clean = _rig()
    tr = occlusion.Tracker(cover_threshold=0.05)
    assert not tr.update(clean).occluded                 # first frame becomes the reference
    o = tr.update(with_hand(clean))
    assert o.occluded and o.missing > 0.05 and o.skin > 0.5, o
    assert not tr.update(clean).occluded                 # hand withdrawn


def test_hand_does_not_shrink_the_reference():
    clean = _rig()
    tr = occlusion.Tracker(cover_threshold=0.05)
    tr.update(clean)
    ref_area = int(tr.reference.sum())
    for _ in range(5):
        assert tr.update(with_hand(clean)).occluded
    assert int(tr.reference.sum()) == ref_area           # still the clean sheet


def test_moving_the_book_rebaselines_immediately():
    clean = _rig()
    tr = occlusion.Tracker(cover_threshold=0.05)
    tr.update(clean)
    shifted = np.roll(clean, clean.shape[1] // 6, axis=1)   # book slid sideways: paper lost AND gained
    o = tr.update(shifted)
    assert o.moved and not o.occluded, o
    assert not tr.update(shifted).occluded


def test_a_non_skin_intrusion_is_accepted_after_a_while():
    """A bookmark or a shadow that stays put must not hold the shutter forever."""
    clean = _rig()
    tr = occlusion.Tracker(cover_threshold=0.05, rebaseline_after=5)
    tr.update(clean)
    dark = clean.copy()
    h, w = dark.shape[:2]
    cv2.rectangle(dark, (int(w * 0.4), int(h * 0.2)), (int(w * 0.6), int(h * 0.8)), (30, 30, 30), -1)
    results = [tr.update(dark).occluded for _ in range(6)]
    assert results[0] and not results[-1], results


def test_a_different_page_of_the_same_book_is_not_an_occlusion():
    a = text_page()
    b = text_page(lines=["completely different text on this page"] * 14)
    tr = occlusion.Tracker(cover_threshold=0.05)
    tr.update(on_desk(a, margin=120))
    assert not tr.update(on_desk(b, margin=120)).occluded


def _settings(**kw) -> Settings:
    kw.setdefault("settle_frames", 6)
    return Settings(stable_frames=3, novelty_threshold=12, blur_threshold=10.0, sharp_ratio=0.3, **kw)


def _run(g, frame, n, adjusting=False):
    return [g.evaluate(frame, None, adjusting=adjusting)["gate"] for _ in range(n)]


def test_gate_orders_hand_before_new_page():
    """A hand makes the frame 'new' by every other measure; the hand gate has to win."""
    clean = _rig()
    g = Gate(_settings())
    for _ in range(5):
        r = g.evaluate(clean, None, adjusting=False)
    assert r["gate"] == "READY", r["gate"]
    g.captured(r["hash"])
    # a capture spends the scene's quiet: it has to settle again before it may fire a second time
    assert _run(g, clean, 4)[-1] == "same page"
    hand = with_hand(clean)
    states = _run(g, hand, 5)
    assert states[0] == "moving" and states[-1] == "hand", states
    # a hand is a *major* disturbance: the short stillness that used to be enough no longer is
    # (the first frame after it leaves is "moving" — the hand vanishing is itself a change)
    assert "READY" not in _run(g, clean, 4)
    states = _run(g, clean, 6)
    assert states[-1] == "same page", states           # same page as the last capture, hand gone


def test_gate_reports_focusing_and_blur():
    clean = _rig()
    g = Gate(_settings())
    for _ in range(4):
        g.evaluate(clean, None, adjusting=False)
    assert g.evaluate(clean, None, adjusting=True)["gate"] == "focusing"
    soft = cv2.GaussianBlur(clean, (0, 0), 6)
    states = _run(g, soft, 8)
    assert states[-1] == "blurry", states


def test_gate_reset_forgets_the_scene():
    g = Gate(_settings())
    clean = _rig()
    for _ in range(4):
        g.evaluate(clean, None, adjusting=False)
    g.reset_scene()
    assert g.still == 0 and g.tracker.reference is None


def test_sharpest_burst_keeps_the_sharpest_frame():
    """The shutter reads a short burst once the gates pass and keeps the frame with the most detail."""
    import cv2
    from ocrmd import capture
    from ocrmd.camview import CameraView
    sharp = text_page()
    soft = cv2.GaussianBlur(sharp, (7, 7), 0)

    class Cap:
        def __init__(self, frames): self.frames = list(frames)
        def read(self):
            return (True, self.frames.pop(0)) if self.frames else (False, None)

    view = CameraView()
    s0 = capture.sharpness(soft)
    best, score, spare = capture.sharpest_burst(Cap([soft, sharp, soft]), view, soft, s0, None, 4)
    assert score > s0 and best is sharp
    # the losers are handed back sharpest first: on a camera that cannot be focus-bracketed they are
    # the only other exposures of the page there are
    assert len(spare) == 3 and all(sh.sharpness <= score for sh in spare)
    # burst of one: nothing is read, the gated frame is kept
    best, score, spare = capture.sharpest_burst(Cap([sharp]), view, soft, s0, None, 1)
    assert best is soft and score == s0 and spare == []
    # the camera stalling mid-burst is not an error
    best, score, spare = capture.sharpest_burst(Cap([]), view, soft, s0, None, 3)
    assert best is soft and spare == []


# --- focus bracket ---------------------------------------------------------------------------
class _Cap:
    def __init__(self, frames): self.frames = list(frames)
    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)


def _fake_camera(monkeypatch, lens=0.5, mode="continuous", supported=True):
    """Stand in for AVFoundation: record what the bracket asks the lens to do."""
    from ocrmd import camctl
    log = {"lens": [], "focus": []}
    monkeypatch.setattr(camctl, "focus_state", lambda **kw: {"lens": lens, "focus": mode, "adjusting": False})
    monkeypatch.setattr(camctl, "supports_lens_position", lambda **kw: supported)
    monkeypatch.setattr(camctl, "set_lens_position",
                        lambda pos, **kw: (log["lens"].append(round(pos, 3)), max(0.0, min(1.0, pos)))[1])
    monkeypatch.setattr(camctl, "set_focus", lambda m, **kw: (log["focus"].append(m), True)[1])
    return log


def test_bracket_offsets_step_outwards_alternately():
    from ocrmd.capture import bracket_offsets
    assert bracket_offsets(3, 0.05) == [-0.05, 0.05]
    assert [round(x, 3) for x in bracket_offsets(5, 0.05)] == [-0.05, 0.05, -0.1, 0.1]
    assert bracket_offsets(1, 0.05) == []


def test_focus_bracket_keeps_the_sharpest_shot_and_hands_back_the_others(monkeypatch):
    """The bracket shoots either side of where the lens is, returns the sharpest exposure as the
    capture, and keeps the rest as variants for the OCR stage to consult."""
    import cv2
    from ocrmd import capture
    from ocrmd.camview import CameraView
    log = _fake_camera(monkeypatch, lens=0.5)
    sharp = text_page()
    soft = cv2.GaussianBlur(sharp, (9, 9), 0)
    softer = cv2.GaussianBlur(sharp, (15, 15), 0)
    s = _settings(); s.focus_bracket_settle = 0.0; s.focus_bracket_shots = 3; s.focus_keep = 2

    best, score, variants = capture.focus_bracket(_Cap([sharp, softer]), CameraView(), soft,
                                                  capture.sharpness(soft), None, s, "cam")
    assert best is sharp and score == capture.sharpness(sharp)
    assert log["lens"] == [0.45, 0.55]                    # nearest offset first, either side
    assert [round(v.lens, 2) for v in variants] == [0.5, 0.55]     # the two losers, sharpest first
    assert variants[0].image is soft and variants[1].image is softer
    assert log["focus"] == ["continuous"]                 # autofocus handed back, not left locked


def test_focus_bracket_keeps_only_what_it_is_asked_to(monkeypatch):
    import cv2
    from ocrmd import capture
    from ocrmd.camview import CameraView
    _fake_camera(monkeypatch)
    sharp = text_page()
    soft = cv2.GaussianBlur(sharp, (9, 9), 0)
    s = _settings(); s.focus_bracket_settle = 0.0; s.focus_bracket_shots = 3; s.focus_keep = 0
    _best, _score, variants = capture.focus_bracket(_Cap([sharp, soft]), CameraView(), soft,
                                                    capture.sharpness(soft), None, s, "cam")
    assert variants == []


def test_focus_bracket_restores_a_locked_lens_to_where_it_was(monkeypatch):
    """`--focus settle` locks the lens on purpose; the bracket must put it back, not leave the scan
    focused wherever its last shot happened to land."""
    import cv2
    from ocrmd import capture
    from ocrmd.camview import CameraView
    log = _fake_camera(monkeypatch, lens=0.4, mode="locked")
    sharp = text_page()
    soft = cv2.GaussianBlur(sharp, (9, 9), 0)
    s = _settings(); s.focus_bracket_settle = 0.0; s.focus_bracket_shots = 3
    capture.focus_bracket(_Cap([sharp, soft]), CameraView(), soft, capture.sharpness(soft), None, s, "cam")
    assert log["lens"][-1] == 0.4 and log["focus"] == []


def test_a_camera_without_lens_control_costs_nothing(monkeypatch):
    from ocrmd import capture
    from ocrmd.camview import CameraView
    _fake_camera(monkeypatch, supported=False)
    page = text_page()
    cap = _Cap([page, page])
    s = _settings(); s.focus_bracket_settle = 0.0
    best, score, variants = capture.focus_bracket(cap, CameraView(), page, 42.0, None, s, "cam")
    assert best is page and score == 42.0 and variants == [] and len(cap.frames) == 2   # never read


def test_the_camera_stalling_mid_bracket_is_not_an_error(monkeypatch):
    import cv2
    from ocrmd import capture
    from ocrmd.camview import CameraView
    log = _fake_camera(monkeypatch)
    sharp = text_page()
    soft = cv2.GaussianBlur(sharp, (9, 9), 0)
    s = _settings(); s.focus_bracket_settle = 0.0; s.focus_bracket_shots = 5
    best, _score, variants = capture.focus_bracket(_Cap([sharp]), CameraView(), soft,
                                                   capture.sharpness(soft), None, s, "cam")
    assert best is sharp and len(variants) == 1
    assert log["focus"] == ["continuous"]                 # and the lens is still handed back
