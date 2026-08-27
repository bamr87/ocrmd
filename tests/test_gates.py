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
    return Settings(stable_frames=3, novelty_threshold=12, blur_threshold=10.0, sharp_ratio=0.3, **kw)


def test_gate_orders_hand_before_new_page():
    """A hand makes the frame 'new' by every other measure; the hand gate has to win."""
    clean = _rig()
    g = Gate(_settings())
    for _ in range(5):
        r = g.evaluate(clean, None, adjusting=False)
    assert r["gate"] == "READY", r["gate"]
    g.captured(r["hash"])
    assert g.evaluate(clean, None, adjusting=False)["gate"] == "same page"
    hand = with_hand(clean)
    states = [g.evaluate(hand, None, adjusting=False)["gate"] for _ in range(5)]
    assert states[0] == "moving" and states[-1] == "hand", states
    states = [g.evaluate(clean, None, adjusting=False)["gate"] for _ in range(5)]
    assert states[-1] == "same page", states           # same page as the last capture, hand gone


def test_gate_reports_focusing_and_blur():
    clean = _rig()
    g = Gate(_settings())
    for _ in range(4):
        g.evaluate(clean, None, adjusting=False)
    assert g.evaluate(clean, None, adjusting=True)["gate"] == "focusing"
    soft = cv2.GaussianBlur(clean, (0, 0), 6)
    states = [g.evaluate(soft, None, adjusting=False)["gate"] for _ in range(5)]
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
    best, score = capture.sharpest_burst(Cap([soft, sharp, soft]), view, soft, s0, None, 4)
    assert score > s0 and best is sharp
    # burst of one: nothing is read, the gated frame is kept
    best, score = capture.sharpest_burst(Cap([sharp]), view, soft, s0, None, 1)
    assert best is soft and score == s0
    # the camera stalling mid-burst is not an error
    best, score = capture.sharpest_burst(Cap([]), view, soft, s0, None, 3)
    assert best is soft
