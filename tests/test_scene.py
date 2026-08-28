"""The shutter's model of the desk: page-turn sequences, frame by frame.

These are the cases the old memoryless gate got wrong. Each test walks a scripted sequence of frames
rather than asserting on one, because the whole point of the rebuild is that the decision depends on
what happened before it.
"""
from __future__ import annotations

import pytest

from ocrmd.scene import MAJOR, Outline, OutlineTracker, Scene

STEADY = Outline(0.50, 0.50, 0.40, 0.60)
NUDGED = Outline(0.51, 0.50, 0.40, 0.60)     # within tolerance: sensor noise, not a moved page
MOVED = Outline(0.62, 0.55, 0.40, 0.60)      # a page turn


def scene(**kw) -> Scene:
    kw.setdefault("stable_frames", 3)
    kw.setdefault("settle_frames", 10)
    kw.setdefault("window", 3)
    return Scene(**kw)


def quiet(sc: Scene, n: int, outline: Outline | None = STEADY):
    """`n` undisturbed frames with the page where it is. Returns the verdicts."""
    out = []
    for _ in range(n):
        sc.note_outline(outline)
        out.append(sc.observe(moving=False, adjusting=False, occluded=False))
    return out


def settle(sc: Scene, outline: Outline | None = STEADY, limit: int = 40):
    """Run quiet frames until the scene settles. A cold scene has no outline history, which counts as
    a major disturbance, so it always serves the long wait once at the start of a run."""
    for i in range(limit):
        if quiet(sc, 1, outline)[-1].settled:
            return i + 1
    raise AssertionError("scene never settled")


# --- the outline tracker ------------------------------------------------------------------------
def test_an_outline_seen_once_is_not_yet_evidence_of_stillness():
    """The frame after a page lands looks exactly like the frame before it starts to move."""
    t = OutlineTracker(window=3, tolerance=0.03)
    assert t.settled                       # nothing reported: nothing to say, so nothing is held
    t.update(STEADY)
    assert not t.settled
    t.update(STEADY)
    assert not t.settled
    t.update(STEADY)
    assert t.settled and t.drift == 0.0


def test_movement_inside_the_tolerance_is_not_movement():
    t = OutlineTracker(window=3, tolerance=0.03)
    for _ in range(2):
        t.update(STEADY); t.update(NUDGED)
    assert t.settled and 0 < t.drift <= 0.03


def test_a_page_that_moves_unsettles_the_window_until_it_has_scrolled_out():
    t = OutlineTracker(window=3, tolerance=0.03)
    for _ in range(3):
        t.update(STEADY)
    assert t.settled
    t.update(MOVED)
    assert not t.settled and t.drift > 0.03
    t.update(MOVED)
    assert not t.settled                    # the old position is still in the window
    t.update(MOVED)
    assert t.settled                        # three detections in the new position


def test_a_page_appearing_or_vanishing_is_the_largest_change_there_is():
    t = OutlineTracker(window=3, tolerance=0.03)
    t.update(STEADY); t.update(None); t.update(STEADY)
    assert not t.settled and t.drift == 1.0


def test_a_detector_that_consistently_finds_nothing_is_stable():
    """A page filling the frame edge to edge is never detected as a bright blob. Refusing to ever
    capture on such a rig would be worse than the problem this gate solves."""
    t = OutlineTracker(window=3, tolerance=0.03)
    for _ in range(3):
        t.update(None)
    assert t.settled and t.drift == 0.0


# --- the scene ----------------------------------------------------------------------------------
def test_a_still_desk_settles_and_fires():
    sc = scene()
    v = quiet(sc, 3)
    assert [x.blocker for x in v] == ["page moved", "page moved", "settling"]   # window filling up
    v = quiet(sc, 10)
    assert v[-1].settled and v[-1].describe() == "settled"


def test_the_shutter_is_held_through_a_page_turn():
    """The failure this was rebuilt for. Halfway through a turn the pixels go quiet for a moment, the
    lens has not moved and the frame is certainly novel — every per-frame test passes, and the old
    gate fired. The page outline is what says the page is not where it will end up."""
    sc = scene()
    settle(sc)                                           # settled on the page we are leaving

    # a hand comes in and starts the turn. It is reported as "moving" while it travels and as "hand"
    # once it comes to rest over the page — which is the moment the old gate had nothing left to say.
    sc.note_outline(STEADY)
    assert sc.observe(moving=True, adjusting=False, occluded=True).blocker == "moving"
    assert sc.observe(moving=False, adjusting=False, occluded=True).blocker == "hand"
    # the page is mid-fall: still moving, and the outline is somewhere between the two positions
    mid = Outline(0.56, 0.52, 0.40, 0.60)
    for _ in range(3):
        sc.note_outline(mid)
        sc.observe(moving=False, adjusting=False, occluded=False)
    # ...and here is the trap: the scene goes completely quiet while the outline is still wrong
    held = [v for v in quiet(sc, 4, MOVED)]
    assert all(not v.settled for v in held), [v.describe() for v in held]
    assert held[0].blocker == "page moved"               # the outline window still holds `mid`

    # only after the page has held its new position, *and* the long settle has elapsed, may it fire
    v = quiet(sc, 12, MOVED)
    assert not v[3].settled and v[3].disturbance == "hand"
    assert v[-1].settled
    # the shutter stayed shut for the whole of the long settle, not the short one
    assert not any(x.settled for x in v[:6]) and all(x.needed == 10 for x in v[:6])


def test_a_hand_demands_the_long_settle_and_ordinary_noise_does_not():
    for disturbance, needed in (({"occluded": True}, 10), ({"moving": True}, 3)):
        sc = scene()
        settle(sc)
        sc.note_outline(STEADY)
        sc.observe(**{"moving": False, "adjusting": False, "occluded": False, **disturbance})
        v = quiet(sc, needed - 1)
        assert not v[-1].settled, (disturbance, v[-1].describe())
        assert v[-1].needed == needed
        assert quiet(sc, 1)[-1].settled, disturbance


def test_the_worst_disturbance_in_a_run_is_the_one_that_counts():
    """A page turn that ends with a moment of ordinary jitter is still a page turn. Remembering only
    the last blocker would let it off with the short wait."""
    sc = scene()
    settle(sc)
    sc.note_outline(STEADY)
    sc.observe(moving=False, adjusting=False, occluded=True)        # hand: major
    sc.observe(moving=True, adjusting=False, occluded=False)        # then merely moving
    v = quiet(sc, 3)
    assert v[-1].disturbance == "hand" and v[-1].needed == 10 and not v[-1].settled


def test_a_capture_spends_the_quiet_it_took_to_earn_it():
    """One settled period, one capture: a scene still being handled cannot produce a burst."""
    sc = scene()
    settle(sc)
    sc.captured()
    assert not quiet(sc, 1)[-1].settled
    assert quiet(sc, 2)[-1].settled


def test_reset_forgets_the_scene():
    sc = scene()
    settle(sc)
    sc.observe(moving=False, adjusting=False, occluded=True)
    sc.reset()
    assert sc.quiet == 0 and sc.disturbance == "" and sc.outlines.settled


@pytest.mark.parametrize("blocker", MAJOR)
def test_every_major_disturbance_demands_the_long_settle(blocker):
    sc = scene()
    settle(sc)
    if blocker == "hand":
        sc.note_outline(STEADY); sc.observe(moving=False, adjusting=False, occluded=True)
    else:
        sc.note_outline(MOVED); sc.observe(moving=False, adjusting=False, occluded=False)
    assert sc.needed == 10


def test_the_verdict_explains_itself():
    sc = scene()
    settle(sc)
    sc.note_outline(STEADY)
    sc.observe(moving=False, adjusting=False, occluded=True)
    assert sc.observe(moving=False, adjusting=False, occluded=False).describe() == "settling after hand (1/10)"


def test_outline_is_read_from_an_opencv_rect_regardless_of_how_it_names_the_sides():
    """minAreaRect calls the same sheet (w, h, -90) or (h, w, 0). A gate that saw those as different
    shapes would never settle on a page that never moved."""
    a = Outline.from_rect(((960.0, 540.0), (800.0, 1200.0), -90.0), (1080, 1920, 3))
    b = Outline.from_rect(((960.0, 540.0), (1200.0, 800.0), 0.0), (1080, 1920, 3))
    assert a.drift(b) == 0.0
    assert a.cx == 0.5 and a.cy == 0.5
