"""Watching the lens in the image, because the driver will not say where it is."""
from __future__ import annotations

from ocrmd import focus


def steady(w: focus.Watch, value: float, n: int, t0: float = 0.0) -> focus.Verdict:
    v = focus.Verdict()
    for i in range(n):
        v = w.observe(value, now=t0 + i * 0.03)
    return v


# --- is the lens moving? ---------------------------------------------------------------------------
def test_sensor_noise_is_not_a_hunt():
    """Measured on a still, in-focus rig: the rolling 12-frame spread was p99 0.026, max 0.027 over
    598 frames. Anything the band calls movement at that level would block the shutter forever."""
    w = focus.Watch(best=1000.0)
    v = focus.Verdict()
    for i in range(60):
        v = w.observe(1000 + (i % 3) * 12, now=i * 0.03)          # ~2.4 % peak to peak
    assert not v.hunting and v.spread < focus.HUNT_BAND


def test_a_lens_travelling_is_a_hunt():
    w = focus.Watch(best=1000.0)
    v = focus.Verdict()
    for i in range(focus.WINDOW):
        v = w.observe(400 + i * 60, now=i * 0.03)                 # sharpness climbing as the lens moves
    assert v.hunting and v.spread > focus.HUNT_BAND


def test_the_camera_saying_so_is_believed_on_top():
    """`isAdjustingFocus` was false on all 599 frames measured on this rig, so it cannot be the only
    signal — but a camera that does report it is worth listening to."""
    w = focus.Watch(best=1000.0)
    assert w.observe(1000.0, adjusting=True).hunting


def test_a_hand_is_not_the_lens():
    """Skin is sharper than paper; feeding that to the trace would read as a hunt and then as a
    recovery when it leaves."""
    w = focus.Watch(best=1000.0)
    steady(w, 1000.0, 20)
    v = w.observe(4000.0, occluded=True)
    assert not v.hunting and not v.soft
    assert list(w.samples) == []                                   # the window is dropped, not polluted


# --- has the page gone soft? ------------------------------------------------------------------------
def test_a_working_rig_never_looks_soft():
    """Over 20 s of a real in-focus rig the ratio to its own best never fell below 0.939."""
    w = focus.Watch(best=1100.0)
    v = steady(w, 1033.0, 200)
    assert not v.soft and v.action is None


def test_a_session_that_starts_out_of_focus_is_caught_and_refocused():
    """The measured failure: the same book at sharpness 18 where the rig had managed 1030 minutes
    earlier, holding there through a mode toggle, with the camera reporting focus=continuous."""
    w = focus.Watch(best=1030.0, soft_frames=10)
    assert steady(w, 18.0, 9).action is None                       # not yet: one soft frame proves nothing
    v = w.observe(18.0, now=1.0)
    assert v.soft and v.action == "retrigger" and "18" in v.note


def test_a_retrigger_is_given_time_to_work_before_the_next_one():
    w = focus.Watch(best=1000.0, soft_frames=1, recovery_gap=6.0)
    assert w.observe(10.0, now=100.0).action == "retrigger"
    assert w.observe(10.0, now=101.0).action is None               # inside the gap
    assert w.observe(10.0, now=110.0).action == "retrigger"        # second attempt, then it escalates
    assert w.observe(10.0, now=120.0).action == "reopen"


def test_recovery_escalates_and_then_gives_up_honestly():
    w = focus.Watch(best=1000.0, soft_frames=1, recovery_gap=0.0)
    got = [w.observe(10.0, now=float(i)).action for i in range(6)]
    assert got[:5] == ["retrigger", "retrigger", "reopen", "reopen", "accept"]
    assert w.accepted and w.best == 10.0
    assert steady(w, 10.0, 30).action is None                      # and stops nagging about it


def test_a_page_that_gets_sharper_moves_the_bar_up():
    w = focus.Watch(best=500.0)
    steady(w, 900.0, 5)
    assert w.best == 900.0


# --- the floor the shutter uses ---------------------------------------------------------------------
def test_the_floor_is_anchored_to_the_rig_not_only_to_the_session():
    """Without the anchor, a session that starts soft lowers its own bar: `best_sharp` is seeded by
    the first frames and decays, so every later frame clears it."""
    w = focus.Watch(best=1000.0)
    assert w.floor(absolute=60.0, session_best=40.0, ratio=0.55) == 550.0
    assert w.floor(absolute=60.0, session_best=2000.0, ratio=0.55) == 1100.0    # the session may raise it


def test_an_accepted_rig_stops_being_held_to_its_old_best():
    w = focus.Watch(best=1000.0, soft_frames=1, recovery_gap=0.0)
    for i in range(6):
        w.observe(10.0, now=float(i))
    assert w.accepted
    assert w.floor(absolute=60.0, session_best=40.0, ratio=0.55) == 60.0


# --- the rig's memory -------------------------------------------------------------------------------
def test_the_best_survives_the_session_and_does_not_disturb_the_camera_view(root):
    from ocrmd.camview import CameraView, load_view, save_view
    save_view(root.root, "C525", CameraView(rotation=90, zoom=2.0))
    focus.save_best(root.root, "C525", 1042.7)
    assert focus.load_best(root.root, "C525") == 1042.7
    assert load_view(root.root, "C525").rotation == 90              # the view is still readable
    save_view(root.root, "C525", CameraView(rotation=180))
    assert focus.load_best(root.root, "C525") == 1042.7             # ...and writing it keeps the best


def test_an_unknown_camera_has_no_memory_and_no_opinion(root):
    assert focus.load_best(root.root, "never seen") == 0.0
    w = focus.Watch(best=0.0)
    v = steady(w, 12.0, 100)
    assert not v.soft and v.action is None and w.best == 12.0
