"""The scan session's contract with the capture loop, with the camera replaced by a stub.

`auto_shutter` needs a real device, so nothing here opens one: the generator is swapped for a stand-in
that reports the way the real one does. Both tests are the same scan seen from the operator's side —
the first is what a soft page did to a live session (the focus watch spoke *with a level*, the callback
took one argument, the capture thread died), the second is what the UI showed afterwards (nothing).
"""
from __future__ import annotations

from ocrmd import capture
from ocrmd.session import ScanOptions, ScanSession


def _stub_camera(monkeypatch, shutter):
    monkeypatch.setattr(capture, "resolve_camera", lambda _camera: 0)
    monkeypatch.setattr(capture, "camera_name", lambda _index: "Stub Cam")
    monkeypatch.setattr(capture, "auto_shutter", shutter)


def test_scan_survives_a_status_line_that_carries_a_level(root, store, monkeypatch):
    def shutter(s, on_status=print, **_kw):
        on_status("camera 'Stub Cam' 1792x1008")                          # the one-argument form
        on_status("out of focus (ratio 0.31): retriggering", "warn")      # what the focus watch sends
        return
        yield                                                             # a generator that captures nothing

    _stub_camera(monkeypatch, shutter)
    ses = ScanSession(root, store, ScanOptions(doc="manual"), None, preview=False)
    ses.run()
    assert ses.error is None and not ses.running
    by_line = {e.line: e.level for e in ses.events}
    assert by_line["out of focus (ratio 0.31): retriggering"] == "warn"   # the level reached the event log
    assert by_line["camera 'Stub Cam' 1792x1008"] == "info"


def test_a_crash_in_the_capture_loop_is_reported_not_swallowed(root, store, monkeypatch):
    def shutter(s, **_kw):
        raise TypeError("takes 1 positional argument but 2 were given")
        yield

    _stub_camera(monkeypatch, shutter)
    ses = ScanSession(root, store, ScanOptions(doc="manual"), None, preview=False)
    ses.run()                                                             # must not raise: the thread ends cleanly
    assert not ses.running
    assert ses.error and "TypeError" in ses.error
    assert any(e.level == "error" and "TypeError" in e.line for e in ses.events)
    assert ses.snapshot()["error"] == ses.error                           # what /api/scan hands the browser
