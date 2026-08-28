"""Unit checks for ocrmd. Run: .venv/bin/python -m pytest tests -q  (or: python tests/test_ocrmd.py)

Every case here is a regression guard for a bug found by running the tool against a real book:
orientation, the suspect/blank statuses, order-insensitive overlap, and the strict JSON schema.
"""
from __future__ import annotations
import json, re, sys, tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocrmd import llm, preprocess                                    # noqa: E402
from ocrmd.camview import CameraView, handle_key, load_view, save_view  # noqa: E402
from ocrmd.hud import ScanState, draw as draw_hud                     # noqa: E402
from ocrmd.reconcile import reconcile, token_overlap                 # noqa: E402

LONG = ("it is a truth universally acknowledged that a single man in possession of a good fortune "
        "must be in want of a wife however little known the feelings or views of such a man may be")
WORDS = [(w, 0.95) for w in LONG.split()]


def test_camera_view_keys_and_persistence():
    v = CameraView()
    for k in "r++d":
        handle_key(v, ord(k))
    assert (v.rotation, round(v.zoom, 2)) == (90, 1.56)
    assert handle_key(v, ord("z")) is False          # unknown key is a no-op
    root = Path(tempfile.mkdtemp())
    save_view(root, "cam", v)
    assert load_view(root, "cam") == v
    assert load_view(root, "other") == CameraView()  # unknown camera falls back to defaults


def test_camera_view_pan_follows_screen_not_sensor():
    """Panning "right" must move right on screen whatever the rotation."""
    a, b = CameraView(zoom=2.0), CameraView(rotation=90, zoom=2.0)
    a.pan(0.05, 0); b.pan(0.05, 0)
    assert a.cx > 0.5 and b.cy > 0.5                 # 90deg maps screen-x onto sensor-y


def test_reconcile_statuses():
    assert reconcile(LONG, WORDS, LONG).status == "verified"
    assert reconcile("", [], "").status == "blank"
    assert reconcile(LONG, WORDS, "completely different vocabulary entirely unrelated invented content "
                                  "about something else nothing whatsoever shared").status == "suspect"


def test_numeric_mismatch_is_never_auto_resolved():
    ocr_text = LONG + " total 44.60 paid"
    r = reconcile(ocr_text, [(w, 0.95) for w in ocr_text.split()], LONG + " total 44.80 paid")
    assert r.status == "flagged"
    assert any(f.chosen is None for f in r.flags)     # unresolved, not silently picked
    assert "==[" in r.markdown and "[^ocrmd-1]" in r.markdown


def test_overlap_is_order_insensitive():
    """OCR reads contents pages out of order; a sequence diff alone would call it a different page."""
    assert token_overlap(LONG.split(), list(reversed(LONG.split()))) == 1.0


def test_transcription_schema_is_strict():
    """Structured outputs reject any object without additionalProperties:false (a real 400)."""
    def strict(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node
            return all(strict(v) for v in node.values())
        return all(strict(v) for v in node) if isinstance(node, list) else True
    assert strict(llm.TRANSCRIPTION_SCHEMA)


def test_hint_cleaning_drops_garbage_lines():
    assert llm.clean_hint("Good line here\nДрли Окриле ви L co\nAnother fine line") == \
        "Good line here\nAnother fine line"


def _on_desk(page: np.ndarray, margin: int = 90) -> np.ndarray:
    """Put a page on a darker background with a margin, the way a real capture sees it."""
    h, w = page.shape[:2]
    desk = np.full((h + 2 * margin, w + 2 * margin, 3), 60, dtype=np.uint8)
    desk[margin:margin + h, margin:margin + w] = page
    return desk


def _text_page(upside_down: bool = False) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1400, 1800), (245, 242, 235))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 40)
    y = 200
    for line in LONG.split(" however ")[0].split(" that ") * 6:
        d.text((120, y), line, font=f, fill="black"); y += 70
    a = np.array(img)[:, :, ::-1].copy()
    return cv2.rotate(a, cv2.ROTATE_180) if upside_down else a


def test_orientation_detects_upside_down_text():
    for upside_down, expected in ((False, 0), (True, 180)):
        parts = preprocess.process(_on_desk(_text_page(upside_down)))
        assert {m["orientation"] for _, _, m in parts} == {expected}


def test_low_resolution_page_gets_advice():
    small = _on_desk(cv2.resize(_text_page(), (400, 514)), margin=40)
    parts = preprocess.process(small)
    assert any("move the camera closer" in (m.get("advice") or "") for _, _, m in parts)
    assert all(not m["clipped_edges"] for _, _, m in parts)


def test_text_running_past_the_frame_is_reported():
    """A page bleeding off the frame loses text; that must outrank the resolution advice."""
    page = _text_page()[:, 300:]                      # sliced through the text block
    parts = preprocess.process(page)
    assert any("left" in m["clipped_edges"] for _, _, m in parts)
    assert any("cut off" in (m.get("advice") or "") for _, _, m in parts)


def test_blank_margin_at_the_frame_edge_is_not_reported():
    """Paper touching the border is normal and harmless; only lost *text* is worth interrupting for.
    Reporting bare geometry cried wolf on nearly every real capture."""
    page = _text_page()                               # fills the frame, but margins are blank
    parts = preprocess.process(page)
    assert all(not m["clipped_edges"] for _, _, m in parts), [m["clipped_edges"] for _, _, m in parts]


def test_sharpness_prefers_the_detected_page():
    """A frame-centre crop scores whatever is central; the gate needs the page's own sharpness."""
    from ocrmd import capture
    sharp = _on_desk(_text_page())
    soft = sharp.copy()
    soft[:] = cv2.GaussianBlur(soft, (0, 0), 3)
    quad = preprocess.find_page_quad(sharp)
    assert capture.sharpness(sharp, quad=quad) > capture.sharpness(soft, quad=quad) * 3


def _spread() -> np.ndarray:
    """Two pages with a gutter shadow between them, which is what the splitter keys on."""
    page = _text_page()
    gutter = np.full((page.shape[0], 26, 3), 90, dtype=np.uint8)
    return _on_desk(np.hstack([page, gutter, page]))


def test_spread_is_split_at_the_gutter():
    """The invariant is that the cut falls in blank space, not that the halves come out the same
    width: the binding plus its two inner margins is rarely symmetric, and a split anywhere inside
    that band keeps both pages whole. Equal widths was the old assertion and it was the wrong one."""
    parts = preprocess.process(_spread())
    assert len(parts) == 2
    assert all(m["page_width_px"] > 400 for _, _, m in parts)     # neither half is an offcut
    crop = _spread()
    x, depth = preprocess.find_gutter(crop)
    prof = preprocess.ink_columns(crop)
    body = float(np.median(prof[int(0.05 * len(prof)):int(0.95 * len(prof))]))
    assert depth >= preprocess.GUTTER_DEPTH
    assert prof[x] <= body * (1 - preprocess.GUTTER_DEPTH)        # the cut runs through no text


def test_estimated_page_width_matches_what_the_pipeline_produces():
    """A single portrait page and a two-page spread have nearly the same aspect ratio, so the live
    readout must agree with the width `process` actually yields, not guess from the outline."""
    for page in (_on_desk(_text_page()), _spread()):
        rect = preprocess.find_page_rect(page)
        estimate = preprocess.estimated_page_width(page, rect)
        actual = preprocess.process(page)[0][2]["page_width_px"]
        assert abs(estimate - actual) <= max(60, 0.1 * actual), (estimate, actual)


def test_scan_state_keeps_a_bounded_history():
    st = ScanState()
    for i in range(12):
        st.note(f"{i:04d}", "flagged")
    assert len(st.recent) == 6 and st.recent[-1][0] == "0011"


def test_hud_draws_on_any_frame_shape():
    """The overlay must never raise: it runs inside the capture loop, where an exception stops the scan."""
    st = ScanState(queued=3, working="page 0002: llm", captured=7, advice="move closer", error=None)
    st.note("0001", "verified")
    for shape in ((960, 1280, 3), (720, 1280, 3), (120, 160, 3)):
        frame = np.full(shape, 120, dtype=np.uint8)
        quad = np.array([[10, 10], [shape[1] - 10, 12], [shape[1] - 12, shape[0] - 10], [12, shape[0] - 12]], "float32")
        out = draw_hud(frame, gate="READY", motion=0.6, still=8, stable_frames=8, novelty=40,
                       novelty_threshold=12, sharp=280, blur_threshold=60, view_desc="rot=0 zoom=1.00x",
                       focus="continuous", adjusting=True, resolution=(shape[1], shape[0]),
                       page_quad=quad, state=st, help_text="keys", flash=0.8)
        assert out.shape[1] == 1000 and out.dtype == np.uint8


def test_hud_survives_missing_page_and_empty_state():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    out = draw_hud(frame, gate="blurry", motion=9.0, still=0, stable_frames=8, novelty=0,
                   novelty_threshold=12, sharp=5, blur_threshold=60, view_desc="rot=0 zoom=1.00x",
                   focus="n/a", adjusting=False, resolution=(640, 480), page_quad=None,
                   state=ScanState(), help_text="keys")
    assert out.shape[1] == 1000


def test_store_connection_crosses_threads():
    """scan processes pages on a worker thread; a default connection would raise ProgrammingError."""
    import threading
    from ocrmd.store import Store
    db = Path(tempfile.mkdtemp()) / "t.db"
    st = Store(db)
    st.ensure_document("doc")
    err = []

    def work():
        try:
            st.add_page("p1", "doc", "sha", "0" * 64, "left")
        except Exception as e:                      # pragma: no cover
            err.append(e)

    t = threading.Thread(target=work); t.start(); t.join()
    assert not err, err
    assert len(st.pages("doc")) == 1


def test_launch_json_is_valid_jsonc():
    src = re.sub(r"^\s*//.*$", "", (Path(__file__).parent.parent / ".vscode/launch.json").read_text(), flags=re.M)
    cfgs = json.loads(src)["configurations"]
    assert len(cfgs) >= 9 and all("envFile" in c and c["type"] == "debugpy" for c in cfgs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} checks passed")


# --- spread splitting ----------------------------------------------------------------------------
def _binding(gap_px: int = 60, lamp: bool = False):
    """Two pages of text with a blank binding between them, optionally lit from one side."""
    import numpy as np
    from conftest import text_page
    page = text_page(["the quick brown fox jumps over the lazy dog"] * 14, size=(700, 900))
    gutter = np.full((page.shape[0], gap_px, 3), 245, np.uint8)
    img = np.hstack([page, gutter, page])
    if lamp:                                    # a smooth brightness ramp across the sheet
        ramp = np.linspace(0.45, 1.0, img.shape[1], dtype=np.float32)[None, :, None]
        img = np.clip(img * ramp, 0, 255).astype(np.uint8)
    return img


def test_a_real_spread_splits_at_its_binding():
    from ocrmd import preprocess
    img = _binding()
    x, depth = preprocess.find_gutter(img)
    assert depth >= preprocess.GUTTER_DEPTH
    assert abs(x - img.shape[1] // 2) < 0.08 * img.shape[1]     # in the blank band around the middle
    halves = preprocess.split_spread(img)
    assert len(halves) == 2
    assert min(h.shape[1] for h in halves) > 0.3 * img.shape[1]   # two pages, not a page and an offcut


def test_a_lamp_to_one_side_does_not_move_the_gutter():
    """The bug this replaced: on a global threshold the brightness ramp is the strongest signal in
    the frame, so the darkest column is the far edge of the search band, not the binding."""
    from ocrmd import preprocess
    lit = _binding(lamp=True)
    x, depth = preprocess.find_gutter(lit)
    assert depth >= preprocess.GUTTER_DEPTH
    assert abs(x - lit.shape[1] // 2) < 0.08 * lit.shape[1]

    # what the old rule would have chosen: the darkest column in the band, which the ramp decides
    import cv2, numpy as np
    gray = cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY)
    col = cv2.GaussianBlur(gray, (1, 51), 0).mean(axis=0)
    w = lit.shape[1]
    old_x = w // 3 + int(np.argmin(col[w // 3:2 * w // 3]))
    assert abs(old_x - w // 2) > abs(x - w // 2)                # the old choice is further off


def test_a_single_page_with_a_blank_stripe_is_not_cut_in_two():
    """A wide crop is not a spread, and an empty column is not a binding. Measured on the real store,
    splitting these produced a page and a 440-px offcut, and the unsplit sheet reads longer and more
    confidently. A ragged right margin or a figure leaves a stripe just as empty as a gutter — what
    tells them apart is whether there is text on both sides of it."""
    import numpy as np
    from ocrmd import preprocess
    from conftest import text_page
    page = text_page(["the quick brown fox jumps over the lazy dog"] * 14, size=(700, 900))
    blank = np.full((page.shape[0], 560, 3), 245, np.uint8)     # a wide empty right-hand region
    wide = np.hstack([page, blank])
    assert wide.shape[1] / wide.shape[0] >= preprocess.SPREAD_RATIO
    _x, depth = preprocess.find_gutter(wide)
    assert depth < preprocess.GUTTER_DEPTH                      # empty, but text on only one side
    assert len(preprocess.split_spread(wide)) == 1


def test_a_narrow_crop_is_never_considered():
    from ocrmd import preprocess
    from conftest import text_page
    assert len(preprocess.split_spread(text_page(size=(700, 900)))) == 1


def test_the_live_page_width_estimate_agrees_with_the_split():
    """The HUD must not promise 900px a page on a sheet the pipeline is about to keep whole."""
    from ocrmd import preprocess
    from conftest import on_desk
    import numpy as np
    from conftest import text_page
    page = text_page(["the quick brown fox jumps over the lazy dog"] * 14, size=(700, 900))
    lone = np.hstack([page, np.full((page.shape[0], 560, 3), 245, np.uint8)])
    for img, spread in ((_binding(), True), (lone, False)):
        frame = on_desk(img, margin=100)
        rect = preprocess.find_page_rect(frame)
        assert rect is not None
        est = preprocess.estimated_page_width(frame, rect)
        parts = preprocess.process(frame)
        assert (len(parts) == 2) == spread
        widest = max(p[0].shape[1] for p in parts)
        assert est <= widest * 1.15                  # the estimate tracks the crops actually made
