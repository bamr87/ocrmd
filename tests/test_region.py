"""The flat rendering, and reading one selected region of a page harder than the page was read."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import on_desk, text_page
from ocrmd import pipeline, preprocess
from ocrmd.ocr import available_engine

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")
LINES = ["the quick brown fox jumps over the lazy dog",
         "unmistakable zebras quibble expressively downwind",
         "and then it runs away across the open field"]


def lamplit(img: np.ndarray, low: float = 0.35) -> np.ndarray:
    """A page under a lamp to one side: the commonest thing that makes a webcam page unlike a scan."""
    ramp = np.linspace(low, 1.0, img.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def separation(gray: np.ndarray, ink: np.ndarray) -> float:
    """Median paper minus median ink, over the pixels a clean render says are which."""
    return float(np.median(gray[~ink])) - float(np.median(gray[ink]))


# --- the flat rendering ---------------------------------------------------------------------------
def test_the_flat_rendering_is_greyscale_at_the_pages_own_size():
    img = text_page(LINES, size=(900, 700))
    out = preprocess.legible(img)
    assert out.ndim == 2 and out.dtype == np.uint8 and out.shape == img.shape[:2]


def test_a_lamp_to_one_side_is_divided_out():
    """The lit half and the shaded half of the paper end up the same brightness; that is what makes
    the page readable at a glance, and it is the step `bin` does implicitly and destructively."""
    img = lamplit(text_page(LINES, size=(1200, 900)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = preprocess.legible(img)
    left_in, right_in = np.percentile(gray[:, :200], 85), np.percentile(gray[:, -200:], 85)
    left_out, right_out = np.percentile(out[:, :200], 85), np.percentile(out[:, -200:], 85)
    assert right_in - left_in > 60                       # the input really is unevenly lit
    assert abs(right_out - left_out) < 12, (left_out, right_out)


def test_ink_and_paper_end_further_apart_than_they_started():
    clean = text_page(LINES, size=(1200, 900))
    ink = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) < 128
    lit = lamplit(clean, low=0.5)
    before = separation(cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY), ink)
    after = separation(preprocess.legible(lit), ink)
    assert after > before, (before, after)


def test_a_blank_crop_is_not_stretched_into_noise():
    """The tone curve divides by the ink-to-paper spread. On an empty margin that spread is sensor
    noise, and stretching it turns a blank crop into static."""
    rng = np.random.default_rng(0)
    blank = np.clip(np.full((200, 300, 3), 238, np.float32) + rng.normal(0, 1.5, (200, 300, 3)), 0, 255).astype(np.uint8)
    out = preprocess.legible(blank)
    assert float(out.std()) < 20, out.std()


@needs_ocr
def test_the_flat_rendering_stays_readable_to_an_engine():
    """It exists to be read — by a person, and by the engines `read_region` points at it. A rendering
    that looks cleaner but loses words would be a worse `bin`, not a better one."""
    from ocrmd import ocr
    img = text_page(LINES, size=(1400, 700))
    rec = ocr.Recognizer([available_engine()])
    plain = rec.run(img).text.lower()
    flat = rec.run(cv2.cvtColor(preprocess.legible(img), cv2.COLOR_GRAY2BGR)).text.lower()
    assert "quick brown fox" in plain                      # the fixture is legible to begin with
    assert len(flat.split()) >= 0.9 * len(plain.split()), (plain, flat)


# --- the artifact ---------------------------------------------------------------------------------
def _page(root, store, slug="flatbook"):
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(LINES)), slug, "test").pages[0]
    return store.page(pid)


@needs_ocr
def test_the_flat_file_is_written_beside_the_crop_and_rebuilt_if_deleted(root, store):
    page = _page(root, store)
    p = pipeline.flat_path(root, page["id"])
    assert p.exists()                                     # written when the page was derived
    p.unlink()
    again = pipeline.ensure_flat(root, page["id"])
    assert again is not None and again.exists()
    stamp = again.stat().st_mtime_ns
    assert pipeline.ensure_flat(root, page["id"]).stat().st_mtime_ns == stamp   # cached, not redone


def test_a_page_with_no_crop_on_disk_reports_rather_than_crashes(root, store):
    assert pipeline.ensure_flat(root, "nosuchpage") is None


# --- reading a region -----------------------------------------------------------------------------
class RegionProvider:
    """A provider that can answer about a region, and records the crop it was given."""
    name, model = "anthropic", "fake-model"

    def __init__(self, answer="the model's reading"):
        self.answer, self.crops, self.hints, self.questions = answer, [], [], []

    def read_region(self, image_bgr, ocr_hint="", question=None):
        self.crops.append(image_bgr); self.hints.append(ocr_hint); self.questions.append(question)
        return self.answer


@needs_ocr
def test_a_region_is_read_on_its_own_terms(root, store):
    page = _page(root, store)
    res = pipeline.run_ocr(root, store, page)
    line = next(b for b in res.blocks if "zebras" in b.text.lower() or "quibble" in b.text.lower())
    out = pipeline.read_region(root, store, page, list(line.bbox))
    assert out.scale > 1, out                              # a region buys enlargement a page cannot
    assert "quibble" in out.text.lower(), out.text
    assert "quick brown fox" not in out.text.lower()       # and only that region


@needs_ocr
def test_reading_a_region_changes_nothing_but_the_flat_cache(root, store):
    """It answers a question; it does not edit the book. Anything else would put an unreconciled
    single reading into the page file behind the operator's back. The one thing it may create is the
    flat rendering it reads from, which is a cache of the crop and nothing else."""
    page = _page(root, store)
    pipeline.flat_path(root, page["id"]).unlink()
    stamp = lambda d: sorted((f.name, f.stat().st_mtime_ns) for f in d.glob("*"))   # noqa: E731
    before, ocr_before = stamp(root.processed), stamp(root.ocr_dir)
    pipeline.read_region(root, store, page, [0.1, 0.1, 0.6, 0.2])
    assert stamp(root.ocr_dir) == ocr_before
    after = stamp(root.processed)
    assert [f for f, _ in after] == sorted([f for f, _ in before] + [f"{page['id']}.flat.png"])
    assert [x for x in after if not x[0].endswith(".flat.png")] == before
    assert store.page(page["id"])["status"] == "unverified"


@needs_ocr
def test_the_model_sees_the_same_crop_and_the_ocr_draft(root, store):
    page = _page(root, store)
    prov = RegionProvider()
    out = pipeline.read_region(root, store, page, [0.05, 0.1, 0.9, 0.25], prov, "what is the second word?")
    assert out.llm == "the model's reading" and out.llm_error is None
    assert prov.questions == ["what is the second word?"]
    assert prov.hints[0] == out.text                       # the hint is the reading it is checking
    assert prov.crops[0].shape[1] / max(1, out.size[0]) == pytest.approx(out.scale, abs=0.5)


@needs_ocr
def test_a_provider_that_cannot_read_regions_is_simply_not_asked(root, store):
    class Old:                                             # e.g. an older provider, or xAI
        name = model = "x"
    out = pipeline.read_region(root, store, _page(root, store), [0.1, 0.1, 0.5, 0.2], Old())
    assert out.llm is None and out.llm_error is None and out.text


def test_a_stray_click_is_refused(root, store):
    page = _page(root, store)
    with pytest.raises(ValueError, match="too small"):
        pipeline.read_region(root, store, page, [0.5, 0.5, 0.005, 0.005])


def test_a_selection_running_off_the_page_is_clamped(root, store):
    page = _page(root, store)
    crop = pipeline.region_crop(root, page["id"], [0.8, 0.9, 0.5, 0.4])
    full = cv2.imread(str(root.processed / f"{page['id']}.png"))
    assert crop is not None and crop.size and crop.shape[0] < full.shape[0] and crop.shape[1] < full.shape[1]
