"""A PDF as a source: the text layer is a reading, the images are figures, nothing is guessed that the
file already knew."""
from __future__ import annotations
import cv2
import numpy as np
import pytest

from conftest import FakeProvider, text_page
from ocrmd import pdfsrc, pipeline, render
from ocrmd.ocr import available_engine

pytestmark = pytest.mark.skipif(not pdfsrc.available(), reason="pymupdf not installed")
needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")
LINES = ["Chapter One", "the quick brown fox jumps over the lazy dog",
         "and then it runs away across the field", "while nobody is watching at all"]


def born_digital(path, lines=LINES, picture: bool = True, pages: int = 1):
    import pymupdf
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        y = 90
        for i, ln in enumerate(lines):
            page.insert_text((72, y), ln, fontsize=22 if i == 0 else 12, fontname="helv")
            y += 28
        if picture:
            art = np.full((200, 300, 3), 255, np.uint8)
            cv2.circle(art, (150, 100), 70, (0, 0, 0), 6)
            ok, buf = cv2.imencode(".png", art)
            page.insert_image(pymupdf.Rect(150, 400, 450, 600), stream=buf.tobytes())
    doc.save(path); doc.close()
    return path


def scanned(path):
    """A PDF that is only a picture of a page: no text layer at all."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    ok, buf = cv2.imencode(".png", text_page(LINES))
    page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(path); doc.close()
    return path


def test_a_pdf_page_is_read_not_looked_at(tmp_path):
    pg = next(pdfsrc.pages(born_digital(tmp_path / "a.pdf")))
    assert pg.text_layer and [b.text for b in pg.blocks] == LINES
    assert all(b.confidence == 1.0 and b.engine == "pdf_text" for b in pg.blocks)
    assert pg.blocks[0].bbox[1] < pg.blocks[1].bbox[1] < 0.2         # positioned where they were set
    assert len(pg.images) == 1 and 0.4 < pg.images[0][1] < 0.6        # the picture, half way down
    assert pg.image.shape[1] > 2000                                    # 300 dpi: above the OCR floor


def test_a_born_digital_pdf_ingests_with_its_text_layer_as_the_reading(root, store, tmp_path):
    res = pipeline.ingest_pdf(root, store, born_digital(tmp_path / "a.pdf"), "paper")
    assert len(res) == 1 and len(res[0][1].pages) == 1
    pid = res[0][1].pages[0]
    assert pipeline.pdf_text(root, pid) is not None
    meta = pipeline.run_preprocess(root, store, store.page(pid))
    assert meta["source"] == "pdf" and meta["pdf"]["text_layer"] and len(meta["pdf"]["images"]) == 1
    r = pipeline.run_page(root, store, store.page(pid), None, "paper")
    fm, body = render.read_page(r.path)
    assert fm["source"]["type"] == "pdf" and fm["ocr"]["engine"].startswith("pdf_text")
    assert "quick brown fox" in body
    stage = store.stage(pid, "ocr")
    assert stage["version"].startswith("ocr@") and ":pdf_text" in stage["version"]
    lay = pipeline.load_layout(root, pid)
    assert len(lay.figures) == 1 and lay.figures[0].source == "pdf" and lay.figures[0].asset
    assert (pipeline.figures_dir(root, "paper") / lay.figures[0].asset).exists()
    assert "![figure](../figures/" in body
    # the same file again is an exact duplicate, not a second copy of every page
    again = pipeline.ingest_pdf(root, store, tmp_path / "a.pdf", "paper")
    assert again[0][1].duplicate_of == pid and len(store.pages("paper")) == 1


@needs_ocr
def test_the_engines_vote_on_the_text_layer(root, store, tmp_path):
    """Confidence 1.0 is what the file claims; the vote is the measurement."""
    pid = pipeline.ingest_pdf(root, store, born_digital(tmp_path / "a.pdf", picture=False), "paper")[0][1].pages[0]
    res = pipeline.run_ocr(root, store, store.page(pid))
    assert res.engines[0].startswith("pdf_text") and len(res.engines) >= 2
    assert res.agreement is not None and res.agreement > 0.7
    assert [b.text for b in res.blocks][:4] == LINES               # the layer's lines, confirmed


@needs_ocr
def test_a_scanned_pdf_falls_back_to_the_engines(root, store, tmp_path):
    pid = pipeline.ingest_pdf(root, store, scanned(tmp_path / "s.pdf"), "scan")[0][1].pages[0]
    assert pipeline.pdf_text(root, pid) is None
    r = pipeline.run_page(root, store, store.page(pid), FakeProvider(page_number="1"), "scan")
    fm, body = render.read_page(r.path)
    assert not fm["ocr"]["engine"].startswith("pdf_text") and "fox" in body


def test_the_reading_survives_without_any_engine(root, store, tmp_path, monkeypatch):
    from ocrmd import ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "available_engines", lambda: [])
    pid = pipeline.ingest_pdf(root, store, born_digital(tmp_path / "a.pdf", picture=False), "paper")[0][1].pages[0]
    res = pipeline.run_ocr(root, store, store.page(pid))
    assert res.engine.startswith("pdf_text") and res.agreement is None
    assert store.stage(pid, "ocr")["version"] == "ocr@3:pdf_text"
