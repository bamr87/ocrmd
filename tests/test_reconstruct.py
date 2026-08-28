"""The round trip: page -> Markdown -> layout -> SVG/LaTeX -> page again, and how close it came."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import text_page
from ocrmd import layout, reconstruct
from ocrmd.ocr import Block, OcrResult

LINES = ["Chapter One", "the quick brown fox jumps over the lazy dog",
         "and then it runs away across the field", "while nobody is watching at all"]


def result(lines=None, size=(1200, 1000)) -> tuple[OcrResult, np.ndarray]:
    """A page, plus line boxes measured from the ink itself — which is what a real engine returns.
    Boxes guessed from the drawing coordinates are not tight to the glyphs, and a facsimile that
    stretches each line to its box would then be judged against the wrong geometry."""
    lines = lines or LINES
    img = text_page(lines, size=size)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = g < 128
    rows = np.where(ink.any(axis=1))[0]
    bands, start = [], None
    for i in range(len(g) + 1):
        on = i < len(g) and ink[i].any()
        if on and start is None:
            start = i
        elif not on and start is not None:
            bands.append((start, i)); start = None
    assert len(bands) == len(lines), (len(bands), len(lines), rows[:5])
    blocks = []
    for text, (y0, y1) in zip(lines, bands, strict=True):
        cols = np.where(ink[y0:y1].any(axis=0))[0]
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        blocks.append(Block(text=text, confidence=0.95, engine="apple_vision",
                            bbox=[x0 / size[0], y0 / size[1], (x1 - x0) / size[0], (y1 - y0) / size[1]]))
    res = OcrResult(text="\n".join(lines), blocks=blocks, mean_confidence=0.95, engine="test",
                    words=[(w, 0.95) for b in blocks for w in b.text.split()])
    return res, img


# --- layout ---------------------------------------------------------------------------------------
def test_lines_carry_the_reconciled_words_at_the_measured_positions():
    """OCR knows where; the Markdown knows what. Neither alone can rebuild the page."""
    res, img = result()
    md = "# Chapter One\n\nthe quick brown fox jumps over the lazy dog\nand then it runs away across the field\nwhile nobody is watching at all\n"
    lay = layout.build(res, md, (img.shape[1], img.shape[0]))
    assert len(lay.lines) == 4
    assert [ln.bbox for ln in lay.lines] == [b.bbox for b in res.blocks]     # geometry from OCR
    assert lay.lines[0].heading == 1                                        # structure from Markdown
    assert all(ln.source == "reconciled" for ln in lay.lines)


def test_a_line_the_markdown_cannot_place_keeps_its_own_reading():
    res, img = result()
    lay = layout.build(res, "something else entirely unrelated to this page", (img.shape[1], img.shape[0]))
    assert all(ln.source == "ocr" for ln in lay.lines)
    assert lay.lines[1].text == LINES[1]


def test_markdown_words_drop_the_pipelines_own_footnotes():
    md = ("Some ==[doubtful]==[^ocrmd-1] text here.\n\n"
          '[^ocrmd-1]: ocr="doubtfull" llm="doubtful" — unresolved\n')
    words = [w for w, _ in layout.markdown_words(md)]
    assert words == ["Some", "doubtful", "text", "here."]


def test_one_line_seen_twice_is_drawn_once():
    """The cross-engine merge fills lines one engine missed; two engines box a row differently, so a
    page can carry two entries for it. On a reflow that is invisible, on a facsimile they overlap."""
    a = layout.Line("the quick brown fox", [0.1, 0.20, 0.75, 0.05])
    b = layout.Line("the quick brown fox jumps", [0.1, 0.205, 0.76, 0.048])
    c = layout.Line("a different line lower down", [0.1, 0.40, 0.75, 0.05])
    out = layout.dedupe([a, b, c])
    assert [x.text for x in out] == ["the quick brown fox jumps", "a different line lower down"]


# --- figures --------------------------------------------------------------------------------------
def test_a_drawing_is_found_and_the_text_block_is_not():
    res, img = result()
    cv2.circle(img, (600, 800), 90, (20, 20, 20), 6)          # a diagram below the text
    cv2.line(img, (510, 800), (690, 800), (20, 20, 20), 6)
    figs = layout.figures(img, res.blocks)
    assert len(figs) == 1, [f.bbox for f in figs]
    x, y, w, h = figs[0].bbox
    assert y > 0.6 and 0.3 < x < 0.6                          # where the drawing is, not the text


def test_the_text_block_is_never_returned_as_a_figure():
    """Erasing line boxes leaves the gaps between them; closing those up joins the whole column into
    one blob, which then gets pasted back over the reconstruction as a picture of the page."""
    res, img = result(lines=[f"line {i} of a dense block of body text here" for i in range(12)],
                      size=(1200, 1200))
    assert layout.figures(img, res.blocks) == []


def test_line_art_is_traced_and_a_photograph_is_not():
    art = np.full((200, 200, 3), 245, np.uint8)
    cv2.rectangle(art, (40, 40), (160, 160), (10, 10, 10), 5)
    assert reconstruct.looks_like_line_art(art)
    assert reconstruct.trace(art).startswith("<path")
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert not reconstruct.looks_like_line_art(photo)


# --- SVG and LaTeX ---------------------------------------------------------------------------------
def test_the_svg_places_every_line_where_it_was():
    res, img = result()
    lay = layout.build(res, "\n".join(LINES), (img.shape[1], img.shape[0]))
    svg = reconstruct.to_svg(lay)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<text") == len(lay.lines)
    assert f'width="{img.shape[1]}"' in svg
    assert "the quick brown fox jumps over the lazy dog" in svg


def test_the_svg_escapes_page_text():
    """Page text is not markup. An unescaped ampersand makes the whole document unparseable."""
    lay = layout.Layout(width=100, height=100,
                        lines=[layout.Line("Smith & Sons <sales@example.com>", [0.1, 0.1, 0.8, 0.05])])
    svg = reconstruct.to_svg(lay)
    assert "&amp;" in svg and "&lt;sales" in svg and "<sales" not in svg


def test_the_latex_is_a_facsimile_at_the_pages_own_size():
    res, img = result()
    lay = layout.build(res, "\n".join(LINES), (img.shape[1], img.shape[0]))
    tex = reconstruct.to_latex(lay, title="Chapter One")
    assert r"\documentclass{article}" in tex and r"\end{document}" in tex
    assert "paperwidth=8.000in" in tex and "paperheight=6.667in" in tex     # 1200x1000 at 150 dpi
    assert tex.count(r"\begin{textblock}") == len(lay.lines)


def test_the_latex_escapes_what_would_break_it():
    lay = layout.Layout(width=300, height=300,
                        lines=[layout.Line("100% of $5 & #1 with_underscores", [0.1, 0.1, 0.8, 0.1])])
    tex = reconstruct.to_latex(lay)
    for escaped in (r"\%", r"\$", r"\&", r"\#", r"\_"):
        assert escaped in tex
    assert reconstruct.tex_escape("a_b") == r"a\_b"


# --- the whole trip --------------------------------------------------------------------------------
needs_svg = pytest.mark.skipif(not reconstruct.backends(), reason="no SVG rasteriser installed")


def test_similarity_scores_ink_not_lighting():
    """The rebuild is black on white; the original is a photograph under a desk lamp. Correlating raw
    pixels would score the lamp."""
    _res, img = result()
    lit = np.clip(img.astype(np.float32) * np.linspace(0.5, 1.0, img.shape[1])[None, :, None], 0, 255)
    assert reconstruct.image_similarity(img, lit.astype(np.uint8)) > 0.9
    rng = np.random.default_rng(1)
    assert reconstruct.image_similarity(img, rng.integers(0, 255, img.shape, dtype=np.uint8)) < 0.2


@needs_svg
def test_a_page_rebuilds_into_something_that_looks_like_itself():
    res, img = result()
    lay = layout.build(res, "\n".join(LINES), (img.shape[1], img.shape[0]))
    lay.figures = layout.figures(img, res.blocks)
    out = reconstruct.rebuild(img, lay, title="Chapter One")
    assert out.png is not None and out.fidelity is not None
    assert out.fidelity > 0.5, out.describe()
    assert out.png.shape[:2] == img.shape[:2]


@needs_svg
def test_a_rebuild_of_the_wrong_text_scores_worse_than_the_right_one():
    """The score has to be able to tell a good reconstruction from a bad one, or it says nothing."""
    res, img = result()
    good = layout.build(res, "\n".join(LINES), (img.shape[1], img.shape[0]))
    empty = layout.Layout(width=img.shape[1], height=img.shape[0])
    a, _, _ = reconstruct.fidelity(img, reconstruct.to_svg(good))
    b, _, _ = reconstruct.fidelity(img, reconstruct.to_svg(empty))
    assert a is not None and b is not None and a > b + 0.2, (a, b)


def test_no_rasteriser_is_reported_rather_than_guessed(monkeypatch):
    monkeypatch.setattr(reconstruct, "backends", lambda: [])
    score, backend, shot = reconstruct.fidelity(np.zeros((10, 10, 3), np.uint8), "<svg/>")
    assert score is None and shot is None and "rasteriser" in backend
