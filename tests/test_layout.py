"""The page model: Markdown into blocks, blocks onto measured lines, figures to captions, and the
order a person would read it in."""
from __future__ import annotations

import cv2
import numpy as np

from ocrmd import layout, pipeline
from ocrmd.ocr import Block as OcrBlock, OcrResult
from test_reconstruct import result

MD = ("# Chapter One\n\n"
      "the quick brown fox jumps over the lazy dog\nand then it runs away across the field\n\n"
      "- first item\n- second item\n\n"
      "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
      "> **Figure:** a fox in a field\n\n"
      "$$ e = mc^2 $$\n\n"
      "![a picture](../figures/abc-1.png)\n\n"
      "Some ==[doubtful]==[^ocrmd-1] text here.\n\n"
      '[^ocrmd-1]: ocr="doubtfull" llm="doubtful" — unresolved\n'
      "[^2]: a real footnote\n")


# --- the Markdown side -----------------------------------------------------------------------------
def test_markdown_splits_into_blocks_with_roles():
    roles = [b.role for b in layout.parse_markdown(MD)]
    assert roles == ["heading", "paragraph", "list", "table", "caption", "math", "figure", "paragraph", "footnote"]
    blocks = layout.parse_markdown(MD)
    assert blocks[0].level == 1
    assert blocks[3].raw_words == ["a", "b", "1", "2"]           # table pipes are markup, cells are words
    assert blocks[5].words == [] and blocks[6].words == []       # math and images carry no alignable words
    assert blocks[7].raw_words == ["Some", "doubtful", "text", "here."]


def test_markdown_words_drop_the_pipelines_own_footnotes_but_keep_real_ones():
    words = [w for w, _ in layout.markdown_words(MD)]
    assert "doubtfull" not in words and "doubtful" in words
    assert "footnote" not in words                               # footnotes are their own block, not body


# --- the join ---------------------------------------------------------------------------------------
def test_lines_group_into_the_blocks_their_words_came_from():
    res, img = result()
    md = "# Chapter One\n\nthe quick brown fox jumps over the lazy dog\nand then it runs away across the field\nwhile nobody is watching at all\n"
    lay = layout.build(res, md, (img.shape[1], img.shape[0]), page_id="p1")
    assert [b.role for b in lay.blocks] == ["heading", "paragraph"]
    assert lay.blocks[0].level == 1 and len(lay.blocks[1].lines) == 3
    assert lay.blocks[1].bbox[1] > lay.blocks[0].bbox[1]        # geometry is the union of the lines
    assert [b.order for b in lay.blocks] == [0, 1]
    assert lay.columns == 1 and lay.page_id == "p1"
    assert all(b.source == "llm" for b in lay.blocks)


def test_a_block_ocr_never_read_is_kept_without_geometry():
    res, img = result()
    md = "\n".join(["Chapter One", "the quick brown fox jumps over the lazy dog",
                    "and then it runs away across the field", "while nobody is watching at all"])
    md += "\n\n| col | val |\n|---|---|\n| x | 9 |\n"
    lay = layout.build(res, md, (img.shape[1], img.shape[0]))
    table = next(b for b in lay.blocks if b.role == "table")
    assert table.bbox == [0.0, 0.0, 0.0, 0.0] and table.lines == []
    assert table.order == max(b.order for b in lay.blocks)       # last: it has nowhere to go


def _two_column(n=8):
    blocks = []
    for i in range(n):
        y = 0.2 + i * 0.05
        blocks.append(OcrBlock(text=f"left line {i} of the text", confidence=0.9, bbox=[0.08, y, 0.38, 0.03]))
        blocks.append(OcrBlock(text=f"right line {i} of the text", confidence=0.9, bbox=[0.54, y, 0.38, 0.03]))
    head = OcrBlock(text="A heading across both", confidence=0.9, bbox=[0.08, 0.1, 0.84, 0.04])
    return OcrResult(text="", blocks=[head] + blocks, mean_confidence=0.9, engine="test")


def test_two_columns_are_detected_and_read_column_by_column():
    """Sorting by y alone interleaves the columns line by line, which is how the diff scored two correct
    transcriptions of a contents page at 0 %. The model has to know where the gutter is."""
    res = _two_column()
    lines = [layout.Line(b.text, list(b.bbox)) for b in res.blocks]
    assert layout.detect_columns(lines) == 2
    lay = layout.build(res, "", (1000, 1400))
    assert lay.columns == 2
    texts = [ln.text for ln in lay.lines]
    assert texts[0].startswith("A heading")
    left = [t for t in texts if t.startswith("left")]; right = [t for t in texts if t.startswith("right")]
    assert texts[1:1 + len(left)] == left and texts[1 + len(left):] == right


def test_a_full_width_block_closes_the_band_above_it():
    res = _two_column(4)
    mid = OcrBlock(text="A second heading across both", confidence=0.9, bbox=[0.08, 0.45, 0.84, 0.04])
    res.blocks.append(mid)
    for i in range(4):
        y = 0.55 + i * 0.05
        res.blocks.append(OcrBlock(text=f"lower left {i} of the text", confidence=0.9, bbox=[0.08, y, 0.38, 0.03]))
        res.blocks.append(OcrBlock(text=f"lower right {i} of the text", confidence=0.9, bbox=[0.54, y, 0.38, 0.03]))
    lay = layout.build(res, "", (1000, 1400))
    texts = [ln.text for ln in lay.lines]
    i = texts.index("A second heading across both")
    assert all(t.startswith(("left", "right")) for t in texts[1:i])
    assert all(t.startswith(("lower left", "lower right")) for t in texts[i + 1:])


# --- figures and captions --------------------------------------------------------------------------
def _page_with_drawing():
    res, img = result()
    cv2.circle(img, (600, 800), 90, (20, 20, 20), 6)
    cv2.line(img, (510, 800), (690, 800), (20, 20, 20), 6)
    return res, img


def test_a_caption_finds_its_figure():
    res, img = _page_with_drawing()
    md = "\n".join(b.text for b in res.blocks) + "\n\n> **Figure:** a circle with a bar\n"
    lay = layout.assemble(res, md, img)
    assert len(lay.figures) == 1
    f = lay.figures[0]
    assert f.caption == "Figure: a circle with a bar"
    cap = next(b for b in lay.blocks if b.role == "caption")
    assert cap.caption_of == f.id


def test_a_figure_between_two_paragraphs_is_referenced_between_them():
    res, img = _page_with_drawing()
    lay = layout.assemble(res, "\n".join(b.text for b in res.blocks), img, page_id="p")
    lay.figures[0].asset = "p-1.png"
    body = "Chapter One\nthe quick brown fox jumps over the lazy dog\nand then it runs away across the field\nwhile nobody is watching at all\n"
    out = pipeline.insert_figure_refs(body, lay)
    assert out.count("![figure](../figures/p-1.png)") == 1
    assert out.index("watching at all") < out.index("![figure]")   # the drawing is below the text
    assert pipeline.insert_figure_refs(out, lay) == out               # idempotent


def test_the_model_round_trips_through_json():
    res, img = _page_with_drawing()
    lay = layout.assemble(res, "# Chapter One\n\nthe quick brown fox jumps over the lazy dog", img, page_id="p")
    back = layout.Layout.model_validate_json(lay.model_dump_json())
    assert back == lay
    assert back.lines[0].heading == 1 and back.figures[0].role == "figure"


def test_a_persons_layout_decisions_survive_a_rebuild():
    res, img = _page_with_drawing()
    lay = layout.assemble(res, "\n".join(b.text for b in res.blocks), img, page_id="p")
    old = lay.model_copy(deep=True)
    f = old.figures[0]
    f.bbox = [f.bbox[0] - 0.02, f.bbox[1] - 0.02, f.bbox[2] + 0.04, f.bbox[3] + 0.04]
    f.caption, f.source = "hand-typed caption", "human"
    fresh = layout.assemble(res, "\n".join(b.text for b in res.blocks), img, page_id="p")
    kept = layout.merge_human_layout(fresh, old)
    assert kept == [f.id]
    assert fresh.figures[0].caption == "hand-typed caption" and fresh.figures[0].bbox == f.bbox
    assert len(fresh.figures) == 1                                    # replaced, not duplicated


def test_lines_flat_and_figures_setter_keep_the_facsimile_writer_working():
    lay = layout.Layout(width=100, height=100, lines=[layout.Line("one", [0.1, 0.1, 0.8, 0.05]),
                                                     layout.Line("two", [0.1, 0.2, 0.8, 0.05])])
    assert [ln.text for ln in lay.lines] == ["one", "two"]
    lay.figures = [[0.1, 0.5, 0.3, 0.3]]
    assert len(lay.figures) == 1 and lay.figures[0].bbox == [0.1, 0.5, 0.3, 0.3]
    assert isinstance(np.median([b.order for b in lay.blocks]), float)
