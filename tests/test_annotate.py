"""Keeping the Markdown in step with what someone writes on the paper."""
from __future__ import annotations

import cv2
import pytest

from conftest import text_page
from ocrmd import annotate
from ocrmd.ocr import Block


def page_and_blocks(n_lines: int = 10):
    """A page plus the line boxes a scan of it would have left in the store."""
    lines = [f"line {i} the quick brown fox jumps over the lazy dog" for i in range(n_lines)]
    img = text_page(lines, size=(1200, 1000))
    blocks, y = [], 200
    for text in lines:
        blocks.append(Block(text=text, confidence=0.95, engine="apple_vision",
                            bbox=[120 / 1200, (y - 34) / 1000, 900 / 1200, 48 / 1000]))
        y += 70
    return img, blocks


def highlight(img, block, colour=(70, 240, 250)):
    out = img.copy()
    h, w = out.shape[:2]
    x, y, bw, bh = block.bbox
    ov = out.copy()
    cv2.rectangle(ov, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), colour, -1)
    return cv2.addWeighted(ov, 0.45, out, 0.55, 0)


def rule(img, block, at: float, thickness: int = 4):
    """A pen line across a block, `at` as a fraction of the block height (>1 = below it)."""
    out = img.copy()
    h, w = out.shape[:2]
    x, y, bw, bh = block.bbox
    py = int((y + bh * at) * h)
    cv2.line(out, (int(x * w), py), (int((x + bw) * w), py), (25, 25, 25), thickness)
    return out


def nudge(img, angle: float = 0.5, dx: int = 4, dy: int = 3):
    """As if the book was touched between the two views."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.003)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


# --- alignment ------------------------------------------------------------------------------------
def test_a_nudged_page_registers():
    img, _ = page_and_blocks()
    aligned, cc = annotate.align(nudge(img), img)
    assert aligned is not None and cc > 0.9


def test_a_different_page_does_not_register_as_the_same_one():
    img, _ = page_and_blocks()
    other = text_page(["completely different content on every single line here"] * 10, size=(1200, 1000))
    aligned, cc = annotate.align(other, img)
    assert aligned is None or cc < 0.9


def test_an_unchanged_page_costs_nothing():
    """The whole design rests on this: looking at a page nobody has written on does no work."""
    img, blocks = page_and_blocks()
    d = annotate.compare(img, nudge(img, 0.3, 2, 1), blocks)
    assert d.usable and d.marks == [] and d.needs_llm() == []


# --- what the marks mean ---------------------------------------------------------------------------
def test_a_highlighter_is_found_by_colour_not_by_darkness():
    """A highlighter brightens as often as it darkens; luminance alone misses it."""
    img, blocks = page_and_blocks()
    target = blocks[4]
    d = annotate.compare(img, nudge(highlight(img, target)), blocks)
    assert d.usable
    hits = [m for m in d.marks if m.kind == "highlight"]
    assert hits, [m.describe() for m in d.marks]
    assert "line" in " ".join(hits[0].words) and hits[0].colour


def test_a_highlight_does_not_pick_up_the_neighbouring_lines():
    img, blocks = page_and_blocks()
    target = blocks[4]
    d = annotate.compare(img, nudge(highlight(img, target)), blocks)
    words = " ".join(w for m in d.marks if m.kind == "highlight" for w in m.words)
    assert "line 4" in words
    assert "line 3" not in words and "line 5" not in words, words


def test_a_rule_below_a_line_underlines_it_and_one_through_it_strikes_it_out():
    img, blocks = page_and_blocks()
    for at, kind, idx in ((1.15, "underline", 3), (0.45, "strike", 6)):
        d = annotate.compare(img, nudge(rule(img, blocks[idx], at)), blocks)
        got = [m for m in d.marks if m.kind == kind]
        assert got, (kind, [m.describe() for m in d.marks])
        assert f"line {idx}" in " ".join(got[0].words), (kind, got[0].words)


def test_handwriting_is_the_only_thing_that_needs_a_model():
    img, blocks = page_and_blocks(n_lines=8)
    scribbled = img.copy()
    cv2.putText(scribbled, "check this", (200, 960), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (25, 25, 25), 3)
    d = annotate.compare(img, nudge(scribbled), blocks)
    notes = [m for m in d.marks if m.kind == "note"]
    assert len(notes) >= 1 and d.needs_llm() == notes
    # ...and it is only the note: no printed text was re-read to get here
    assert all(m.kind == "note" for m in d.marks)


def test_a_wholly_different_view_is_refused_rather_than_applied():
    """Half the page 'changing' is a misalignment or the wrong page. Editing from that would be the
    silent corruption the rest of the pipeline exists to prevent."""
    img, blocks = page_and_blocks()
    inverted = cv2.bitwise_not(img)
    d = annotate.compare(img, inverted, blocks)
    assert not d.usable and d.reason


def test_warp_artifacts_at_the_page_edge_are_not_annotations():
    img, blocks = page_and_blocks()
    d = annotate.compare(img, nudge(img, angle=1.2, dx=9, dy=7), blocks)
    assert d.usable and not d.marks, [m.describe() for m in d.marks]


# --- word geometry ----------------------------------------------------------------------------------
def test_words_are_placed_inside_a_line_by_its_own_gaps():
    img, blocks = page_and_blocks()
    boxes = annotate.word_boxes(blocks[0], img)
    assert [w for w, _, _ in boxes] == blocks[0].text.split()
    xs = [x0 for _, x0, _ in boxes]
    assert xs == sorted(xs)                                  # left to right
    assert all(x1 > x0 for _, x0, x1 in boxes)
    assert boxes[0][1] == pytest.approx(blocks[0].bbox[0], abs=1e-6)


# --- writing it back --------------------------------------------------------------------------------
def test_marks_become_markdown():
    body = "The quick brown fox jumps over the lazy dog and then runs away."
    marks = [annotate.Mark("highlight", [0, 0, 1, 1], words=["quick", "brown", "fox"]),
             annotate.Mark("underline", [0, 0, 1, 1], words=["lazy", "dog"]),
             annotate.Mark("strike", [0, 0, 1, 1], words=["runs", "away"])]
    out, changes = annotate.apply_marks(body, marks)
    assert "==quick brown fox==" in out
    assert "**lazy dog**" in out
    assert "~~runs away~~" in out
    assert len(changes) == 3


def test_a_phrase_broken_across_a_line_is_still_found():
    body = "The quick brown\nfox jumps over the lazy dog."
    out, changes = annotate.apply_marks(body, [annotate.Mark("highlight", [0, 0, 1, 1],
                                                            words=["brown", "fox"])])
    assert "==brown\nfox==" in out and len(changes) == 1


def test_seeing_the_same_page_again_does_not_mark_it_twice():
    """The camera stays on the page; the same highlight is seen over and over."""
    body = "The quick brown fox jumps over the lazy dog."
    mark = annotate.Mark("highlight", [0, 0, 1, 1], words=["brown", "fox"])
    once, _ = annotate.apply_marks(body, [mark])
    twice, changes = annotate.apply_marks(once, [mark])
    assert twice == once and changes == []


def test_a_note_is_appended_under_its_own_heading_and_only_once():
    body = "Some page text."
    note = annotate.Mark("note", [0, 0, 1, 1], text="remember to check the source")
    out, changes = annotate.apply_marks(body, [note])
    assert "## Notes" in out and "- remember to check the source" in out and len(changes) == 1
    again, changes2 = annotate.apply_marks(out, [note])
    assert again.count("remember to check the source") == 1 and changes2 == []


def test_an_unread_note_writes_nothing():
    """A note the model could not read must not appear as an empty bullet."""
    out, changes = annotate.apply_marks("Some page text.", [annotate.Mark("note", [0, 0, 1, 1])])
    assert out == "Some page text." and changes == []


def test_a_phrase_that_is_not_in_the_body_is_skipped_quietly():
    out, changes = annotate.apply_marks("Some page text.",
                                        [annotate.Mark("highlight", [0, 0, 1, 1], words=["absent", "words"])])
    assert out == "Some page text." and changes == []
