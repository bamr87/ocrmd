"""Resolving a page's doubtful lines across the focus bracket's exposures."""
from __future__ import annotations

from ocrmd import multifocus
from ocrmd.multifocus import FocusReading, MultiFocus, resolve
from ocrmd.ocr import Block, OcrResult


def line(text, conf, y, x=0.1, w=0.8, h=0.03, engine="apple_vision"):
    return Block(text=text, confidence=conf, bbox=[x, y, w, h], engine=engine)


def result(blocks: list[Block]) -> OcrResult:
    return OcrResult(text="\n".join(b.text for b in blocks), blocks=blocks, engine="apple_vision@26",
                     mean_confidence=sum(b.confidence for b in blocks) / max(1, len(blocks)),
                     words=[(w, b.confidence) for b in blocks for w in b.text.split()])


def variant(label: str, blocks: list[Block]):
    return (FocusReading(label=label, lines=len(blocks)), result(blocks))


def test_a_better_focused_reading_replaces_a_doubtful_line():
    primary = result([line("The quick brown fox", 0.95, 0.1), line("jumpod ovcr thc lazy dog", 0.42, 0.2)])
    v = variant("v1", [line("The quick brown fox", 0.93, 0.1), line("jumped over the lazy dog", 0.96, 0.2)])
    out, mf = resolve(primary, [v])
    assert [b.text for b in out.blocks] == ["The quick brown fox", "jumped over the lazy dog"]
    assert out.blocks[1].focus == "v1" and out.blocks[1].confidence == 0.96
    assert mf.improved == 1 and mf.unresolved == []
    # the per-word confidences follow the swapped text, not the reading it replaced
    assert ("jumped", 0.96) in out.words and not any(w == "jumpod" for w, _ in out.words)


def test_a_confident_line_is_never_second_guessed():
    primary = result([line("The quick brown fox", 0.95, 0.1)])
    out, mf = resolve(primary, [variant("v1", [line("A quicker browner fax", 0.99, 0.1)])])
    assert out.blocks[0].text == "The quick brown fox" and mf.improved == 0 and not mf.unresolved


def test_a_marginal_improvement_is_not_worth_a_swap():
    primary = result([line("jumpod ovcr thc lazy dog", 0.60, 0.2)])
    out, mf = resolve(primary, [variant("v1", [line("jumped over the lazy dog", 0.66, 0.2)])], margin=0.10)
    assert out.blocks[0].text.startswith("jumpod") and mf.improved == 0
    # below trust either way, so the region is handed on with both readings
    assert len(mf.unresolved) == 1
    assert {c.source for c in mf.unresolved[0].candidates} == {"focus", "v1"}
    assert mf.unresolved[0].best.confidence == 0.66


def test_a_reading_of_the_wrong_length_is_rejected():
    """A variant that read three words where the page has twelve is not a better reading of the line."""
    primary = result([line(" ".join(["word"] * 12), 0.5, 0.2)])
    out, mf = resolve(primary, [variant("v1", [line("only three words", 0.99, 0.2)])])
    assert out.blocks[0].text.startswith("word word") and mf.improved == 0
    assert len(mf.unresolved) == 1                      # still doubtful, so still reported


def test_a_confident_full_reading_replaces_a_fragment():
    """What an out-of-focus band actually produces is a fragment, not a misread word. A symmetric
    length window would reject exactly the readings the bracket exists to recover."""
    primary = result([line("milies", 0.5, 0.4)])
    v = variant("v1", [line("is so well fixed in the minds of the surrounding families", 1.0, 0.4)])
    out, mf = resolve(primary, [v])
    assert out.blocks[0].text.startswith("is so well fixed") and out.blocks[0].focus == "v1"
    assert mf.improved == 1 and not mf.unresolved
    # ...but only when the longer reading is certain
    out2, mf2 = resolve(primary, [variant("v1", [line("is so well fixed in the minds", 0.88, 0.4)])])
    assert out2.blocks[0].text == "milies" and mf2.improved == 0 and len(mf2.unresolved) == 1


def test_one_variant_line_cannot_be_spent_on_two_page_lines():
    """Variant crops carry a little geometric jitter, so one variant line can overlap two of the
    page's. Without a guard its text would be written twice."""
    primary = result([line("frag", 0.4, 0.30, h=0.05), line("ment", 0.4, 0.33, h=0.05)])
    v = variant("v1", [line("the whole line read properly", 1.0, 0.31, h=0.05)])
    out, mf = resolve(primary, [v])
    assert [b.text for b in out.blocks] == ["the whole line read properly", "ment"]
    assert mf.improved == 1 and out.text.count("the whole line") == 1


def test_a_line_only_another_focus_saw_is_filled_in_when_it_is_certain():
    primary = result([line("first line here", 0.95, 0.1)])
    v = variant("v1", [line("first line here", 0.95, 0.1),
                       line("a line the capture missed", 0.97, 0.5),
                       line("a doubtful extra reading", 0.7, 0.7),    # below CONFIRMED: not taken
                       line("noise", 0.99, 0.9)])                     # one word: not worth the risk
    out, mf = resolve(primary, [v])
    assert [b.text for b in out.blocks] == ["first line here", "a line the capture missed"]
    assert out.blocks[1].focus == "v1" and mf.filled == 1
    assert out.text.endswith("a line the capture missed")


def test_lines_are_matched_by_position_and_the_result_stays_in_reading_order():
    primary = result([line("alpha beta gamma", 0.4, 0.6), line("delta epsilon zeta", 0.95, 0.1)])
    v = variant("v1", [line("delta epsilon zeta", 0.9, 0.1), line("alpha beta gamma", 0.98, 0.6)])
    out, mf = resolve(primary, [v])
    assert [b.text for b in out.blocks] == ["delta epsilon zeta", "alpha beta gamma"]
    assert mf.improved == 1 and out.blocks[1].focus == "v1"


def test_the_best_of_several_focus_settings_wins():
    primary = result([line("blurrod tox", 0.3, 0.2)])
    out, mf = resolve(primary, [variant("v1", [line("blurred tex", 0.62, 0.2)]),
                                variant("v2", [line("blurred text", 0.94, 0.2)])])
    assert out.blocks[0].text == "blurred text" and out.blocks[0].focus == "v2"
    assert mf.improved == 1 and not mf.unresolved


def test_no_variants_is_a_no_op():
    primary = result([line("some text here", 0.3, 0.2)])
    out, mf = resolve(primary, [])
    assert out is primary and not mf.useful and mf.variants == []


def test_words_are_rebuilt_even_when_the_flat_list_does_not_line_up():
    """An OCR artifact whose `words` do not match its blocks (an older version, a hand edit) must not
    silently attach the wrong confidence to a word."""
    primary = result([line("alpha beta", 0.4, 0.1), line("gamma delta", 0.95, 0.2)])
    primary.words = [("alpha", 0.4)]                   # truncated: counts no longer match
    out, _ = resolve(primary, [variant("v1", [line("alpha bela", 0.99, 0.1)])])
    assert out.words == [("alpha", 0.99), ("bela", 0.99), ("gamma", 0.95), ("delta", 0.95)]


# --- what reaches the model ----------------------------------------------------------------------
def _unresolved_analysis() -> MultiFocus:
    primary = result([line("recerved for tho autbor", 0.44, 0.3)])
    _, mf = resolve(primary, [variant("v1", [line("reserved for the author", 0.51, 0.3)]),
                              variant("v2", [line("rescrved for the autbor", 0.48, 0.3)])])
    return mf


def test_the_variant_holding_the_best_candidate_is_the_one_sent():
    mf = _unresolved_analysis()
    assert len(mf.unresolved) == 1
    assert multifocus.variant_ranking(mf) == ["v1", "v2"]


def test_the_prompt_names_the_extra_images_and_labels_the_candidates_as_guesses():
    mf = _unresolved_analysis()
    text = multifocus.describe(mf, ["v1"])
    assert "1 more exposure(s) of the SAME page" in text and "v1" in text
    assert "never transcribe them as separate pages" in text
    assert "low-confidence machine guesses" in text and "NOT ground truth" in text
    assert 'v1: "reserved for the author" (0.51)' in text
    assert "x=0.10 y=0.30" in text                      # where to look, as a fraction of the page

    # a page with nothing unresolved and nothing attached has nothing to say
    assert multifocus.describe(MultiFocus(variants=[FocusReading(label="v1")]), []) == ""


def test_the_region_list_is_capped_and_says_so():
    mf = MultiFocus(unresolved=_unresolved_analysis().unresolved * 20)
    text = multifocus.describe(mf, [], limit=5)
    assert text.count("  - x=") == 5 and "... and 15 more" in text
