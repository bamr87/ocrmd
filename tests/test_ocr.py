"""Engine-independent OCR logic: the cross-engine merge and the settings that drive it."""
from ocrmd import ocr
from ocrmd.config import Settings
from ocrmd.ocr import Block, merge


def line(text, conf, y, x=0.1, w=0.8, h=0.03, engine="a"):
    return Block(text=text, confidence=conf, bbox=[x, y, w, h], engine=engine)


def test_single_engine_passes_through():
    p = [line("The quick brown fox", 0.5, 0.1)]
    blocks, words, agreement = merge(p, {})
    assert blocks == p and agreement is None
    assert words == [("The", 0.5), ("quick", 0.5), ("brown", 0.5), ("fox", 0.5)]


def test_agreeing_words_are_confirmed_and_differing_words_doubted():
    p = [line("The quick brown fox", 0.7, 0.1)]
    o = {"b": [line("The quick brawn fox", 0.9, 0.105, engine="b")]}
    blocks, words, agreement = merge(p, o)
    conf = dict(words)
    assert conf["The"] >= ocr.CONFIRMED and conf["fox"] >= ocr.CONFIRMED     # both engines read them
    assert conf["brown"] == 0.7 * ocr.DOUBTED                              # the engines disagree
    assert blocks[0].agreed == 0.75 and agreement == 0.75
    assert blocks[0].text == "The quick brown fox"                         # the primary keeps its text


def test_missed_line_is_filled_only_when_confident():
    p = [line("first line here", 0.9, 0.1)]
    o = {"b": [line("first line here", 0.9, 0.1, engine="b"),
               line("second line the primary missed", 0.95, 0.2, engine="b"),
               line("noise", 0.99, 0.3, engine="b"),                       # one word: not worth the risk
               line("a doubtful extra reading", 0.5, 0.4, engine="b")]}    # below FILL_CONFIDENCE
    blocks, words, _ = merge(p, o)
    assert [b.text for b in blocks] == ["first line here", "second line the primary missed"]
    assert blocks[1].engine == "b" and blocks[1].agreed == 0.0
    blocks, _, _ = merge(p, o, fill=False)
    assert [b.text for b in blocks] == ["first line here"]


def test_badly_read_line_is_swapped_for_a_confident_reading():
    p = [line("Tho qvick brovvn fux jumps", 0.3, 0.1)]
    o = {"b": [line("The quick brown fox jumps", 0.95, 0.1, engine="b")]}
    blocks, words, _ = merge(p, o)
    assert blocks[0].text == "The quick brown fox jumps" and blocks[0].engine == "b"
    assert all(c == 0.95 for _, c in words)
    blocks, _, _ = merge(p, o, swap=False)
    assert blocks[0].text.startswith("Tho")
    # a confident primary line is never swapped, however sure the other engine is
    blocks, _, _ = merge([line("Tho qvick brovvn fux jumps", 0.8, 0.1)], o)
    assert blocks[0].text.startswith("Tho")


def test_lines_are_matched_by_position_not_order():
    p = [line("alpha beta", 0.6, 0.5), line("gamma delta", 0.6, 0.1)]
    o = {"b": [line("gamma delta", 0.9, 0.1, engine="b"), line("alpha beta", 0.9, 0.5, engine="b")]}
    blocks, words, agreement = merge(p, o)
    assert agreement == 1.0 and all(b.agreed == 1.0 for b in blocks)
    # a line further down the page never confirms one at the top
    o = {"b": [line("alpha beta", 0.9, 0.9, engine="b")]}
    _, _, agreement = merge([line("alpha beta", 0.6, 0.1)], o, fill=False)
    assert agreement == 0.0


def test_ocr_settings_parse_lists_and_versions():
    s = Settings(ocr_engines="apple_vision+rapidocr", ocr_languages="en-US, fr-FR")
    assert s.ocr_engines == ["apple_vision", "rapidocr"] and s.ocr_languages == ["en-US", "fr-FR"]
    assert Settings(ocr_engines="").ocr_engines == []
    names = ocr.available_engines()
    if names:
        rec = ocr.Recognizer(names[:1])
        assert rec.version == f"{ocr.OCR_VERSION}:{names[0]}"
        assert ocr.Recognizer(names).version.startswith(ocr.OCR_VERSION + ":")
    import pytest
    with pytest.raises(RuntimeError):
        ocr.Recognizer([])
    with pytest.raises(ValueError):
        ocr.make_engine("nope")
