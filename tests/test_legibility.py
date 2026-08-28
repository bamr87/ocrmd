"""The admission test: which captures are worth making a page of."""
from __future__ import annotations

from ocrmd import legibility
from ocrmd.ocr import Block, OcrResult


def reading(words: int, confidence: float = 0.95) -> OcrResult:
    text = " ".join(["word"] * words)
    block = Block(text=text, confidence=confidence, bbox=[0.1, 0.1, 0.8, 0.05])
    return OcrResult(text=text, blocks=[block], mean_confidence=confidence, engine="test",
                     words=[(w, confidence) for w in text.split()])


def test_a_page_the_engines_read_is_admitted():
    v = legibility.assess(reading(200))
    assert v.admit and v.describe() == "admitted"
    assert v.metrics == {"words": 200, "confidence": 0.95, "lines": 1}


def test_a_capture_with_almost_no_readable_text_is_refused():
    v = legibility.assess(reading(12))
    assert not v.admit and "12 word(s)" in v.reason and "60" in v.reason


def test_a_capture_read_without_confidence_is_refused():
    v = legibility.assess(reading(300, confidence=0.31))
    assert not v.admit and "0.31" in v.reason


def test_the_thresholds_are_settings_not_constants():
    assert legibility.assess(reading(40), min_words=30).admit
    assert not legibility.assess(reading(40), min_words=50).admit
    assert legibility.assess(reading(300, 0.4), min_confidence=0.3).admit


def test_nothing_is_refused_on_a_machine_that_cannot_read():
    """No OCR engine means no evidence. Refusing pages there would be silent data loss."""
    v = legibility.assess(None)
    assert v.admit and v.metrics == {"ocr": "unavailable"}


def test_the_operator_is_told_what_to_change_not_just_what_was_dropped():
    assert legibility.advice([]) is None
    text = legibility.advice(["only 3 word(s) legible (needs 60)"] * 3)
    assert "3 capture(s)" in text and "raw frames" in text
    assert "closer" in text and "settle" in text        # the two causes behind the illegible pages
    assert "ocrmd ingest raw/" in text                  # ...and how to override the decision
