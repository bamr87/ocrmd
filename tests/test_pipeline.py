"""End-to-end through the pipeline on synthetic pages: ingest, OCR-only render, LLM upgrade,
edit preservation, continuity, failures, batches, and the document build."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeProvider, on_desk, text_page
from ocrmd import llm as llm_mod, pipeline, render, review
from ocrmd.llm import LLMPageError, LLMUnavailable
from ocrmd.ocr import available_engine

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")


def _ingest(root, store, slug="book", lines=None, source="webcam:HD Webcam C525"):
    ids = pipeline.ingest_frame(root, store, on_desk(text_page(lines)), slug, source)
    assert len(ids) == 1
    return ids[0]


@needs_ocr
def test_ocr_only_page_then_llm_upgrade_keeps_ocr_cached(root, store):
    pid = _ingest(root, store)
    page = store.page(pid)
    r = pipeline.run_page(root, store, page, None, "book")
    fm, body = render.read_page(r.path)
    assert r.status == "unverified" and fm["source"]["type"] == "webcam" and fm["source"]["device"] == "HD Webcam C525"
    assert "possession" in body                       # OCR text rendered immediately
    ocr_mtime = (root.ocr_dir / f"{pid}.json").stat().st_mtime
    prov = FakeProvider(page_number="12", chapter="Chapter 1")
    r2 = pipeline.run_page(root, store, page, prov, "book")
    assert r2.status in ("verified", "flagged") and r2.agreement is not None
    assert (root.ocr_dir / f"{pid}.json").stat().st_mtime == ocr_mtime      # not re-run
    fm2, _ = render.read_page(r2.path)
    assert fm2["page_label"] == "12" and fm2["context"]["chapter"] == "Chapter 1"
    assert fm2["llm"]["prompt"] == root.prompt_version and store.page(pid)["page_label"] == "12"
    hint = prov.calls[0][1]
    assert hint is not None and hint.part is None


@needs_ocr
def test_completed_llm_pass_is_reused_unless_refresh(root, store):
    pid = _ingest(root, store)
    page = store.page(pid)
    prov = FakeProvider(page_number="1")
    pipeline.run_page(root, store, page, prov, "book")
    store.mark_stage(pid, "llm", "transcribe@old:fake-model")  # pretend it was made by an older prompt
    pipeline.run_page(root, store, page, prov, "book")
    assert len(prov.calls) == 1                                 # reused: no token spend
    pipeline.run_page(root, store, page, prov, "book", refresh=True)
    assert len(prov.calls) == 2
    pipeline.run_page(root, store, page, prov, "book", refresh=True)
    assert len(prov.calls) == 2                                 # current prompt+model: nothing to refresh
    other = FakeProvider(page_number="1", model="newer-model")
    pipeline.run_page(root, store, page, other, "book", refresh=True)
    assert len(other.calls) == 1                                # a different model is a refresh


@needs_ocr
def test_human_edits_survive_a_rerun(root, store):
    pid = _ingest(root, store)
    page = store.page(pid)
    prov = FakeProvider(page_number="7")
    r = pipeline.run_page(root, store, page, prov, "book")
    fm, body = render.read_page(r.path)
    fm["context"]["chapter"] = "My chapter"
    fm["page_label"] = "vii"
    r.path.write_text(render.page_frontmatter(**fm) + "\n" + body)
    r2 = pipeline.run_page(root, store, store.page(pid), prov, "book", refresh=True)
    fm2, _ = render.read_page(r2.path)
    assert fm2["context"]["chapter"] == "My chapter" and fm2["page_label"] == "vii"
    assert "context.chapter" in r2.preserved and "page_label" in r2.preserved
    # a reviewer's sign-off freezes the body too
    fm2["verification"]["reviewed_by"] = {"who": "amr", "at": "now"}
    r2.path.write_text(render.page_frontmatter(**fm2) + "\nHand-corrected body.\n")
    r3 = pipeline.run_page(root, store, store.page(pid), prov, "book", refresh=True)
    fm3, body3 = render.read_page(r3.path)
    assert body3.strip() == "Hand-corrected body." and fm3["verification"]["status"] == "reviewed" == r3.status
    # and an untouched pipeline output is not mistaken for a human edit
    r4 = pipeline.run_page(root, store, store.page(pid), None, "book")
    assert r4.status == "reviewed"


@needs_ocr
def test_page_number_continuity_is_checked(root, store):
    a = _ingest(root, store, lines=["first page text line"] * 6)
    b = _ingest(root, store, lines=["the second page has a different shape entirely"] * 14)
    pipeline.run_page(root, store, store.page(a), FakeProvider(page_number="10"), "book")
    r = pipeline.run_page(root, store, store.page(b), FakeProvider(page_number="14"), "book")
    assert r.continuity and "jumped 10 → 14" in r.continuity and "3 page(s)" in r.continuity
    r = pipeline.run_page(root, store, store.page(b), FakeProvider(page_number="10", model="m2"), "book", refresh=True)
    assert "repeats" in r.continuity
    r = pipeline.run_page(root, store, store.page(b), FakeProvider(page_number="11", model="m3"), "book", refresh=True)
    assert r.continuity is None


@needs_ocr
def test_llm_failures_are_recorded_and_page_stays_unverified(root, store):
    pid = _ingest(root, store)
    page = store.page(pid)
    with pytest.raises(LLMUnavailable):
        pipeline.run_page(root, store, page, FakeProvider(raise_exc=LLMUnavailable("down")), "book")
    assert store.stage_error(pid, "llm") == "down"
    r = pipeline.run_page(root, store, page, None, "book")     # OCR-only render still works
    fm, _ = render.read_page(r.path)
    assert r.status == "unverified" and fm["verification"]["llm_error"] == "down"
    with pytest.raises(LLMPageError):
        pipeline.run_page(root, store, page, FakeProvider(raise_exc=LLMPageError("refusal")), "book")
    assert store.stage_error(pid, "llm") == "refusal"
    # a later successful pass clears it
    r = pipeline.run_page(root, store, page, FakeProvider(page_number="3"), "book")
    assert r.status != "unverified" and store.stage_error(pid, "llm") is None


@needs_ocr
def test_illegible_and_suspect_statuses(root, store):
    pid = _ingest(root, store)
    page = store.page(pid)
    r = pipeline.run_page(root, store, page, FakeProvider(markdown="", legible=False), "book")
    assert r.status == "illegible" and r.illegible_reason == "blurry"
    invented = "completely different vocabulary entirely unrelated invented content about something else nothing whatsoever shared"
    r = pipeline.run_page(root, store, page, FakeProvider(markdown=invented, model="m2"), "book", refresh=True)
    assert r.status == "suspect"


def test_duplicates_are_scoped_and_recorded(root, store):
    img = on_desk(text_page())
    a = pipeline.ingest_frame(root, store, img, "book", "file:a.jpg")
    assert pipeline.ingest_frame(root, store, img, "book", "file:a-again.jpg") == []      # exact
    noisy = np.clip(img.astype(np.int16) + np.random.default_rng(0).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    assert pipeline.ingest_frame(root, store, noisy, "book", "file:b.jpg") == []          # near: sensor noise
    assert len(store.duplicates("book")) == 1
    noisy2 = np.clip(img.astype(np.int16) + np.random.default_rng(1).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    assert pipeline.ingest_frame(root, store, noisy2, "other-book", "file:c.jpg")         # other document: a new page
    # same layout, different text: a page of the same book, not a duplicate
    other = on_desk(text_page(["a different line of text on the next page"] * 9))
    assert pipeline.ingest_frame(root, store, other, "book", "file:d.jpg")
    assert len(a) == 1


def test_spread_becomes_two_pages_with_hints(root, store):
    page = text_page()
    spread = on_desk(np.hstack([page, np.full((page.shape[0], 26, 3), 90, np.uint8), page]))
    ids = pipeline.ingest_frame(root, store, spread, "book", "webcam:cam")
    assert [store.page(i)["part"] for i in ids] == ["left", "right"]
    hint = pipeline.page_hint(root, store, store.page(ids[1]), "book")
    assert hint.part == "right"


@needs_ocr
def test_batch_round_trip(root, store):
    class BatchProvider(FakeProvider):
        def batch_submit(self, items):
            self.items = items
            return "msgbatch_x"

        def batch_status(self, batch_id):
            return "ended", {"succeeded": len(self.items)}

        def batch_collect(self, batch_id):
            return {pid: self.transcribe(img, hint_text, hint) for pid, img, hint_text, hint in self.items}

    a = _ingest(root, store, lines=["alpha beta gamma delta"] * 5)
    b = _ingest(root, store, lines=["epsilon zeta eta theta iota kappa lambda"] * 15)
    prov = BatchProvider(page_number="5")
    batch_id, pids = pipeline.submit_batch(root, store, [store.page(a), store.page(b)], prov, lambda p: "book")
    assert batch_id == "msgbatch_x" and set(pids) == {a, b} and store.open_batches()
    assert pipeline.wait_for_batch(prov, batch_id, poll_seconds=0) == "ended"
    outcomes = pipeline.collect_batch(root, store, batch_id, prov)
    assert set(outcomes) == {a, b} and all(v != "unverified" for v in outcomes.values())
    assert store.open_batches() == [] and store.stage_done(a, "llm", pipeline.llm_version(root, "fake-model"))


@needs_ocr
def test_missing_artifacts_are_rederived_from_raw(root, store):
    pid = _ingest(root, store)
    for f in root.processed.glob(f"{pid}.*"):
        f.unlink()
    r = pipeline.run_page(root, store, store.page(pid), None, "book")
    assert r.path.exists() and (root.processed / f"{pid}.png").exists()
    assert pipeline.missing_artifacts(root, store) == {"raw": [], "processed": []}
    (root.ocr_dir / "deadbeef.json").write_text("{}")
    assert [p.name for p in pipeline.orphan_artifacts(root, store)] == ["deadbeef.json"]


# --- render / build ------------------------------------------------------------------------------
def _page_file(doc_dir: Path, seq: int, label, chapter=None, status="verified", body="text"):
    fm = {"ocrmd": 1, "id": f"id{seq}", "document": "book", "sequence": seq, "page_label": label,
          "context": {"title": None, "author": [], "chapter": chapter, "section": None, "tags": []},
          "references": {"related": []},
          "verification": {"status": status, "agreement": 0.9 if status != "unverified" else None}}
    render.write_page(doc_dir, seq, fm, body)


def test_build_inherits_document_meta_and_carries_chapters_forward(root):
    doc_dir = root.documents / "book"
    render.ensure_document_meta(doc_dir, "book", title="The Book", tags=["lsat"], project="study")
    meta = render.load_document_meta(doc_dir)
    assert meta["title"] == "The Book" and meta["tags"] == ["lsat"] and meta["kind"] == "textbook"
    render.ensure_document_meta(doc_dir, "book", tags=["prep"])            # merges, never clobbers
    assert render.load_document_meta(doc_dir)["tags"] == ["lsat", "prep"]
    _page_file(doc_dir, 1, "9", chapter="Introduction")
    _page_file(doc_dir, 2, "10")
    _page_file(doc_dir, 3, "14", chapter="Logic Games", status="flagged", body="a ==[b]==[^ocrmd-1] c\n\n[^ocrmd-1]: ocr=\"x\" llm=\"b\" — unresolved\n")
    _page_file(doc_dir, 4, None, status="illegible")
    out = render.build_document(doc_dir, "book")
    fm, body = render.read_page(out)
    assert fm["title"] == "The Book" and fm["pages"] == 4 and fm["page_labels"] == "9–14"
    assert fm["needs_reshoot"] == [4] and fm["label_gaps"] == ["10→14"] and fm["unresolved_flags"] == 1
    assert "- Introduction (page 0001, p. 9)" in body and "- Logic Games (page 0003, p. 14)" in body
    p2 = render.read_page(render.page_path(doc_dir, 2))[0]
    assert p2["context"]["chapter"] == "Introduction" and p2["context"]["title"] == "The Book"
    assert p2["nav"] == {"prev": "pages/0001.md", "next": "pages/0003.md", "document": "../book.md"}


def test_label_gaps_ignore_roman_numerals_and_big_jumps():
    assert render.label_gaps(["i", "ii", "1", "2", "5", "60", "61"]) == [(2, 5)]
    assert render.label_gaps(["3", None, "4"]) == []


# --- review helpers ------------------------------------------------------------------------------
def test_flag_parsing_and_resolution():
    body = ("The total was ==[44.80]==[^ocrmd-1] paid and ==[recieved]==[^ocrmd-2] by\n\n"
            "[^ocrmd-1]: ocr=\"44.60\" llm=\"44.80\" — unresolved\n[^ocrmd-2]: ocr=\"received\" llm=\"recieved\" — unresolved\n")
    flags = review.parse_flags(body)
    assert [(f.n, f.ocr, f.llm) for f in flags] == [(1, "44.60", "44.80"), (2, "received", "recieved")]
    assert "[44.80]" in flags[0].context(body)
    body = review.resolve(body, 1, "44.60")
    assert "44.60 paid" in body and "[^ocrmd-1]" not in body and review.unresolved(body) == 1
    body = review.resolve(body, 2, "received")
    assert review.unresolved(body) == 0 and body.endswith("by\n")


def test_fingerprint_ignores_nav_and_provenance():
    fm = {"context": {"chapter": "A"}, "page_label": "1", "references": {"related": []},
          "verification": {"reviewed_by": None}, "nav": {"prev": None}, "llm": {"model": "x"}}
    a = review.fingerprint(fm, "body")
    fm["nav"] = {"prev": "pages/0001.md"}; fm["llm"] = {"model": "y"}
    assert review.fingerprint(fm, "body") == a
    fm["context"]["chapter"] = "B"
    assert review.fingerprint(fm, "body") != a


def test_transcription_schema_matches_provider_request():
    """The xAI request carries the same strict schema as Anthropic's structured output."""
    import os
    os.environ["XAI_API_KEY"] = "test"
    p = llm_mod.XAIProvider("grok-4")
    body = p.request_body(np.zeros((20, 20, 3), np.uint8), "draft", llm_mod.PageHint(part="left", prev_page_label="3"))
    assert body["response_format"]["json_schema"]["schema"] is llm_mod.TRANSCRIPTION_SCHEMA
    assert body["response_format"]["json_schema"]["strict"] is True
    text = body["messages"][1]["content"][1]["text"]
    assert "left page" in text and "page label=3" in text and "Local OCR draft" in text
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_anthropic_request_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    img = np.zeros((20, 20, 3), np.uint8)
    opus = llm_mod.AnthropicProvider("claude-opus-5", effort="low")
    p = opus._params(img, "", None)
    assert p["fallbacks"] == "default" and p["betas"] == ["server-side-fallback-2026-07-01"]
    assert p["output_config"]["effort"] == "low" and p["output_config"]["format"]["type"] == "json_schema"
    assert "fallbacks" not in opus._params(img, "", None, batch=True)          # rejected by the Batches API
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert p["messages"][0]["content"][1]["text"] == "Transcribe the page."
    haiku = llm_mod.AnthropicProvider("claude-haiku-4-5")
    ph = haiku._params(img, "", None)
    assert "effort" not in ph["output_config"] and "fallbacks" not in ph


def test_schema_mismatch_is_a_page_error():
    with pytest.raises(LLMPageError):
        llm_mod.parse_transcription('{"kind": "book_page"}')
    assert llm_mod.parse_transcription(json.dumps({"kind": "book_page", "markdown": "x"})).markdown == "x"
