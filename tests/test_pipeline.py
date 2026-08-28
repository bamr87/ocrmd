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
from ocrmd.ocr import OcrResult, available_engine

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")


def _ingest(root, store, slug="book", lines=None, source="webcam:HD Webcam C525"):
    ids = pipeline.ingest_frame(root, store, on_desk(text_page(lines)), slug, source).pages
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
    a = pipeline.ingest_frame(root, store, img, "book", "file:a.jpg").pages
    assert pipeline.ingest_frame(root, store, img, "book", "file:a-again.jpg").pages == []      # exact
    noisy = np.clip(img.astype(np.int16) + np.random.default_rng(0).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    assert pipeline.ingest_frame(root, store, noisy, "book", "file:b.jpg").pages == []          # near: sensor noise
    assert len(store.duplicates("book")) == 1
    noisy2 = np.clip(img.astype(np.int16) + np.random.default_rng(1).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    assert pipeline.ingest_frame(root, store, noisy2, "other-book", "file:c.jpg").pages         # other document: a new page
    # same layout, different text: a page of the same book, not a duplicate
    other = on_desk(text_page(["a different line of text on the next page"] * 9))
    assert pipeline.ingest_frame(root, store, other, "book", "file:d.jpg").pages
    assert len(a) == 1


def test_spread_becomes_two_pages_with_hints(root, store):
    page = text_page()
    spread = on_desk(np.hstack([page, np.full((page.shape[0], 26, 3), 90, np.uint8), page]))
    ids = pipeline.ingest_frame(root, store, spread, "book", "webcam:cam").pages
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
            return {pid: self.transcribe(img, hint_text, hint, focus) for pid, img, hint_text, hint, focus in self.items}

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


# --- focus bracket ------------------------------------------------------------------------------
def _blur(img, k=9):
    import cv2
    return cv2.GaussianBlur(img, (k, k), 0)


@needs_ocr
def test_a_bracketed_capture_carries_its_variants_through_every_stage(root, store):
    """A soft capture with a sharp second exposure: both are stored, both are cropped the same way,
    and the page's OCR is resolved across them."""
    sharp = on_desk(text_page())
    pid = pipeline.ingest_frame(root, store, _blur(sharp), "book", "webcam:test", [sharp]).pages[0]
    page = store.page(pid)
    raw_sha = page["raw_sha256"]
    assert [f.name for f in pipeline.variant_raws(root, raw_sha)] == [f"{raw_sha}.v1.jpg"]
    crops = pipeline.variant_crops(root, pid)
    assert [f.name for f in crops] == [f"{pid}.v1.png"]

    import cv2
    primary = cv2.imread(str(root.processed / f"{pid}.png"))
    # the variant is cropped onto the page's own geometry, or normalised boxes could not be compared
    assert cv2.imread(str(crops[0])).shape == primary.shape

    r = pipeline.run_page(root, store, page, None, "book")
    mf = pipeline.focus_analysis(root, pid)
    assert mf is not None and [v.label for v in mf.variants] == ["v1"]
    fm, _body = render.read_page(r.path)
    assert fm["ocr"]["focus"]["variants"] == 1 and fm["ocr"]["focus"]["artifact"] == f"ocr/{pid}.focus.json"
    # resolving only ever adds or replaces doubtful lines, so the sharp exposure cannot lose text
    res = pipeline.run_ocr(root, store, page)
    assert len(res.words) >= sum(len(b.text.split()) for b in res.blocks if not b.focus)


@needs_ocr
def test_a_bracket_recovers_text_the_capture_could_not_read(root, store):
    """The point of the feature, end to end: a page whose top half is out of focus, bracketed with an
    exposure focused on that half, comes out with more of the page correctly read."""
    import cv2
    from ocrmd import ocr as ocr_mod
    lines = ["it is a truth universally acknowledged that a single man",
             "in possession of a good fortune must be in want of a wife",
             "however little known the feelings or views of such a man",
             "may be on his first entering a neighbourhood this truth",
             "is so well fixed in the minds of the surrounding families"] * 3
    truth = set(" ".join(lines).split())
    page = text_page(lines)
    small = cv2.resize(page, (520, int(page.shape[0] * 520 / page.shape[1])), interpolation=cv2.INTER_AREA)

    def half_blur(img, top):                       # one half in focus, the other not: a curled leaf
        out = img.copy()
        h = out.shape[0] // 2
        sl = slice(0, h) if top else slice(h, out.shape[0])
        out[sl] = cv2.GaussianBlur(out[sl], (11, 11), 0)
        return out

    root.min_page_width = 300
    pid = pipeline.ingest_frame(root, store, on_desk(half_blur(small, True), 30), "book", "file:t",
                                [on_desk(half_blur(small, False), 30)]).pages[0]
    correct = lambda res: sum(1 for w, _ in res.words if w.lower().strip(",.") in truth)
    alone = ocr_mod.recognizer(root).run(cv2.imread(str(root.processed / f"{pid}.png")))
    both = pipeline.run_ocr(root, store, store.page(pid))
    mf = pipeline.focus_analysis(root, pid)
    assert correct(both) > correct(alone), (correct(alone), correct(both), mf.summary())
    assert mf.improved + mf.filled > 0
    assert all(w.lower().strip(",.") in truth for w, _ in both.words)      # and nothing invented


@needs_ocr
def test_an_unbracketed_capture_is_untouched(root, store):
    pid = _ingest(root, store)
    assert pipeline.variant_crops(root, pid) == [] and pipeline.focus_analysis(root, pid) is None
    r = pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="hello"), "book")
    fm, _ = render.read_page(r.path)
    assert "focus" not in fm["ocr"]


@needs_ocr
def test_deleting_a_page_purges_the_whole_bracket(root, store):
    sharp = on_desk(text_page())
    pid = pipeline.ingest_frame(root, store, _blur(sharp), "book", "webcam:test", [sharp]).pages[0]
    sha = store.page(pid)["raw_sha256"]
    pipeline.run_page(root, store, store.page(pid), None, "book")
    out = pipeline.delete_pages(root, store, "book", [pid])
    assert out.raws == 2 and not list(root.raw.glob("*.jpg"))
    assert pipeline.variant_crops(root, pid) == [] and not pipeline.focus_path(root, pid).exists()
    assert not (root.raw / f"{sha}.v1.jpg").exists()


@needs_ocr
def test_a_variant_that_crops_to_a_different_shape_is_dropped(root, store):
    """A variant whose own page detection landed elsewhere would align its text against the wrong
    lines; a wrong reading is worse than no second reading."""
    import cv2
    page = on_desk(text_page())
    squashed = cv2.resize(page, (page.shape[1], int(page.shape[0] * 0.6)))
    pid = pipeline.ingest_frame(root, store, page, "book", "webcam:test", [squashed]).pages[0]
    assert pipeline.variant_crops(root, pid) == []


def test_unresolved_regions_send_the_images_and_the_analysis_to_the_model(root, store):
    """The end of the line: no focus could read a region, so the exposures and what each of them made
    of it go to the LLM rather than the pipeline picking the highest number."""
    import cv2
    from ocrmd import multifocus
    from ocrmd.multifocus import Candidate, FocusReading, MultiFocus, Unresolved
    page = text_page()
    pid = "abc123abc123abc1"
    store.ensure_document("book")
    store.add_page(pid, "book", "sha", "phash", None, page.shape[1], None)
    cv2.imwrite(str(root.processed / f"{pid}.png"), page)
    cv2.imwrite(str(root.processed / f"{pid}.v1.png"), page)
    cv2.imwrite(str(root.processed / f"{pid}.v2.png"), page)
    mf = MultiFocus(variants=[FocusReading(label="v1"), FocusReading(label="v2")],
                    unresolved=[Unresolved(bbox=[0.1, 0.3, 0.5, 0.03], candidates=[
                        Candidate(source="focus", text="recerved", confidence=0.3),
                        Candidate(source="v2", text="reserved", confidence=0.61)])])
    pipeline.focus_path(root, pid).write_text(mf.model_dump_json())

    root.focus_llm_images = 1
    views = pipeline.focus_views(root, pid)
    assert views is not None and len(views.images) == 1        # only the one that read it best
    assert "v2" in views.note and "reserved" in views.note

    prov = FakeProvider(markdown="reserved for the author")
    pipeline.run_llm(root, store, store.page(pid), prov,
                     OcrResult(text="recerved", blocks=[], mean_confidence=0.3, engine="test"))
    assert prov.focus[0] is not None and len(prov.focus[0].images) == 1

    # a page whose regions all resolved locally pays nothing
    pipeline.focus_path(root, pid).write_text(MultiFocus(variants=[FocusReading(label="v1")], improved=2).model_dump_json())
    assert pipeline.focus_views(root, pid) is None
    assert multifocus.variant_ranking(MultiFocus(variants=[FocusReading(label="v1")])) == ["v1"]


def test_extra_exposures_are_labelled_in_the_request():
    """Every extra image is introduced as another view of the same page: an unlabelled second image
    reads as the next page, and a model that believes that merges two readings into one."""
    from ocrmd.llm import FocusViews, content_blocks
    img = np.zeros((20, 20, 3), np.uint8)
    focus = FocusViews(images=[img, img], note="Attached after the page image: 2 more exposure(s).")
    blocks = content_blocks(img, "draft", None, focus)
    assert [b["type"] for b in blocks] == ["image", "text", "image", "text", "image", "text"]
    assert blocks[1]["text"].startswith("Alternative exposure 1 of the SAME page")
    assert blocks[3]["text"].startswith("Alternative exposure 2 of the SAME page")
    assert "2 more exposure(s)" in blocks[-1]["text"] and "Local OCR draft" in blocks[-1]["text"]
    # unbracketed pages send exactly what they always did
    assert [b["type"] for b in content_blocks(img, "draft", None, None)] == ["image", "text"]


def _unreadable(page):
    """Present and inked, but too few pixels on the type to read — the real rig's failure mode."""
    import cv2
    h, w = page.shape[:2]
    return cv2.resize(cv2.resize(page, (int(w * 0.08), int(h * 0.08))), (w, h))


# --- admission: which captures become pages ------------------------------------------------------
@needs_ocr
def test_an_unreadable_capture_is_refused_instead_of_becoming_a_page(root, store):
    """The complaint this answers: captures that were never going to be legible became pages, and
    each one cost an LLM call to find that out."""
    import cv2
    page = on_desk(text_page())
    mush = cv2.GaussianBlur(page, (0, 0), 9)          # nothing survives this
    got = pipeline.ingest_frame(root, store, mush, "book", "webcam:cam", admit=True)
    assert got.pages == [] and got.duplicate_of is None
    assert got.rejected and "word(s) legible" in got.rejected[0]
    assert store.pages("book") == []                  # no page, no page file, no LLM call

    # nothing was deleted: the frame is on disk and the store says why it produced nothing
    rej = store.rejected_captures("book")
    assert len(rej) == 1 and rej[0]["document"] == "book"
    assert (root.raw / f"{rej[0]['sha256']}.jpg").exists()

    # ...and the operator's override works, because file ingest never applies the test
    again = pipeline.ingest_frame(root, store, mush, "book", "file:forced.jpg")
    assert len(again.pages) == 1 and not again.rejected


@needs_ocr
def test_a_readable_capture_is_admitted_and_pays_no_extra_ocr(root, store):
    got = pipeline.ingest_frame(root, store, on_desk(text_page()), "book", "webcam:cam", admit=True)
    assert len(got.pages) == 1 and not got.rejected
    pid = got.pages[0]
    # the admission pass is kept as the page's OCR stage: an admitted crop is read exactly once
    assert (root.ocr_dir / f"{pid}.json").exists()
    mtime = (root.ocr_dir / f"{pid}.json").stat().st_mtime
    pipeline.run_page(root, store, store.page(pid), None, "book")
    assert (root.ocr_dir / f"{pid}.json").stat().st_mtime == mtime


@needs_ocr
def test_admission_is_off_unless_asked_for(root, store):
    """`ingest` and forced captures create the page whatever the shutter thinks of it."""
    import cv2
    mush = cv2.GaussianBlur(on_desk(text_page()), (0, 0), 9)
    got = pipeline.ingest_frame(root, store, mush, "book", "webcam:cam")
    assert len(got.pages) == 1 and not got.rejected


@needs_ocr
def test_half_a_spread_can_be_kept_when_the_other_half_is_not(root, store):
    """Refusing the whole frame because one page of a spread is unreadable would lose a good page."""
    import numpy as np
    good = text_page(["the quick brown fox jumps over the lazy dog"] * 12)
    gutter = np.full((good.shape[0], 60, 3), 240, np.uint8)
    spread = on_desk(np.hstack([good, gutter, _unreadable(good)]), margin=120)
    got = pipeline.ingest_frame(root, store, spread, "book", "webcam:cam", admit=True)
    assert len(got.pages) == 1 and len(got.rejected) == 1
    assert store.page(got.pages[0])["part"] == "left"


@needs_ocr
def test_a_partly_refused_spread_is_not_listed_as_needing_a_re_shoot(root, store):
    """It produced a page. Only captures that came to nothing are the operator's problem."""
    import numpy as np
    good = text_page(["the quick brown fox jumps over the lazy dog"] * 12)
    gutter = np.full((good.shape[0], 60, 3), 240, np.uint8)
    spread = on_desk(np.hstack([good, gutter, _unreadable(good)]), margin=120)
    got = pipeline.ingest_frame(root, store, spread, "book", "webcam:cam", admit=True)
    assert got.pages and got.rejected
    assert store.rejected_captures("book") == []      # nothing here needs shooting again
