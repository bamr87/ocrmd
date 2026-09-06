"""The web API through FastAPI's TestClient, against a temporary root."""
from __future__ import annotations
import time

import cv2
import pytest
from fastapi.testclient import TestClient

from conftest import FakeProvider, on_desk, text_page
from ocrmd import pipeline, render
from ocrmd.ocr import available_engine
from ocrmd.web.app import create_app

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")
LINES = ["the quick brown fox jumps over the lazy dog again and again"] * 8 + ["invoice total 44.60 paid in full"]
# a digit mismatch is never auto-resolved, so this transcription always leaves an open ==[…]== flag
FLAGGED_MD = "\n\n".join(LINES).replace("44.60", "44.80")


@pytest.fixture
def client(root, store):
    app = create_app(root)
    with TestClient(app) as c:
        yield c


def _seed(root, store, slug="web-book"):
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(LINES)), slug, "webcam:cam").pages[0]
    r = pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown=FLAGGED_MD, page_number="3", chapter="One"), slug)
    render.ensure_document_meta(root.documents / slug, slug, title="Web Book")
    render.build_document(root.documents / slug, slug)
    return pid, r


def _wait(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


@needs_ocr
def test_the_flat_view_is_served_and_rendered_on_demand(client, root, store):
    """A page scanned before this existed has no flat file; asking for it makes one, because it is a
    pure function of the crop and needs no reprocess."""
    pid, _ = _seed(root, store)
    from ocrmd import pipeline as pl
    pl.flat_path(root, pid).unlink()
    r = client.get("/api/documents/web-book/pages/1/image/flat")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert pl.flat_path(root, pid).exists()


@needs_ocr
def test_a_selected_region_is_read_and_handed_back_with_its_crop(client, root, store):
    _seed(root, store)
    r = client.post("/api/documents/web-book/pages/1/region",
                    json={"x": 0.05, "y": 0.05, "w": 0.9, "h": 0.3}).json()
    assert r["text"] and r["scale"] >= 1 and r["llm"] is None       # local OCR only unless asked
    assert client.get(r["image"]).status_code == 200
    tiny = client.post("/api/documents/web-book/pages/1/region",
                       json={"x": 0.5, "y": 0.5, "w": 0.004, "h": 0.004})
    assert tiny.status_code == 400 and "too small" in tiny.json()["detail"]


@needs_ocr
def test_extraction_pipelines_are_listed_and_run_as_a_job(client, root, store):
    """The free operators need no provider, so the whole path — list, run, write, download — is
    exercised without credentials."""
    _seed(root, store)
    d = root.documents / "web-book" / "etl"
    d.mkdir(parents=True)
    (d / "chunks.yaml").write_text(
        "name: chunks\n"
        "operations:\n"
        "  - {name: chop, type: split, split_key: text, words: 5, overlap: 0}\n"
        "  - {name: first, type: sample, n: 3}\n"
        "pipeline:\n"
        "  operations: [chop, first]\n"
        "  output: {type: csv, path: chunks.csv}\n")
    (d / "broken.yaml").write_text("name: broken\noperations: [{name: x, type: map, output: {schema: {y: nope}}}]\n")

    listed = client.get("/api/documents/web-book/pipelines").json()
    by_file = {p["file"]: p for p in listed}
    assert by_file["chunks.yaml"]["steps"] == ["chop", "first"] and by_file["chunks.yaml"]["llm_steps"] == []
    assert "error" in by_file["broken.yaml"]              # a broken spec does not hide the working one

    job = _wait(client, client.post("/api/jobs", json={"kind": "etl", "doc": "web-book",
                                                       "options": {"pipeline": "chunks.yaml"}}).json()["id"])
    assert job["status"] == "done", job
    out = root.documents / "web-book" / "extracted" / "chunks.csv"
    assert out.exists() and len(out.read_text().splitlines()) == 4      # header + 3 sampled chunks
    assert out.read_text().startswith("_document,_page")
    assert client.get("/api/documents/web-book/extracted/chunks.csv").status_code == 200
    assert client.get("/api/documents/web-book").json()["extracted"] == ["chunks.csv"]
    assert client.get("/api/documents/web-book/extracted/../../ocrmd.db").status_code in (404, 400)


def test_index_and_system_endpoints(client):
    assert "<title>ocrmd</title>" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    d = client.get("/api/doctor?cameras=false").json()
    assert d["db_integrity"] == "ok" and "config" in d
    assert client.get("/api/settings").json()["prompt_version"].startswith("transcribe@")
    assert client.get("/api/scan").json() == {"running": False, "doc": None}
    assert client.get("/api/documents").json() == []
    assert client.get("/api/documents/nope").status_code == 404
    assert client.get("/api/documents/../etc").status_code in (400, 404)


@needs_ocr
def test_document_page_flags_and_review_flow(client, root, store):
    pid, r = _seed(root, store)
    assert r.status == "flagged"
    docs = client.get("/api/documents").json()
    assert docs[0]["slug"] == "web-book" and docs[0]["title"] == "Web Book" and docs[0]["flagged"] == 1
    doc = client.get("/api/documents/web-book").json()
    assert doc["page_list"][0]["status"] == "flagged" and doc["page_list"][0]["agreement"] is not None
    assert client.get("/api/documents/web-book/document.md").status_code == 200

    page = client.get("/api/documents/web-book/pages/1").json()
    assert page["unresolved"] >= 1 and page["flags"][0]["ocr"] and set(page["images"]) >= {"processed", "raw", "bin", "flat"}
    assert client.get(page["images"]["processed"]).status_code == 200
    assert client.get("/api/documents/web-book/pages/1/image/nope").status_code == 404
    assert client.post("/api/documents/web-book/pages/1/signoff", json={"who": "amr"}).status_code == 409   # unresolved

    for f in page["flags"]:
        page = client.post("/api/documents/web-book/pages/1/resolve", json={"n": f["n"], "chosen": f["ocr"]}).json()
    assert page["unresolved"] == 0 and "==[" not in page["body"]
    assert client.post("/api/documents/web-book/pages/1/resolve", json={"n": 99, "chosen": "x"}).status_code == 404

    page = client.post("/api/documents/web-book/pages/1/signoff", json={"who": "amr"}).json()
    assert page["status"] == "reviewed" and page["frontmatter"]["verification"]["reviewed_by"]["who"] == "amr"
    assert store.page(pid)["status"] == "reviewed"
    page = client.post("/api/documents/web-book/pages/1/reopen").json()
    assert page["status"] == "verified" and store.page(pid)["status"] == "verified"

    # human edits through the API survive a later pipeline run
    page = client.put("/api/documents/web-book/pages/1", json={"page_label": "iv", "context": {"chapter": "Mine"}}).json()
    assert page["page_label"] == "iv" and page["frontmatter"]["context"]["chapter"] == "Mine"
    r2 = pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown=FLAGGED_MD, page_number="3", model="m2"), "web-book", refresh=True)
    assert "context.chapter" in r2.preserved and render.read_page(r2.path)[0]["page_label"] == "iv"


@needs_ocr
def test_meta_jobs_and_upload(client, root, store, tmp_path, monkeypatch):
    _seed(root, store)
    meta = client.put("/api/documents/web-book/meta", json={"title": "Renamed", "tags": ["x"], "year": 2020}).json()
    assert meta["title"] == "Renamed" and meta["tags"] == ["x"]
    assert render.load_document_meta(root.documents / "web-book")["year"] == 2020

    job = client.post("/api/jobs", json={"kind": "build", "doc": "web-book"}).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done" and any("built" in line["line"] for line in j["log"])
    # the document file carries the new title; page context keeps what it already had (a page's own value wins)
    assert render.read_page(root.documents / "web-book" / "web-book.md")[0]["title"] == "Renamed"
    assert client.post("/api/jobs", json={"kind": "bogus"}).status_code == 400

    # upload -> ingest job (OCR only: no provider needed)
    import ocrmd.web.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "make_provider", lambda *a, **k: None)
    img = tmp_path / "new.png"
    cv2.imwrite(str(img), on_desk(text_page(["a brand new page with other words"] * 10)))
    with open(img, "rb") as f:
        job = client.post("/api/documents/web-book/ingest", files=[("files", ("new.png", f, "image/png"))]).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done", j
    assert client.get("/api/documents/web-book").json()["pages"] == 2
    assert client.post("/api/documents/web-book/ingest", files=[("files", ("x.txt", b"hi", "text/plain"))]).status_code == 400

    job = client.post("/api/jobs", json={"kind": "verify", "doc": "web-book", "options": {"no_llm": True}}).json()
    assert _wait(client, job["id"])["status"] == "done"
    assert client.get("/api/jobs").json()[0]["id"] == job["id"]


def test_scan_start_with_unknown_camera_reports_error(client, monkeypatch):
    import ocrmd.capture as cap
    monkeypatch.setattr(cap.camctl, "map_cv2_indices", lambda *a, **k: {})
    r = client.post("/api/scan/start", json={"doc": "cam-test", "camera": "no-such-camera", "no_llm": True})
    assert r.status_code == 200
    for _ in range(20):
        snap = client.get("/api/scan").json()
        if not snap["running"]:
            break
        time.sleep(0.1)
    assert not snap["running"] and "no camera matching" in (snap["error"] or "")
    assert client.post("/api/scan/command", json={"command": "capture"}).status_code == 409
    assert client.get("/api/scan/frame.jpg").status_code == 404
    assert client.post("/api/scan/start", json={"doc": "bad slug!", "no_llm": True}).status_code == 400


@needs_ocr
def test_delete_pages_and_document(client, root, store, monkeypatch):
    import numpy as np
    from ocrmd.web import jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "make_provider", lambda *a, **k: None)
    # a spread -> two pages sharing one raw capture, plus a single page
    page = text_page()
    spread = on_desk(np.hstack([page, np.full((page.shape[0], 26, 3), 90, np.uint8), page]))
    a, b = pipeline.ingest_frame(root, store, spread, "del-book", "webcam:cam").pages
    c = pipeline.ingest_frame(root, store, on_desk(text_page(["a different page entirely"] * 12)), "del-book", "file:x").pages[0]
    for pid in (a, b, c):
        pipeline.run_page(root, store, store.page(pid), None, "del-book")
    render.build_document(root.documents / "del-book", "del-book")
    raw_ab = root.raw / f"{store.page(a)['raw_sha256']}.jpg"
    assert raw_ab.exists() and client.get("/api/documents/del-book").json()["pages"] == 3

    # delete one half: its files go, the raw stays (the other half still needs it)
    r = client.delete("/api/documents/del-book/pages/1").json()
    assert r["deleted"] == [1] and r["raws"] == 0
    assert not (root.processed / f"{a}.png").exists() and not render.page_path(root.documents / "del-book", 1).exists()
    assert raw_ab.exists() and store.page(a) is None and store.page(b) is not None
    assert client.get("/api/documents/del-book/pages/1").status_code == 404
    assert client.get("/api/documents/del-book").json()["pages"] == 2

    # the same frame is still a duplicate: its other half is still in the document
    assert pipeline.ingest_frame(root, store, spread, "del-book", "webcam:cam").pages == []

    # bulk delete of the whole capture: the remaining half, raw purged
    r = client.post("/api/documents/del-book/pages/delete", json={"sequences": [2], "whole_capture": True}).json()
    assert r["deleted"] == [2] and r["raws"] == 1 and not raw_ab.exists()
    assert client.post("/api/documents/del-book/pages/delete", json={"sequences": [99]}).status_code == 404
    left = client.get("/api/documents/del-book").json()
    assert [p["sequence"] for p in left["page_list"]] == [3]
    assert "page: 0003" in (root.documents / "del-book" / "del-book.md").read_text()   # rebuilt
    # ...and now the frame can be scanned again: nothing refers to it any more
    assert len(pipeline.ingest_frame(root, store, spread, "del-book", "webcam:cam").pages) == 2
    assert [p["sequence"] for p in client.get("/api/documents/del-book").json()["page_list"]] == [3, 4, 5]

    # whole document: confirmation must repeat the slug
    assert client.delete("/api/documents/del-book?confirm=nope").status_code == 400
    r = client.delete("/api/documents/del-book?confirm=del-book").json()
    assert r["deleted"] == [3, 4, 5] and r["raws"] == 2
    assert not (root.documents / "del-book").exists() and "del-book" not in store.documents()
    assert client.get("/api/documents/del-book").status_code == 404
    assert client.delete("/api/documents/del-book?confirm=del-book").status_code == 404
    assert pipeline.orphan_artifacts(root, store) == []


# --- the page model ---------------------------------------------------------------------------------
@needs_ocr
def test_the_page_model_is_served_drawn_and_editable(client, root, store):
    pid, _ = _seed(root, store)
    p = client.get("/api/documents/web-book/pages/1").json()
    assert p["layout"] == "/api/documents/web-book/pages/1/layout" and isinstance(p["figures"], list)
    lay = client.get(p["layout"]).json()
    assert lay["page_id"] == pid and lay["blocks"] and lay["version"].startswith("layout@")
    boxes = client.get("/api/documents/web-book/pages/1/boxes").json()
    assert boxes["lines"] and boxes["blocks"] and boxes["flags"]
    assert boxes["flags"][0]["bbox"] is not None                    # the flag is placed on the image
    # a human edit: one block's role corrected, sent back whole
    lay["blocks"][0]["role"] = "heading"
    r = client.put("/api/documents/web-book/pages/1/layout", json={"blocks": lay["blocks"]})
    assert r.status_code == 200 and r.json()["blocks"][0]["source"] == "human"
    assert pipeline.load_layout(root, pid).blocks[0].role == "heading"
    assert client.put("/api/documents/web-book/pages/1/layout", json={"blocks": [{"bbox": "no"}]}).status_code == 400


@needs_ocr
def test_the_document_exports_as_one_json_object(client, root, store):
    pid, _ = _seed(root, store)
    (root.documents / "web-book" / "extracted").mkdir(exist_ok=True)
    (root.documents / "web-book" / "extracted" / "cases.json").write_text('[{"_page": 1, "name": "x"}]')
    d = client.get("/api/documents/web-book/export.json").json()
    assert d["document"] == "web-book" and d["meta"]["title"] == "Web Book"
    assert d["pages"][0]["id"] == pid and d["pages"][0]["layout"]["blocks"]
    assert d["pages"][0]["body"] and d["pages"][0]["frontmatter"]["ocrmd"] == 1
    assert d["extractions"] == {"cases": [{"_page": 1, "name": "x"}]}
    assert client.get("/api/documents/nope/export.json").status_code == 404
