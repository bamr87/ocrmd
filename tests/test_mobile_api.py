"""The phone app's side of the server: pairing, who may call what, and `/api/v1` end to end (TestClient)."""
from __future__ import annotations
import io, time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeProvider, on_desk, text_page
from ocrmd import devices, pipeline, render
from ocrmd.api import Library
from ocrmd.ocr import available_engine
from ocrmd.store import Store
from ocrmd.web import v1
from ocrmd.web.app import create_app

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")
LINES = ["the quick brown fox jumps over the lazy dog again and again"] * 8 + ["invoice total 44.60 paid in full"]
FLAGGED_MD = "\n\n".join(LINES).replace("44.60", "44.80")     # a digit mismatch is never auto-resolved
LOCAL, PHONE = ("127.0.0.1", 50001), ("192.168.1.20", 50002)


@pytest.fixture
def paired_app(root):
    app = create_app(root, access="paired")
    with TestClient(app, client=LOCAL) as local, TestClient(app, client=PHONE) as phone:
        yield local, phone


def _pair(local, phone, name="Test phone"):
    code = local.post("/api/v1/devices/pairings", json={"name": name}).json()["code"]
    r = phone.post("/api/v1/pair", json={"code": code, "name": "Pixel 9", "platform": "android"})
    assert r.status_code == 200, r.text
    return r.json(), {"Authorization": f"Bearer {r.json()['token']}"}


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def _wait(client, job_id, headers, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


# --- pairing --------------------------------------------------------------------------------------------
def test_a_pairing_code_works_once_and_only_its_hash_is_stored(store):
    p = devices.start_pairing(store, "kitchen phone")
    row = store.devices()[0]
    assert p.code not in str(dict(row)) and row["code_hash"] == devices.hash_secret(p.code)
    dev, token = devices.claim(store, p.display.lower(), "Pixel", "android")   # case and dash do not matter
    assert dev.name == "kitchen phone" and dev.platform == "android" and not dev.pending
    assert token.startswith("ocrmd_") and token not in str(dict(store.devices()[0]))
    assert devices.claim(store, p.code, "again") is None                         # used
    assert devices.authenticate(store, token).id == dev.id
    devices.revoke(store, dev.id)
    assert devices.authenticate(store, token) is None


def test_an_expired_code_is_refused_and_no_longer_listed(store):
    p = devices.start_pairing(store, ttl=-1)
    assert devices.claim(store, p.code, "late") is None
    assert devices.list_devices(store) == []
    assert [d.describe().split()[-2:] for d in devices.list_devices(store, include_revoked=True)] == [["code", "expired"]]


def test_ten_wrong_codes_void_every_open_code(store):
    guard, p = devices.ClaimGuard(limit=10), devices.start_pairing(store)
    tripped = [guard.failed(store) for _ in range(10)]
    assert tripped == [False] * 9 + [True]
    assert devices.claim(store, p.code, "phone") is None                         # the real one died with them


def test_the_pairing_link_carries_the_server_and_the_code():
    link = devices.pair_link("http://192.168.1.5:8765", "ABCD2345")
    assert link == "ocrmd://pair?server=http%3A%2F%2F192.168.1.5%3A8765&code=ABCD2345"
    assert devices.server_url("192.168.1.5", 8765) == "http://192.168.1.5:8765"
    assert devices.server_url("::1", 8765).startswith("http://")


# --- access -----------------------------------------------------------------------------------------------
def test_lan_mode_lets_this_machine_in_and_asks_everyone_else_for_a_token(paired_app):
    local, phone = paired_app
    assert local.get("/api/library").status_code == 200                          # the browser UI is unchanged
    assert local.get("/api/v1/documents").status_code == 200
    for path in ("/api/library", "/api/v1/documents", "/api/v1/server", "/"):
        r = phone.get(path)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer", path
    ping = phone.get("/api/v1/ping").json()                                      # open: is this ocrmd, must I pair
    assert ping["ocrmd"] and ping["access"] == "paired" and not ping["local"] and not ping["paired"]
    out, auth = _pair(local, phone)
    assert out["server"]["api"] == v1.API_VERSION and out["device"]["name"] == "Test phone"
    assert phone.get("/api/v1/documents", headers=auth).status_code == 200
    assert phone.get("/api/library", headers=auth).status_code == 200         # a paired device is the operator's
    assert phone.get("/api/v1/me", headers=auth).json()["platform"] == "android"
    assert phone.get("/api/v1/ping", headers=auth).json()["paired"]


def test_a_paired_device_cannot_manage_devices(paired_app):
    local, phone = paired_app
    out, auth = _pair(local, phone)
    assert phone.get("/api/v1/devices", headers=auth).status_code == 403
    assert phone.post("/api/v1/devices/pairings", headers=auth, json={}).status_code == 403
    assert phone.delete(f"/api/v1/devices/{out['device']['id']}", headers=auth).status_code == 403
    assert local.post("/api/v1/devices/pairings", headers=auth, json={}).status_code == 403   # not even from here


def test_a_revoked_or_unknown_token_is_refused_in_every_mode(root, paired_app):
    local, phone = paired_app
    out, auth = _pair(local, phone)
    revoked = local.delete(f"/api/v1/devices/{out['device']['id']}").json()
    assert revoked["revoked_at"] and phone.get("/api/v1/documents", headers=auth).status_code == 401
    with TestClient(create_app(root), client=LOCAL) as plain:                   # local mode checks a token it is sent
        assert plain.get("/api/v1/documents").status_code == 200
        assert plain.get("/api/v1/documents", headers=auth).status_code == 401
        assert plain.get("/api/v1/documents", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_this_machine_dialling_its_own_lan_address_counts_as_local(root):
    with TestClient(create_app(root, access="paired"), client=("testserver", 50003)) as same_host:
        assert same_host.get("/api/v1/documents").status_code == 200
        assert same_host.get("/api/v1/ping").json()["local"]


def test_wrong_codes_over_http_are_refused_and_eventually_cancel_open_codes(paired_app):
    local, phone = paired_app
    code = local.post("/api/v1/devices/pairings", json={}).json()["code"]
    for _ in range(10):
        r = phone.post("/api/v1/pair", json={"code": "WRNG-CDE2", "name": "x"})
        assert r.status_code == 403
    assert "cancelled" in r.json()["detail"]
    assert phone.post("/api/v1/pair", json={"code": code, "name": "x"}).status_code == 403
    assert [d["pending"] for d in local.get("/api/v1/devices").json()] == []


def test_the_pairing_answer_has_a_link_and_a_qr_code_when_segno_is_installed(paired_app):
    local, _ = paired_app
    p = local.post("/api/v1/devices/pairings", json={"name": "tablet"}).json()
    assert p["link"].startswith("ocrmd://pair?server=") and p["code"].replace("-", "") in p["link"]
    pytest.importorskip("segno")
    assert p["qr_svg"].startswith("<svg")


# --- documents, captures and review ------------------------------------------------------------------------
def test_a_document_is_created_once_with_its_title(paired_app):
    local, phone = paired_app
    _, auth = _pair(local, phone)
    r = phone.post("/api/v1/documents", headers=auth, json={"slug": "field-notes", "title": "Field Notes", "tags": ["trip"]})
    assert r.status_code == 201 and r.json()["title"] == "Field Notes" and r.json()["pages"] == 0
    assert phone.post("/api/v1/documents", headers=auth, json={"slug": "field-notes"}).status_code == 409
    assert phone.post("/api/v1/documents", headers=auth, json={"slug": "../etc"}).status_code == 422
    assert [d["slug"] for d in phone.get("/api/v1/documents", headers=auth).json()] == ["field-notes"]


@needs_ocr
def test_a_phone_capture_becomes_a_page_attributed_to_the_device_and_a_retry_is_the_same_job(root, paired_app):
    local, phone = paired_app
    _, auth = _pair(local, phone)
    shot = _jpeg(on_desk(text_page(LINES)))
    send = lambda: phone.post("/api/v1/documents/phone-book/captures", headers=auth,
                              files={"file": ("shot.jpg", shot, "image/jpeg")}, data={"client_id": "cap-1", "no_llm": "true"})
    first = send()
    assert first.status_code == 202, first.text
    assert send().json()["id"] == first.json()["id"]                            # the offline queue retried
    job = _wait(phone, first.json()["id"], auth)
    assert job["status"] == "done", job
    assert job["results"] == [{"file": job["results"][0]["file"], "client_id": "cap-1", "duplicate_of": None,
                               "pages": [{"sequence": 1, "status": "unverified"}], "rejected": [], "error": None}]
    page = phone.get("/api/v1/documents/phone-book/pages/1", headers=auth).json()
    assert page["frontmatter"]["source"]["type"] == "mobile" and page["frontmatter"]["source"]["device"] == "Test phone"
    assert set(page["images"]) >= {"processed", "raw"} and all(u.startswith("/api/v1/") for u in page["images"].values())
    # the same photo again, under a new client id (the server restarted, the queue lost the job): a duplicate, no page
    again = phone.post("/api/v1/documents/phone-book/captures", headers=auth,
                       files={"file": ("shot.jpg", shot, "image/jpeg")}, data={"client_id": "cap-2", "no_llm": "true"})
    dup = _wait(phone, again.json()["id"], auth)["results"][0]
    assert dup["pages"] == [] and dup["duplicate_of"] == {"document": "phone-book", "sequence": 1}
    assert phone.get("/api/v1/documents/phone-book", headers=auth).json()["pages"] == 1
    # a page the phone's document scanner already cropped keeps that in its provenance
    flat = _jpeg(text_page(LINES[:5] + ["a different page entirely, with words of its own"] * 4))
    scan = phone.post("/api/v1/documents/phone-book/captures", headers=auth,
                      files={"file": ("scan.jpg", flat, "image/jpeg")}, data={"origin": "scanner", "no_llm": "true"})
    seq = _wait(phone, scan.json()["id"], auth)["results"][0]["pages"][0]["sequence"]
    src = phone.get(f"/api/v1/documents/phone-book/pages/{seq}", headers=auth).json()["frontmatter"]["source"]
    assert src["type"] == "mobile_scan" and src["device"] == "Test phone"


@needs_ocr
def test_a_phone_photo_is_read_the_way_up_its_exif_says(root, paired_app):
    """Phones store the sensor's pixels and an EXIF orientation; the page must come out upright."""
    from PIL import Image
    local, phone = paired_app
    _, auth = _pair(local, phone)
    upright = on_desk(text_page(LINES))                                         # portrait
    sideways = Image.fromarray(cv2.cvtColor(np.rot90(upright), cv2.COLOR_BGR2RGB))   # stored landscape...
    exif = Image.Exif()
    exif[0x0112] = 6                                                            # ...to be shown rotated 90° CW
    buf = io.BytesIO()
    sideways.save(buf, "JPEG", quality=92, exif=exif)
    r = phone.post("/api/v1/documents/exif/captures", headers=auth,
                   files={"file": ("IMG_0001.jpg", buf.getvalue(), "image/jpeg")}, data={"no_llm": "true"})
    assert _wait(phone, r.json()["id"], auth)["status"] == "done"
    raw = cv2.imread(str(root.raw / f"{Store(root.db_path).page_by_sequence('exif', 1)['raw_sha256']}.jpg"))
    assert raw.shape[0] > raw.shape[1], raw.shape                               # portrait again


def test_a_non_image_upload_is_refused(paired_app):
    local, phone = paired_app
    _, auth = _pair(local, phone)
    r = phone.post("/api/v1/documents/x/captures", headers=auth, files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


@needs_ocr
def test_review_from_the_phone_resolves_signs_off_as_the_device_and_reopens(root, store, paired_app):
    local, phone = paired_app
    _, auth = _pair(local, phone)
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(LINES)), "web-book", "webcam:cam").pages[0]
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown=FLAGGED_MD, page_number="3"), "web-book")
    # the disagreement written in the page dialect, so the test does not depend on which engine misread what
    path = render.page_path(root.documents / "web-book", 1)
    fm, _ = render.read_page(path)
    render.write_page(root.documents / "web-book", 1, fm, 'invoice total ==[44.80]==[^ocrmd-1] paid in full\n\n'
                      '[^ocrmd-1]: ocr="44.60" llm="44.80" — unresolved\n')
    store.set_status(pid, "flagged")
    base = "/api/v1/documents/web-book/pages/1"
    page = phone.get(base, headers=auth).json()
    assert page["status"] == "flagged" and page["flags"] == [{"n": 1, "ocr": "44.60", "llm": "44.80",
                                                              "context": page["flags"][0]["context"]}]
    assert phone.post(f"{base}/signoff", headers=auth).status_code == 409       # not with a disagreement open
    assert phone.post(f"{base}/resolve", headers=auth, json={"n": 999, "chosen": "x"}).status_code == 404
    for f in page["flags"]:
        page = phone.post(f"{base}/resolve", headers=auth, json={"n": f["n"], "chosen": f["ocr"]}).json()
    assert page["unresolved"] == 0
    page = phone.post(f"{base}/signoff", headers=auth).json()
    assert page["status"] == "reviewed" and page["reviewed_by"] == "Test phone"
    page = phone.post(f"{base}/reopen", headers=auth).json()
    assert page["status"] in ("verified", "flagged") and page["reviewed_by"] is None
    detail = phone.get("/api/v1/documents/web-book", headers=auth).json()
    assert detail["page_list"][0]["thumb"].endswith("?max=360") and detail["cover"] == detail["page_list"][0]["thumb"]


@needs_ocr
def test_a_page_image_is_scaled_down_for_a_phone(root, store, paired_app):
    local, phone = paired_app
    _, auth = _pair(local, phone)
    pipeline.ingest_frame(root, store, on_desk(text_page(LINES)), "web-book", "webcam:cam")
    full = phone.get("/api/v1/documents/web-book/pages/1/image/raw", headers=auth)
    small = phone.get("/api/v1/documents/web-book/pages/1/image/raw?max=300", headers=auth)
    assert full.status_code == small.status_code == 200 and small.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(small.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 300
    assert phone.get("/api/v1/documents/web-book/pages/1/image/thumb", headers=auth).status_code == 422
    assert phone.get("/api/v1/documents/web-book/pages/9/image/raw", headers=auth).status_code == 404


@needs_ocr
def test_human_edits_from_any_client_survive_the_next_verify(root, store):
    """edits.py is the one write path: the browser, the phone and the Library all leave the fingerprint alone."""
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(LINES)), "kept", "webcam:cam").pages[0]
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="\n\n".join(LINES), page_number="3"), "kept")
    with Library(settings=root, provider=None) as lib:
        lib.edit("kept", 1, page_label="iii", context={"chapter": "Preface", "ignored": "x"})
        info = lib.page("kept", 1)
        assert info.page_label == "iii" and info.frontmatter["context"]["chapter"] == "Preface"
        assert "ignored" not in info.frontmatter["context"]
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="\n\n".join(LINES), page_number="9"), "kept")
    fm, _ = render.read_page(render.page_path(root.documents / "kept", 1))
    assert fm["page_label"] == "iii" and fm["context"]["chapter"] == "Preface"


# --- the contract -------------------------------------------------------------------------------------------
def test_the_v1_contract_covers_only_v1_and_types_every_json_answer(root):
    doc = v1.openapi(create_app(root))
    assert doc["paths"] and all(p.startswith("/api/v1/") for p in doc["paths"])
    for path, ops in doc["paths"].items():
        for method, op in ops.items():
            ok = next(v for k, v in op["responses"].items() if k.startswith("2"))
            content = ok.get("content", {})
            if "application/json" in content:
                assert content["application/json"]["schema"] != {}, f"{method} {path} has an untyped answer"
            assert op["operationId"].isidentifier()
    assert "Page" in doc["components"]["schemas"] and "devices" not in str(doc["components"]["schemas"].keys())


@needs_ocr
def test_a_hand_edited_page_with_odd_frontmatter_still_answers(root, store, paired_app):
    """The page file is the person's to edit; a typed answer must not turn their YAML into a 500."""
    local, phone = paired_app
    _, auth = _pair(local, phone)
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(LINES)), "odd", "webcam:cam").pages[0]
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="\n\n".join(LINES)), "odd")
    fm, body = render.read_page(render.page_path(root.documents / "odd", 1))
    fm["verification"]["agreement"], fm["context"] = "n/a", "see notes"
    render.write_page(root.documents / "odd", 1, fm, body)
    r = phone.get("/api/v1/documents/odd/pages/1", headers=auth)
    assert r.status_code == 200 and r.json()["agreement"] is None and r.json()["context"] == {}
