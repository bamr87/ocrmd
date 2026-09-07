"""Destinations: what leaves the machine, what a receiver declares, and what happens when it cannot.

This is ocrmd's only outbound path, so the tests are mostly about restraint — nothing is sent unless
a destination is configured, the page's identity always travels with the text, the photograph never
does, options are checked against the receiver's own descriptor before anything leaves, and a receiver
that is down is reported rather than swallowed.
"""
from __future__ import annotations
import io, json
from urllib.error import HTTPError, URLError

import pytest
from fastapi.testclient import TestClient

from conftest import FakeProvider, on_desk, text_page
from ocrmd import pipeline, render, sinks
from ocrmd.config import SinkConfig, load_settings
from ocrmd.ocr import available_engine
from ocrmd.web.app import create_app

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")
LINES = ["the mayor's transit proposal assumes that ridership will rise once fares are cut"] * 6

DESCRIPTOR = {"ocrmd_sink": 1, "name": "Study platform", "description": "turns a page into a practice question",
              "accepts": ["page", "region"],
              "options": [{"key": "hint", "label": "which item", "type": "string", "help": "e.g. question 15"},
                          {"key": "section", "label": "section", "type": "enum", "choices": ["LR", "RC", "LG"]},
                          {"key": "max_items", "label": "max items", "type": "integer", "default": 5},
                          {"key": "notify", "type": "boolean"}]}
ACCEPTED = {"summary": "1 question(s) queued for review (key B)",
            "items": [{"id": "prop-abc123", "title": "Which one…", "detail": "LR/med · key B", "status": "pending"}],
            "warnings": ["rejected at the duplicate gate: …"]}


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status
    def read(self): return json.dumps(self.payload).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeHttp:
    """Stands in for urllib: answers the descriptor GET and the envelope POST, remembers every request."""
    def __init__(self, descriptor=DESCRIPTOR, response=ACCEPTED, error=None, get_error=None):
        self.descriptor, self.response, self.error, self.get_error = descriptor, response, error, get_error
        self.requests: list[dict] = []

    def __call__(self, req, timeout=None):
        self.requests.append({"method": req.get_method(), "url": req.full_url, "headers": dict(req.headers),
                              "body": json.loads(req.data.decode()) if req.data else None, "timeout": timeout})
        if req.get_method() == "GET":
            if self.get_error is not None:
                raise self.get_error
            return _Resp(self.descriptor)
        if self.error is not None:
            raise self.error
        return _Resp(self.response, 202)

    @property
    def posts(self):
        return [r for r in self.requests if r["method"] == "POST"]


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(sinks.urllib.request, "urlopen", fake)
    return fake


@pytest.fixture
def sink(root):
    root.sinks = [SinkConfig(name="study", url="http://localhost:8000/api/intake/", api_key="s3cret",
                             options={"section": "LR"})]
    return sinks.get(root, "study")


@pytest.fixture
def seeded(root, store):
    """A document with one page whose page file is on disk."""
    slug = "book"
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(LINES)), slug, "webcam:cam").pages[0]
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="\n\n".join(LINES)), slug)
    render.ensure_document_meta(root.documents / slug, slug, title="A Prep Book", tags=["study"])
    return slug, store.page_by_sequence(slug, 1)


class TestConfiguration:
    def test_nothing_configured_means_no_destination(self, root):
        assert sinks.configured(root) is False
        with pytest.raises(sinks.SinkError, match="add a \\[\\[sinks\\]\\] table"):
            sinks.get(root, "")

    def test_the_environment_takes_the_list_as_json(self, tmp_path):
        s = load_settings(root=tmp_path, env={"OCRMD_SINKS": json.dumps([{"name": "a", "url": "http://a/", "timeout": 5}])})
        assert [x.name for x in s.sinks] == ["a"] and s.sinks[0].timeout == 5.0

    def test_a_broken_list_fails_fast_naming_the_problem(self, tmp_path):
        with pytest.raises(RuntimeError, match="sinks"):
            load_settings(root=tmp_path, env={"OCRMD_SINKS": "[{not json"})
        with pytest.raises(RuntimeError, match="name and a url"):
            load_settings(root=tmp_path, env={"OCRMD_SINKS": json.dumps([{"name": "", "url": "http://a/"}])})

    def test_the_only_destination_may_be_unnamed_on_the_command_line(self, root, sink):
        assert sinks.get(root, "").name == "study"
        with pytest.raises(sinks.SinkError, match="configured: study"):
            sinks.get(root, "other")


class TestDescriptor:
    def test_the_receiver_declares_its_options(self, sink, http):
        d = sink.describe()
        assert d["reachable"] is True and d["name"] == "Study platform"
        assert [o["key"] for o in d["options"]] == ["hint", "section", "max_items", "notify"]
        assert d["defaults"] == {"section": "LR"}           # from ocrmd.toml, shown beside the receiver's own
        assert http.requests[0]["method"] == "GET" and http.requests[0]["headers"]["X-api-key"] == "s3cret"

    def test_the_descriptor_is_cached(self, sink, http):
        sink.describe(); sink.describe()
        assert len(http.requests) == 1

    def test_a_receiver_with_an_intake_but_no_descriptor_still_works(self, sink, monkeypatch):
        fake = FakeHttp(get_error=HTTPError("u", 405, "Method Not Allowed", {}, io.BytesIO(b"")))
        monkeypatch.setattr(sinks.urllib.request, "urlopen", fake)
        d = sink.describe()
        assert d["reachable"] is True and d["options"] == [] and "405" in d["error"]

    def test_a_receiver_that_is_down_is_reported_not_hidden(self, sink, monkeypatch):
        monkeypatch.setattr(sinks.urllib.request, "urlopen", FakeHttp(get_error=URLError("Connection refused")))
        d = sink.describe()
        assert d["reachable"] is False and "unreachable" in d["error"]


class TestOptions:
    def test_typed_by_the_descriptor(self):
        out = sinks.coerce_options(DESCRIPTOR, {"hint": "question 15", "max_items": "3", "notify": "yes", "extra": "kept"})
        assert out == {"hint": "question 15", "max_items": 3, "notify": True, "extra": "kept"}

    def test_a_value_outside_an_enum_never_reaches_the_wire(self):
        with pytest.raises(sinks.SinkError, match="section must be one of LR, RC, LG"):
            sinks.coerce_options(DESCRIPTOR, {"section": "WS"})

    def test_blank_means_let_the_receiver_decide(self):
        assert sinks.coerce_options(DESCRIPTOR, {"section": "", "hint": None}) == {}

    def test_command_line_pairs(self):
        assert sinks.parse_option_args(["hint=question 15", "max_items=2"]) == {"hint": "question 15", "max_items": "2"}
        with pytest.raises(sinks.SinkError, match="key=value"):
            sinks.parse_option_args(["nonsense"])


@needs_ocr
class TestWhatTravels:
    def test_the_page_body_goes_and_the_page_id_goes_with_it(self, root, store, seeded, sink, http):
        slug, row = seeded
        out = sinks.send_page(root, slug, row, sink, {"hint": "question 15"})
        post = http.posts[0]
        assert post["url"] == "http://localhost:8000/api/intake/"
        body = post["body"]
        assert body["ocrmd"] == 1 and body["kind"] == "page"
        assert "transit proposal" in body["text"]
        assert body["options"] == {"section": "LR", "hint": "question 15"}     # toml default + the operator's
        prov = body["provenance"]
        assert prov["tool"] == "ocrmd" and prov["document"] == slug and prov["page"] == 1 and prov["page_id"] == row["id"]
        assert prov["image_url"].endswith(f"/api/documents/{slug}/pages/1/image/flat")
        assert prov["page_url"].endswith(f"#/doc/{slug}/page/1")
        assert prov["unresolved"] == 0 and prov["status"] == row["status"]
        assert body["document"]["title"] == "A Prep Book" and body["document"]["tags"] == ["study"]
        assert out.items[0]["id"] == "prop-abc123" and out.summary.startswith("1 question(s) queued")
        assert out.warnings == ["rejected at the duplicate gate: …"] and out.ok

    def test_the_photograph_does_not_travel(self, root, store, seeded, sink, http):
        """Only the reconciled text goes out. The image is a link back, not a payload: a receiver with
        the picture and a vision model would re-read the page single-pass and overrule the cross-check."""
        slug, row = seeded
        sinks.send_page(root, slug, row, sink)
        body = http.posts[0]["body"]
        assert set(body) == {"ocrmd", "kind", "text", "provenance", "document", "options"}
        assert "base64" not in json.dumps(body)
        assert len(json.dumps(body)) < len(body["text"]) + 2500

    def test_a_region_reading_travels_with_its_box(self, root, store, seeded, sink, http):
        slug, row = seeded
        sinks.send_page(root, slug, row, sink, text="15. A boxed question.", bbox=[0.1, 0.2, 0.3, 0.4])
        body = http.posts[0]["body"]
        assert body["kind"] == "region" and body["text"] == "15. A boxed question."
        assert body["provenance"]["bbox"] == [0.1, 0.2, 0.3, 0.4] and "region.png" in body["provenance"]["image_url"]

    def test_a_receiver_that_takes_only_pages_refuses_a_region_here(self, root, store, seeded, sink, monkeypatch):
        fake = FakeHttp(descriptor={**DESCRIPTOR, "accepts": ["page"]})
        monkeypatch.setattr(sinks.urllib.request, "urlopen", fake)
        slug, row = seeded
        with pytest.raises(sinks.SinkError, match="does not take a region"):
            sinks.send_page(root, slug, row, sink, text="x", bbox=[0, 0, 1, 1])
        assert fake.posts == []

    def test_credentials_are_sent_as_headers(self, root, store, seeded, http):
        root.sinks = [SinkConfig(name="s", url="http://h/", api_key="k", bearer="tok")]
        slug, row = seeded
        sinks.send_page(root, slug, row, sinks.get(root, "s"))
        h = http.posts[0]["headers"]
        assert h["X-api-key"] == "k" and h["Authorization"] == "Bearer tok"

    def test_a_dry_run_shows_the_envelope_and_sends_nothing(self, root, store, seeded, sink, http):
        slug, row = seeded
        out = sinks.send_page(root, slug, row, sink, {"max_items": "2"}, dry_run=True)
        assert out.dry_run and out.ok and out.raw["options"] == {"section": "LR", "max_items": 2}
        assert out.raw["provenance"]["page_id"] == row["id"]
        assert http.posts == []


class TestDelivery:
    def test_a_receiver_answering_in_its_own_words_gets_an_honest_summary(self):
        d = sinks.parse_delivery("s", "u", 202, {"accepted": [{"id": 1}], "notes": "x"})
        assert d.summary == "delivered (HTTP 202)" and d.items == [] and d.raw["notes"] == "x"
        d = sinks.parse_delivery("s", "u", 200, {"items": [{"title": "a"}, "not-a-dict"]})
        assert d.summary == "1 item(s) taken in" and len(d.items) == 1

    def test_describe_is_a_few_terminal_lines(self):
        d = sinks.parse_delivery("s", "u", 202, ACCEPTED)
        text = d.describe()
        assert text.splitlines()[0] == ACCEPTED["summary"]
        assert "pending" in text and "prop-abc123" in text and "! rejected" in text


class TestFailuresAreReported:
    def test_an_unreachable_receiver_says_so(self, root, sink, monkeypatch):
        monkeypatch.setattr(sinks.urllib.request, "urlopen", FakeHttp(error=URLError("Connection refused")))
        with pytest.raises(sinks.SinkError, match="cannot reach study"):
            sink.send({"text": "x", "provenance": {}}, {})

    def test_a_refusal_carries_the_receivers_reason(self, root, sink, monkeypatch):
        err = HTTPError("u", 422, "Unprocessable", {}, io.BytesIO(json.dumps({"detail": "bad section"}).encode()))
        monkeypatch.setattr(sinks.urllib.request, "urlopen", FakeHttp(error=err))
        with pytest.raises(sinks.SinkError, match="422.*bad section"):
            sink.send({"text": "x", "provenance": {}}, {})

    def test_a_page_without_a_file_is_an_error_not_an_empty_send(self, root, store):
        store.ensure_document("d")
        store.add_page("pid1", "d", "sha", "0" * 16)
        with pytest.raises(sinks.SinkError, match="no page file yet"):
            sinks.page_text(root, "d", store.page_by_sequence("d", 1))


class TestNothingWasWritten:
    @needs_ocr
    def test_sending_marks_no_stage_and_changes_no_file(self, root, store, seeded, sink, http):
        """The same contract as `read_region`: this is a send, not a stage."""
        slug, row = seeded
        path = render.page_path(root.documents / slug, 1)
        before, mtime = path.read_text(), path.stat().st_mtime_ns
        stages_before = {(h["stage"], h["version"]) for h in store.stage_history(row["id"])}
        sinks.send_page(root, slug, row, sink, {"hint": "question 15"})
        assert path.read_text() == before and path.stat().st_mtime_ns == mtime
        assert {(h["stage"], h["version"]) for h in store.stage_history(row["id"])} == stages_before
        assert store.page_by_sequence(slug, 1)["status"] == row["status"]


class TestEntryPoints:
    def test_a_python_sink_registers_through_the_entry_point_group(self, root, monkeypatch):
        class Notebook:
            name = "notebook"
            def __init__(self, s): self.sent = []
            def describe(self): return {"name": "Notebook", "options": [], "accepts": ["page", "region"]}
            def send(self, envelope, options=None):
                self.sent.append(envelope)
                return sinks.Delivery(sink=self.name, url="", status=200, summary="filed")
        class EP:
            name = "notebook"
            def load(self): return Notebook
        monkeypatch.setattr(sinks, "entry_points", lambda group: [EP()])
        have = sinks.registry(root)
        assert [x.name for x in have] == ["notebook"]
        assert have[0].send({"text": "x"}).summary == "filed"


@needs_ocr
class TestThroughTheWebApi:
    @pytest.fixture
    def client(self, root, store):
        with TestClient(create_app(root)) as c:
            yield c

    def test_no_destinations_means_an_empty_list(self, client):
        assert client.get("/api/sinks").json() == []

    def test_destinations_come_with_their_descriptors(self, client, root, sink, http):
        body = client.get("/api/sinks").json()
        assert body[0]["name"] == "study" and body[0]["descriptor"]["name"] == "Study platform"
        assert body[0]["descriptor"]["defaults"] == {"section": "LR"}

    def test_settings_never_expose_a_credential(self, client, root, sink):
        body = client.get("/api/settings").json()
        assert body["sinks"][0]["api_key"] == "•••" and "s3cret" not in json.dumps(body)

    def test_post_returns_what_the_receiver_did(self, client, root, seeded, sink, http):
        slug, _row = seeded
        r = client.post(f"/api/documents/{slug}/pages/1/send", json={"sink": "study", "options": {"hint": "question 15"}})
        assert r.status_code == 200, r.text
        assert r.json()["items"][0]["id"] == "prop-abc123" and r.json()["ok"] is True
        assert http.posts[0]["body"]["options"]["hint"] == "question 15"

    def test_an_unreachable_receiver_is_a_502_not_a_500(self, client, root, seeded, sink, monkeypatch):
        """This server is fine; the page did not get through to the other one."""
        monkeypatch.setattr(sinks.urllib.request, "urlopen", FakeHttp(get_error=URLError("down"), error=URLError("down")))
        slug, _row = seeded
        r = client.post(f"/api/documents/{slug}/pages/1/send", json={"sink": "study"})
        assert r.status_code == 502 and "cannot reach study" in r.json()["detail"]

    def test_an_unknown_destination_is_a_404(self, client, root, seeded, sink, http):
        slug, _row = seeded
        r = client.post(f"/api/documents/{slug}/pages/1/send", json={"sink": "nope"})
        assert r.status_code == 404 and "configured: study" in r.json()["detail"]
