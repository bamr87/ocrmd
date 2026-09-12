"""The embedding API: ocrmd as a library, driven the way a host program would drive it."""
from __future__ import annotations
import cv2
import pytest

from conftest import FakeProvider, on_desk, text_page
from ocrmd.api import Library, IngestReport
from ocrmd.ocr import available_engine
from ocrmd.sinks import Delivery

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")
LINES_A = ["It is a truth universally acknowledged that a single man in possession of a good fortune"] * 6
LINES_B = ["My dear Mr Bennet said his lady to him one day have you heard that Netherfield Park is let"] * 6


@pytest.fixture
def lib(tmp_path):
    # the fake model echoes the OCR draft, so each page's body is its own text (page two must stay page two)
    with Library(tmp_path / "scans", provider=FakeProvider(page_number="12")) as lib:
        yield lib


@pytest.fixture
def photos(tmp_path):
    d = tmp_path / "photos"; d.mkdir()
    cv2.imwrite(str(d / "p1.png"), on_desk(text_page(LINES_A)))
    cv2.imwrite(str(d / "p2.png"), on_desk(text_page(LINES_B)))
    (d / "notes.txt").write_text("not a page")
    return d


@needs_ocr
class TestIngestAndRead:
    def test_a_folder_becomes_a_document(self, lib, photos):
        seen = []
        report = lib.ingest([photos], "austen", title="Pride and Prejudice", tags=["novel"], on_page=seen.append)
        assert isinstance(report, IngestReport) and report.files == 2 and len(report.pages) == 2
        assert [p.sequence for p in seen] == [1, 2]
        assert "2 page(s) from 2 file(s)" in report.summary()
        doc = lib.document("austen")
        assert doc.title == "Pride and Prejudice" and doc.tags == ["novel"] and doc.pages == 2 and doc.built
        page = lib.page("austen", 1)
        assert page.status == "verified" and "truth universally" in page.text and page.page_label == "12"
        assert page.path and page.path.exists() and {"flat", "processed", "raw"} <= set(page.images)
        assert page.frontmatter["llm"]["model"] == "fake-model" and page.done
        assert [d.slug for d in lib.documents()] == ["austen"]

    def test_ocr_only_when_no_provider(self, tmp_path, photos):
        with Library(tmp_path / "scans", provider=None) as lib:
            report = lib.ingest([photos / "p1.png"], "book")
            assert report.pages[0].status == "unverified" and report.pages[0].frontmatter.get("llm") is None
            assert "truth universally" in report.pages[0].text.lower()

    def test_llm_false_overrides_the_provider_for_one_call(self, lib, photos):
        report = lib.ingest([photos / "p1.png"], "book", llm=False)
        assert report.pages[0].status == "unverified"
        assert lib.verify("book")[0].status == "verified"          # the LLM pass, filled in later

    def test_bytes_are_a_capture_too(self, lib):
        ok, buf = cv2.imencode(".png", on_desk(text_page(LINES_A)))
        report = lib.ingest_image(buf.tobytes(), "book", source="api:test")
        assert len(report.pages) == 1 and report.pages[0].frontmatter["source"]["type"] == "file"
        assert lib.ingest_image(b"not an image", "book").unreadable == ["<bytes>"]

    def test_a_duplicate_is_counted_not_repeated(self, lib, photos):
        lib.ingest([photos / "p1.png"], "book")
        again = lib.ingest([photos / "p1.png"], "book")
        assert again.duplicates == 1 and again.pages == [] and lib.document("book").pages == 1

    def test_exports_search_and_the_envelope(self, lib, photos):
        lib.ingest([photos], "austen", title="P&P")
        for fmt in ("md", "txt", "json", "jsonl"):
            assert lib.export("austen", fmt).exists()
        assert lib.document("austen").exports.keys() == {"md", "txt", "json", "jsonl"}
        hits = lib.search("Bennet")                      # a word the engines read whole on this synthetic page
        assert [(h.document, h.sequence) for h in hits] == [("austen", 2)]
        env = lib.envelope("austen", 1)
        assert env["kind"] == "page" and env["provenance"]["page_id"] == lib.page("austen", 1).id
        assert env["document"]["title"] == "P&P" and "truth universally" in env["text"]
        region = lib.read_region("austen", 1, [0.05, 0.05, 0.9, 0.3])
        assert region.text.strip()

    def test_send_goes_through_a_destination_object(self, lib, photos):
        lib.ingest([photos / "p1.png"], "book")
        sent = []

        class Sink:
            name = "notebook"
            def describe(self): return {"name": "Notebook", "accepts": ["page"], "options": [{"key": "shelf", "type": "string"}]}
            def send(self, envelope, options=None):
                sent.append((envelope, options))
                return Delivery(sink=self.name, url="", status=200, summary="filed")

        out = lib.send("book", 1, Sink(), {"shelf": "law"})
        assert out.summary == "filed" and sent[0][1] == {"shelf": "law"} and sent[0][0]["provenance"]["page"] == 1
        assert lib.send("book", 1, Sink(), dry_run=True).dry_run

    def test_deleting(self, lib, photos):
        lib.ingest([photos], "austen")
        out = lib.delete_pages("austen", [2])
        assert out.pages == [2] and lib.document("austen").pages == 1
        lib.delete_document("austen")
        assert lib.documents() == []
        with pytest.raises(KeyError):
            lib.page("austen", 1)


def test_the_root_is_created_and_the_doctor_answers(tmp_path):
    with Library(tmp_path / "fresh", provider=None) as lib:
        assert (tmp_path / "fresh" / "ocrmd.db").exists() and lib.documents() == []
        rep = lib.doctor()
        assert rep["library"]["documents"] == 0 and rep["sinks"] == []
