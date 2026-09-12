"""Exports: the same pages in other shapes, with the page ids kept and the disagreements not silently resolved."""
from __future__ import annotations
import json

import pytest

from ocrmd import render
from ocrmd.export import FORMATS, export_document, text as text_mod, jsonl as jsonl_mod
from test_library import seed

FLAGGED = ("## Chapter 1\n\nThe **mayor's** transit proposal ==[assumes]==[^ocrmd-1] that ridership will rise.\n\n"
           "![figure 1](../figures/x-1.png)\n\n[^ocrmd-1]: ocr=\"assumed\" llm=\"assumes\" — unresolved\n")


class TestText:
    def test_marks_stay_visible_by_default(self):
        out = text_mod.page_text(FLAGGED)
        assert "assumes[?assumed]" in out and "[^ocrmd-1]" not in out and "— unresolved" not in out
        assert out.startswith("Chapter 1\n") and "**" not in out and "[figure: figure 1]" in out

    def test_one_side_can_be_chosen_on_purpose(self):
        assert "assumes that" in text_mod.page_text(FLAGGED, flags="llm") and "assumed" not in text_mod.page_text(FLAGGED, flags="llm")
        assert "assumed that" in text_mod.page_text(FLAGGED, flags="ocr")
        with pytest.raises(ValueError):
            text_mod.page_text(FLAGGED, flags="nope")

    def test_the_document_has_a_header_and_a_separator_per_page(self, root, store):
        seed(root, store, "d", [("iv", "front matter"), ("1", FLAGGED)], title="A Book")
        import yaml
        p = render.document_meta_path(root.documents / "d")
        meta = render.load_document_meta(root.documents / "d"); meta["author"] = ["Someone"]; meta["year"] = 2001
        p.write_text(yaml.safe_dump(meta))
        out = text_mod.document_text(root, store, "d")
        assert out.startswith("A Book\nSomeone\nyear: 2001\n")
        assert "------------ page 0001 (p. iv) [verified] ------------" in out
        assert "page 0002 (p. 1)" in out and "assumes[?assumed]" in out


class TestJsonl:
    def test_one_row_per_page_with_identity_and_verdict(self, root, store):
        seed(root, store, "d", [("1", FLAGGED), ("2", "plain page")], title="A Book")
        path = export_document(root, store, "d", "jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["sequence"] for r in rows] == [1, 2]
        r = rows[0]
        assert r["document"] == "d" and r["page_id"] == store.page_by_sequence("d", 1)["id"]
        assert r["text"] == FLAGGED.rstrip() + "\n" or r["text"].strip() == FLAGGED.strip()
        assert r["unresolved"] == 1 and r["verification"]["agreement"] == 0.98 and r["status"] == "verified"
        assert r["context"]["title"] == "A Book" and r["llm"]["model"] == "fake"
        assert store.doc_exports("d")[0]["format"] == "jsonl"


class TestFormats:
    def test_every_format_writes_under_the_export_dir_and_is_recorded(self, root, store):
        seed(root, store, "d", [("1", "hello world")], title="A Book")
        for fmt in FORMATS:
            path = export_document(root, store, "d", fmt)
            assert path == root.export_dir / "d" / f"d.{fmt}" and path.stat().st_size > 0
        assert sorted(r["format"] for r in store.doc_exports("d")) == sorted(FORMATS)
        assert json.loads((root.export_dir / "d" / "d.json").read_text())["pages"][0]["body"].strip() == "hello world"
        assert "# A Book" in (root.export_dir / "d" / "d.md").read_text()

    def test_an_explicit_path_wins(self, root, store, tmp_path):
        seed(root, store, "d", [("1", "hello")])
        out = export_document(root, store, "d", "txt", tmp_path / "book.txt")
        assert out == tmp_path / "book.txt" and "hello" in out.read_text()

    def test_unknown_things_are_refused(self, root, store):
        seed(root, store, "d", [("1", "hello")])
        with pytest.raises(ValueError, match="unknown export format"):
            export_document(root, store, "d", "docx")
        with pytest.raises(KeyError):
            export_document(root, store, "nope", "md")

    def test_the_jsonl_rows_are_reusable_without_writing(self, root, store):
        seed(root, store, "d", [("1", "hello")])
        assert jsonl_mod.rows(root, store, "d")[0]["text"].strip() == "hello"
