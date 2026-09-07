"""The inventory: document.yaml mirrored into the store, and search over every page ever scanned.

Pages are seeded straight into the store and the page files here — no OCR engine needed — because
what is under test is the join and the index, not the pipeline that fills them."""
from __future__ import annotations
import os, time

from ocrmd import library, render
from ocrmd.store import fts_query, like_escape


def seed(root, store, slug, pages, title=None, **meta):
    """`pages` = [(label, body), ...]. Writes rows and page files the way the pipeline would."""
    store.ensure_document(slug)
    render.ensure_document_meta(root.documents / slug, slug, title=title, **meta)
    for i, (label, body) in enumerate(pages, 1):
        pid = f"{slug}-{i:02d}".ljust(16, "0")
        seq = store.add_page(pid, slug, f"sha-{pid}", "0" * 16)
        fm = {"ocrmd": 1, "id": pid, "document": slug, "sequence": seq, "captured_at": "2026-09-06T10:00:00", "kind": "book_page",
              "page_label": label, "context": {"title": title}, "verification": {"status": "verified", "agreement": 0.98,
                                                                                    "overlap": 0.99, "flagged_spans": 0},
              "ocr": {"engine": "fake@1", "confidence": 0.97}, "llm": {"provider": "anthropic", "model": "fake"}}
        render.write_page(root.documents / slug, seq, fm, body)
        store.set_status(pid, "verified", label)
    return slug


class TestCatalogue:
    def test_document_yaml_is_mirrored_into_the_store(self, root, store):
        seed(root, store, "austen", [("1", "It is a truth universally acknowledged."), ("2", "My dear Mr Bennet.")],
             title="Pride and Prejudice", tags=["novel"])
        p = render.document_meta_path(root.documents / "austen")
        meta = render.load_document_meta(root.documents / "austen")
        meta.update(author=["Jane Austen"], year=1813, isbn="978-0-14-143951-8")
        import yaml
        p.write_text(yaml.safe_dump(meta))
        assert library.sync_document(root, store, "austen") is True
        assert library.sync_document(root, store, "austen") is False          # unchanged file: one stat, no rewrite
        row = store.document_row("austen")
        assert row["title"] == "Pride and Prejudice" and row["author"] == "Jane Austen" and row["year"] == 1813
        assert row["isbn"] == "978-0-14-143951-8" and row["tags"] == "novel"

    def test_the_catalogue_row_knows_progress_and_printed_range(self, root, store):
        seed(root, store, "austen", [("iv", "a"), ("1", "b"), ("2", "c")], title="P&P")
        [d] = library.catalog(root, store)
        assert d.slug == "austen" and d.title == "P&P" and d.pages == 3 and d.done == 3
        assert d.page_labels == "iv–2" and d.built is False and d.needs_attention == 0
        assert "P&P" in d.describe() and d.to_dict()["done"] == 3

    def test_a_year_written_as_prose_still_becomes_a_number(self, root, store):
        seed(root, store, "b", [("1", "x")], title="B")
        import yaml
        p = render.document_meta_path(root.documents / "b")
        meta = render.load_document_meta(root.documents / "b"); meta["year"] = "1994 (2nd printing)"
        p.write_text(yaml.safe_dump(meta))
        library.sync_document(root, store, "b")
        assert store.document_row("b")["year"] == 1994

    def test_a_broken_yaml_does_not_break_the_inventory(self, root, store):
        seed(root, store, "b", [("1", "x")], title="B")
        render.document_meta_path(root.documents / "b").write_text("title: [unclosed")
        library.sync_document(root, store, "b")
        assert library.info(root, store, "b").pages == 1


class TestSearch:
    def test_a_word_finds_its_page(self, root, store):
        seed(root, store, "austen", [("1", "It is a truth universally acknowledged that a single man"),
                                     ("2", "My dear Mr Bennet said his lady to him one day")])
        seed(root, store, "other", [("9", "Netherfield Park is let at last")])
        hits = library.search(root, store, "Bennet")
        assert [(h.document, h.sequence) for h in hits] == [("austen", 2)]
        assert "[Bennet]" in hits[0].snippet and hits[0].page_label == "2" and hits[0].status == "verified"
        assert library.search(root, store, "Netherfield", doc="austen") == []
        assert library.search(root, store, "Netherfield")[0].document == "other"

    def test_an_edited_page_is_reindexed_by_its_mtime(self, root, store):
        seed(root, store, "d", [("1", "alpha bravo charlie")])
        assert library.search(root, store, "charlie")
        p = render.page_path(root.documents / "d", 1)
        fm, _ = render.read_page(p)
        render.write_page(root.documents / "d", 1, fm, "alpha bravo delta")
        os.utime(p, (time.time() + 5, time.time() + 5))     # a same-second rewrite still counts as a change
        assert library.search(root, store, "charlie") == []
        assert library.search(root, store, "delta")[0].sequence == 1
        assert library.reindex(root, store, "d") == 0        # up to date: nothing re-read

    def test_a_deleted_page_leaves_the_index(self, root, store):
        seed(root, store, "d", [("1", "alpha"), ("2", "bravo")])
        library.reindex(root, store)
        assert store.indexed_count() == 2
        store.delete_page(store.page_by_sequence("d", 2)["id"])
        assert store.indexed_count() == 1 and library.search(root, store, "bravo") == []

    def test_phrases_prefixes_and_punctuation(self, root, store):
        seed(root, store, "d", [("1", "the C.R.S. § 18-3-102 says ridership will rise")])
        assert library.search(root, store, "rider*")                  # prefix
        assert library.search(root, store, '"ridership will rise"')  # phrase
        assert library.search(root, store, "18-3-102")                # a hyphenated token is text, not NOT
        assert fts_query('ridership fare*') == '"ridership" "fare"*'
        assert like_escape("100%_x") == "100\\%\\_x"

    def test_the_substring_fallback_when_sqlite_has_no_fts5(self, root, store):
        seed(root, store, "d", [("1", "It is a truth universally acknowledged that a single man in possession")])
        store.fts = False
        [hit] = library.search(root, store, "universally")
        assert hit.sequence == 1 and "[universally]" in hit.snippet and hit.snippet.startswith("…") is False
        assert library.search(root, store, "nowhere") == []

    def test_an_empty_query_finds_nothing(self, root, store):
        seed(root, store, "d", [("1", "x")])
        assert library.search(root, store, "   ") == []
