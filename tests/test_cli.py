"""CLI surface through typer's runner, against a temporary root."""
from __future__ import annotations
import pytest
from typer.testing import CliRunner

from conftest import FakeProvider, on_desk, text_page
from ocrmd import cli, pipeline, render
from ocrmd.ocr import available_engine

runner = CliRunner()
needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")


@pytest.fixture
def cli_root(root, monkeypatch):
    monkeypatch.setattr(cli, "S", root)
    return root


def test_help_lists_every_command():
    out = runner.invoke(cli.app, ["--help"]).output
    for cmd in ("scan", "ingest", "verify", "reprocess", "review", "status", "build", "doctor", "read", "etl"):
        assert cmd in out


@needs_ocr
def test_ingest_status_review_list_and_build(cli_root, tmp_path, monkeypatch):
    import cv2
    img_dir = tmp_path / "in"; img_dir.mkdir()
    cv2.imwrite(str(img_dir / "p1.png"), on_desk(text_page()))
    monkeypatch.setattr(cli, "_provider", lambda *a, **k: FakeProvider(page_number="3", markdown="it is a truth universally acknowledged that a single man in possesion of a good fortune"))
    r = runner.invoke(cli.app, ["ingest", str(img_dir), "--doc", "cli-book", "--title", "CLI Book", "--tag", "t1"])
    assert r.exit_code == 0, r.output
    assert "p1.png -> 1 page(s)" in r.output
    assert render.load_document_meta(cli_root.documents / "cli-book")["title"] == "CLI Book"
    assert (cli_root.documents / "cli-book" / "cli-book.md").exists()

    r = runner.invoke(cli.app, ["status", "--all"])
    assert r.exit_code == 0 and "cli-book" in r.output and "   1      3" in r.output, r.output

    r = runner.invoke(cli.app, ["review", "--doc", "cli-book", "--list"])
    assert r.exit_code == 0, r.output

    r = runner.invoke(cli.app, ["build", "--doc", "cli-book"])
    assert r.exit_code == 0 and "cli-book.md" in r.output

    r = runner.invoke(cli.app, ["verify", "--doc", "cli-book"])
    assert r.exit_code == 0, r.output


@needs_ocr
def test_verify_degrades_after_repeated_service_failures(cli_root, monkeypatch):
    from ocrmd.llm import LLMUnavailable
    from ocrmd.store import Store
    st = Store(cli_root.db_path)
    texts = ["alpha bravo charlie delta echo", "foxtrot golf hotel india juliet kilo", "lima mike november oscar",
             "papa quebec romeo sierra tango uniform victor"]
    for i, words in enumerate(texts):
        pipeline.ingest_frame(cli_root, st, on_desk(text_page([words] * (3 + 5 * i))), "down", "file:x")
    assert len(st.pages("down")) == 4
    cli_root.llm_failure_limit = 2
    prov = FakeProvider(raise_exc=LLMUnavailable("network down"))
    monkeypatch.setattr(cli, "_provider", lambda *a, **k: prov)
    r = runner.invoke(cli.app, ["verify", "--doc", "down"])
    assert r.exit_code == 0, r.output
    assert len(prov.calls) == 2 and "LLM unavailable; finishing the rest OCR-only" in r.output
    assert all(p["status"] == "unverified" for p in st.pages("down"))
    assert all(render.page_path(cli_root.documents / "down", p["sequence"]).exists() for p in st.pages("down"))


def test_review_interactive_resolves_and_signs_off(cli_root, store):
    doc_dir = cli_root.documents / "rev"
    store.ensure_document("rev"); store.add_page("pr", "rev", "sha", "0" * 64)
    store.set_status("pr", "flagged", "5")
    fm = {"ocrmd": 1, "id": "pr", "document": "rev", "sequence": 1, "page_label": "5", "context": {},
          "references": {"related": []}, "verification": {"status": "flagged", "reviewed_by": None}}
    body = "total ==[44.80]==[^ocrmd-1] paid\n\n[^ocrmd-1]: ocr=\"44.60\" llm=\"44.80\" — unresolved\n"
    render.write_page(doc_dir, 1, fm, body)
    r = runner.invoke(cli.app, ["review", "--doc", "rev", "--who", "amr"], input="o\n")
    assert r.exit_code == 0, r.output
    fm2, body2 = render.read_page(render.page_path(doc_dir, 1))
    assert "total 44.60 paid" in body2 and fm2["verification"]["status"] == "reviewed"
    assert fm2["verification"]["reviewed_by"]["who"] == "amr" and store.page("pr")["status"] == "reviewed"
    assert "1 page(s) signed off" in r.output


@needs_ocr
def test_delete_command(cli_root, store):
    for words in ("one page of text here", "another page of different text"):
        pid = pipeline.ingest_frame(cli_root, store, on_desk(text_page([words] * 10)), "d", "file:x").pages[0]
        pipeline.run_page(cli_root, store, store.page(pid), None, "d")
    r = runner.invoke(cli.app, ["delete", "--doc", "d", "--page", "1", "-y"])
    assert r.exit_code == 0 and "1 page(s)" in r.output and len(store.pages("d")) == 1, r.output
    r = runner.invoke(cli.app, ["delete", "--doc", "d", "--page", "7", "-y"])
    assert r.exit_code == 1 and "no such page" in r.output
    r = runner.invoke(cli.app, ["delete", "--doc", "d", "--all", "-y"])
    assert r.exit_code == 0 and "d" not in store.documents() and not (cli_root.documents / "d").exists()
    assert runner.invoke(cli.app, ["delete", "--doc", "d", "--all", "-y"]).exit_code == 1


@needs_ocr
def test_read_prints_a_region_and_writes_nothing(cli_root, store, tmp_path):
    pid = pipeline.ingest_frame(cli_root, store, on_desk(text_page()), "cli-book", "test").pages[0]
    crop = tmp_path / "region.png"
    r = runner.invoke(cli.app, ["read", "--doc", "cli-book", "--page", "1",
                                "--region", "0.05,0.05,0.9,0.4", "--save", str(crop)])
    assert r.exit_code == 0, r.output
    assert "enlarged" in r.output and "truth" in r.output.lower()
    assert crop.exists()
    assert not (cli_root.documents / "cli-book" / "pages").exists()      # nothing written to the book
    assert store.page(pid)["status"] == "unverified"


def test_read_refuses_a_page_that_is_not_there(cli_root):
    r = runner.invoke(cli.app, ["read", "--doc", "nope", "--page", "1"])
    assert r.exit_code == 1 and "no page 1" in r.output


FREE_PIPELINE = """
name: chunks
operations:
  - {name: chop, type: split, split_key: text, words: 6, overlap: 0}
  - {name: first, type: sample, n: 2}
pipeline:
  operations: [chop, first]
  output: {type: json, path: chunks.json}
"""


@needs_ocr
def test_etl_runs_a_pipeline_and_writes_where_it_says(cli_root, store, tmp_path, monkeypatch):
    import json
    pid = pipeline.ingest_frame(cli_root, store, on_desk(text_page()), "cli-book", "test").pages[0]
    monkeypatch.setattr(cli, "_provider", lambda *a, **k: FakeProvider(markdown="one two three four five six seven eight"))
    pipeline.run_page(cli_root, store, store.page(pid), FakeProvider(markdown="one two three four five six seven eight"), "cli-book")
    spec = tmp_path / "p.yaml"; spec.write_text(FREE_PIPELINE)

    r = runner.invoke(cli.app, ["etl", "--doc", "cli-book", "-p", str(spec)])
    assert r.exit_code == 0, r.output
    out = cli_root.documents / "cli-book" / "extracted" / "chunks.json"
    assert out.exists()
    rows = json.loads(out.read_text())
    assert len(rows) == 2 and rows[0]["_page"] == 1 and rows[0]["_document"] == "cli-book"


@needs_ocr
def test_etl_exports_a_docetl_pipeline_without_running_anything(cli_root, store, tmp_path, monkeypatch):
    import yaml as yaml_mod
    pid = pipeline.ingest_frame(cli_root, store, on_desk(text_page()), "cli-book", "test").pages[0]
    pipeline.run_page(cli_root, store, store.page(pid), FakeProvider(markdown="some text"), "cli-book")
    spec = tmp_path / "p.yaml"; spec.write_text(FREE_PIPELINE)
    r = runner.invoke(cli.app, ["etl", "--doc", "cli-book", "-p", str(spec), "--export-docetl", str(tmp_path / "x")])
    assert r.exit_code == 0 and "docetl run" in r.output
    got = yaml_mod.safe_load((tmp_path / "x" / "pipeline.yaml").read_text())
    assert [o["name"] for o in got["operations"]] == ["chop", "first"]
    assert got["pipeline"]["steps"][0]["operations"] == ["chop", "first"]
    assert not (cli_root.documents / "cli-book" / "extracted").exists()


def test_etl_says_which_line_of_a_bad_pipeline_is_wrong(cli_root, tmp_path):
    spec = tmp_path / "bad.yaml"
    spec.write_text("name: x\noperations: [{name: a, type: map, output: {schema: {y: timestamp}}}]\n")
    r = runner.invoke(cli.app, ["etl", "--doc", "cli-book", "-p", str(spec)])
    assert r.exit_code == 1 and "unknown output type" in r.output
