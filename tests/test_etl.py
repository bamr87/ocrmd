"""The extraction layer: DocETL's operators over ocrmd's pages, with ocrmd's provenance and caching."""
from __future__ import annotations
import json

import pytest

from conftest import FakeProvider, on_desk, text_page
from ocrmd import etl, pipeline
from ocrmd.llm import LLMPageError
from ocrmd.ocr import available_engine

needs_ocr = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")

SPEC = """
name: cases
operations:
  - name: find
    type: map
    prompt: "Cases on page {{ input.page_label }}:\\n{{ input.text }}"
    output:
      schema:
        cases: "list[{name: string, year: integer}]"
  - name: flatten
    type: unnest
    unnest_key: cases
pipeline:
  operations: [find, flatten]
  output: {type: json, path: cases.json}
"""


class JsonProvider:
    """A provider that answers structured questions from a canned list, and counts the asking."""
    name, model = "anthropic", "fake-model"

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def complete_json(self, system, user, schema, max_tokens=4000):
        from ocrmd.llm import JsonResult
        self.prompts.append(user)
        data = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        if isinstance(data, Exception):
            raise data
        return JsonResult(data=data, provider=self.name, model=self.model, request_id="msg_x",
                          usage={"input_tokens": 7, "output_tokens": 3}, latency_ms=1)


def spec(tmp_path, text=SPEC):
    p = tmp_path / "p.yaml"
    p.write_text(text)
    return p


def rows(n=2):
    return [{"_document": "d", "_page": i + 1, "_page_id": f"p{i}", "page_label": str(i + 1),
             "chapter": "One" if i == 0 else "Two", "text": f"page {i} text"} for i in range(n)]


def runner(root, store, prov=None, **kw):
    return etl.Runner(root, store, "d", prov, **kw)


# --- the spec ---------------------------------------------------------------------------------------
def test_the_type_notation_becomes_a_strict_schema():
    s = etl.object_schema({"name": "string", "year": "integer", "tags": "list[string]",
                           "kind": "enum[a, b]", "cites": "list[{case: string, page: integer}]"})
    assert s["required"] == ["name", "year", "tags", "kind", "cites"]
    assert s["additionalProperties"] is False
    assert s["properties"]["kind"]["enum"] == ["a", "b"]
    assert s["properties"]["cites"]["items"]["properties"]["page"] == {"type": "integer"}
    assert s["properties"]["cites"]["items"]["additionalProperties"] is False   # nested objects too


def test_an_unknown_type_names_itself():
    with pytest.raises(ValueError, match="unknown output type"):
        etl.object_schema({"x": "timestamp"})


def test_a_pipeline_naming_an_operation_that_does_not_exist_is_refused(tmp_path):
    p = spec(tmp_path, SPEC.replace("[find, flatten]", "[find, nope]"))
    with pytest.raises(ValueError, match="not defined"):
        etl.Pipeline.load(p)


def test_a_missing_template_field_is_an_error_not_an_empty_string():
    """A silently empty {{ input.x }} changes the question without changing the prompt, and the model
    answers the changed one."""
    with pytest.raises(KeyError, match="not a field"):
        etl.template("{{ input.nope }}", {"text": "hi"})


# --- operators --------------------------------------------------------------------------------------
def test_map_keeps_the_page_a_row_came_from(root, store, tmp_path):
    prov = JsonProvider([{"cases": [{"name": "Roe", "year": 1973}]}])
    out = runner(root, store, prov).run(etl.Pipeline.load(spec(tmp_path)), rows(1))
    assert out == [{"_document": "d", "_page": 1, "_page_id": "p0", "page_label": "1", "chapter": "One",
                    "text": "page 0 text", "name": "Roe", "year": 1973}]
    assert "page 0 text" in prov.prompts[0]


def test_unnest_turns_one_page_into_one_row_per_thing_found(root, store, tmp_path):
    prov = JsonProvider([{"cases": [{"name": "A", "year": 1}, {"name": "B", "year": 2}]}])
    out = runner(root, store, prov).run(etl.Pipeline.load(spec(tmp_path)), rows(1))
    assert [r["name"] for r in out] == ["A", "B"]
    assert all(r["_page"] == 1 for r in out)


def test_filter_drops_rows_the_model_rejects_but_keeps_ones_it_could_not_answer(root, store):
    op = etl.Operation(name="f", type="filter", prompt="{{ input.text }}", output={"keep": "boolean"})
    prov = JsonProvider([{"keep": True}, {"keep": False}, LLMPageError("refusal")])
    r = runner(root, store, prov)
    out = r._filter(op, rows(3))
    assert [x["_page"] for x in out] == [1, 3]           # kept, dropped, unanswered-so-kept
    assert r.stats.errors and "refusal" in r.stats.errors[0]


def test_reduce_groups_and_records_which_pages_it_read(root, store):
    op = etl.Operation(name="r", type="reduce", reduce_key="chapter", prompt="{{ inputs }}",
                       output={"summary": "string"})
    prov = JsonProvider([{"summary": "s1"}, {"summary": "s2"}])
    out = runner(root, store, prov)._reduce(op, rows(2))
    assert [x["chapter"] for x in out] == ["One", "Two"]
    assert out[0]["_pages"] == [1] and out[0]["summary"] == "s1"
    assert "--- page 1" in prov.prompts[0]                # the group is labelled by page in the prompt


def test_reduce_with_no_key_folds_the_whole_document(root, store):
    op = etl.Operation(name="r", type="reduce", prompt="{{ inputs }}", output={"summary": "string"})
    out = runner(root, store, JsonProvider([{"summary": "all"}]))._reduce(op, rows(3))
    assert len(out) == 1 and out[0]["_pages"] == [1, 2, 3] and out[0]["_rows"] == 3


def test_resolve_only_asks_about_pairs_that_could_be_the_same(root, store):
    """The blocking is the point: all-pairs is quadratic, and most pairs are obviously unrelated."""
    op = etl.Operation(name="s", type="resolve", resolve_key="name", comparison_prompt="same? {{ input.a }} {{ input.b }}",
                       output={"same": "boolean"})
    data = [{"name": "Roe v. Wade"}, {"name": "Roe v Wade"}, {"name": "Roe v. Wade"},
            {"name": "Marbury v. Madison"}]
    prov = JsonProvider([{"same": True}])
    out = runner(root, store, prov)._resolve(op, data)
    assert len({r["name"] for r in out}) == 2                    # two real cases
    assert out[1]["name"] == "Roe v. Wade" and out[1]["name_as_written"] == "Roe v Wade"
    assert len(prov.prompts) == 1                                # Marbury was never compared to Roe
    assert "Roe v Wade" in prov.prompts[0]


def test_resolve_without_a_prompt_merges_only_near_identical_strings(root, store):
    op = etl.Operation(name="s", type="resolve", resolve_key="name", output={"same": "boolean"})
    out = runner(root, store, None)._resolve(op, [{"name": "Roe v. Wade"}, {"name": "Roe v. Wade "},
                                                  {"name": "Roe v. Jones"}])
    assert len({r["name"] for r in out}) == 2


def test_split_and_sample_cost_nothing(root, store):
    long = [{"_page": 1, "text": " ".join(str(i) for i in range(1000))}]
    r = runner(root, store, None)
    chunks = r._split(etl.Operation(name="c", type="split", words=400, overlap=40), long)
    assert len(chunks) == 3 and chunks[0]["_chunks"] == 3 and chunks[1]["text"].startswith("360 ")
    assert len(r._sample(etl.Operation(name="s", type="sample", n=2), rows(5))) == 2
    assert r.stats.calls == 0


# --- the cache --------------------------------------------------------------------------------------
def test_a_second_run_costs_nothing_and_a_changed_prompt_costs_again(root, store, tmp_path):
    p = spec(tmp_path)
    prov = JsonProvider([{"cases": [{"name": "Roe", "year": 1973}]}])
    first = runner(root, store, prov)
    a = first.run(etl.Pipeline.load(p), rows(2))
    assert first.stats.calls == 2 and first.stats.cached == 0
    assert (root.root / "etl" / "d" / "find.json").exists()

    second = runner(root, store, prov)
    assert second.run(etl.Pipeline.load(p), rows(2)) == a
    assert second.stats.calls == 0 and second.stats.cached == 2

    (tmp_path / "x").mkdir(exist_ok=True)
    changed = spec(tmp_path / "x", SPEC.replace("Cases on page", "Every case on page"))
    third = runner(root, store, prov)
    third.run(etl.Pipeline.load(changed), rows(2))
    assert third.stats.calls == 2 and third.stats.cached == 0


def test_the_run_is_recorded_in_the_stores_token_accounting(root, store, tmp_path):
    runner(root, store, JsonProvider([{"cases": []}])).run(etl.Pipeline.load(spec(tmp_path)), rows(2))
    assert [dict(r)["n"] for r in store.costs()] == [2]


def test_a_dry_run_spends_nothing_and_shows_the_prompt(root, store, tmp_path):
    prov = JsonProvider([{"cases": []}])
    r = runner(root, store, prov, dry_run=True)
    r.run(etl.Pipeline.load(spec(tmp_path)), rows(2))
    assert prov.prompts == [] and r.stats.calls == 0
    assert len(r.prompts) == 2 and "page 0 text" in r.prompts[0][1]
    assert not (root.root / "etl").exists()


def test_without_a_provider_the_rows_come_back_unchanged_and_the_count_says_why(root, store, tmp_path):
    """`--no-llm` is a legitimate way to run a pipeline. It must produce an honest empty answer with a
    count of the calls it did not make, not a plausible one."""
    r = runner(root, store, None)
    kept = r._map(etl.Pipeline.load(spec(tmp_path)).op("find"), rows(2))
    assert kept == rows(2) and r.stats.skipped == 2
    assert r.run(etl.Pipeline.load(spec(tmp_path)), rows(2)) == []      # unnest has no list to flatten
    assert "4 not asked" in r.stats.describe()


# --- source rows and output -------------------------------------------------------------------------
@needs_ocr
def test_pages_become_rows_with_their_labels_and_text(root, store):
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(["the quick brown fox"] * 6)), "d", "test").pages[0]
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="the quick brown fox", page_number="7"), "d")
    rows_ = etl.page_rows(root, store, "d")
    assert len(rows_) == 1
    assert rows_[0]["_page_id"] == pid and rows_[0]["page_label"] == "7" and "quick brown fox" in rows_[0]["text"]


@needs_ocr
def test_a_blank_page_is_not_offered_to_an_operator(root, store):
    """An operator asked to find something in an empty page is being invited to invent it."""
    pid = pipeline.ingest_frame(root, store, on_desk(text_page(["the quick brown fox"] * 6)), "d", "test").pages[0]
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="x"), "d")
    store.set_status(pid, "blank", None)
    assert etl.page_rows(root, store, "d") == []


def test_the_spec_can_narrow_the_columns_but_never_the_provenance(root, tmp_path):
    """A map over pages carries the page text into every row — right in the JSON, unreadable in a
    spreadsheet."""
    data = [{"_document": "d", "_page": 3, "text": "a whole page", "topic": "t"}]
    etl.write(data, tmp_path / "n.csv", "csv", ["topic"])
    assert (tmp_path / "n.csv").read_text().splitlines()[0] == "_document,_page,topic"
    assert "a whole page" not in (tmp_path / "n.csv").read_text()
    assert json.loads(etl.write(data, tmp_path / "n.json", "json", ["topic"]).read_text())[0] == {
        "_document": "d", "_page": 3, "topic": "t"}


def test_output_writers_lead_with_the_provenance(root, tmp_path):
    data = [{"name": "Roe", "_page": 3, "_document": "d", "tags": ["a"]}]
    etl.write(data, tmp_path / "o.csv", "csv")
    head = (tmp_path / "o.csv").read_text().splitlines()[0]
    assert head.startswith("_document,_page")
    assert json.loads(etl.write(data, tmp_path / "o.json", "json").read_text())[0]["name"] == "Roe"
    assert "| Roe |" in etl.write(data, tmp_path / "o.md", "md").read_text()


# --- handing it to DocETL ----------------------------------------------------------------------------
def test_the_same_pipeline_exports_as_a_docetl_one(tmp_path):
    out = tmp_path / "x"
    spec_ = etl.export_docetl(etl.Pipeline.load(spec(tmp_path)), rows(2), out, "claude-haiku-5-5")
    assert spec_["default_model"] == "anthropic/claude-haiku-5-5"
    assert [o["name"] for o in spec_["operations"]] == ["find", "flatten"]
    assert spec_["operations"][0]["output"]["schema"]["cases"] == "list[{name: string, year: integer}]"
    assert spec_["operations"][1]["unnest_key"] == "cases"
    assert spec_["pipeline"]["steps"][0]["input"] == "pages"
    assert (out / "pipeline.yaml").exists()
    assert json.loads((out / "pages.json").read_text())[0]["_page"] == 1


def test_the_shipped_example_pipeline_parses():
    from pathlib import Path
    p = etl.Pipeline.load(Path(__file__).resolve().parent.parent / "examples" / "cases.pipeline.yaml")
    assert p.steps == ["find_cases", "one_row_per_case", "same_case"]
    assert p.op("find_cases").schema["properties"]["cases"]["type"] == "array"
    assert p.output["type"] == "csv"


# --- the page model as a source and a sink ----------------------------------------------------------
from ocrmd import render                          # noqa: E402

needs_ocr_engine = pytest.mark.skipif(available_engine() is None, reason="no local OCR engine")


def _seed(root, store, slug="d"):
    import cv2
    img = text_page(["Chapter One", "the quick brown fox jumps over the lazy dog",
                     "and then it runs away across the field", "while nobody is watching at all"] * 2)
    cv2.circle(img, (700, 1300), 120, (20, 20, 20), 8)
    pid = pipeline.ingest_frame(root, store, on_desk(img), slug, "file:t").pages[0]
    md = "# Chapter One\n\n" + "\n".join(["the quick brown fox jumps over the lazy dog",
                                         "and then it runs away across the field", "while nobody is watching at all"] * 2)
    pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown=md, page_number="9", chapter="One"), slug)
    render.ensure_document_meta(root.documents / slug, slug, title="Book")
    return pid


def test_a_spec_names_its_source_and_its_sink_and_is_refused_when_they_are_wrong(tmp_path):
    p = spec(tmp_path, SPEC.replace("pipeline:\n", "source: blocks\npipeline:\n"))
    assert etl.Pipeline.load(p).source == "blocks"
    with pytest.raises(ValueError, match="unknown source"):
        etl.Pipeline.load(spec(tmp_path, SPEC.replace("pipeline:\n", "source: moon\npipeline:\n")))
    bad = SPEC.replace("output: {type: json, path: cases.json}", "output: {type: frontmatter, fields: {x: ocr.engine}}")
    with pytest.raises(ValueError, match="may not write"):
        etl.Pipeline.load(spec(tmp_path, bad))
    ok = SPEC.replace("output: {type: json, path: cases.json}", "output: {type: frontmatter, fields: {cases: references.cites}}")
    assert etl.Pipeline.load(spec(tmp_path, ok)).writes_back


def test_a_filter_decides_on_the_named_boolean(tmp_path, root, store):
    s = """name: f
operations:
  - name: keep
    type: filter
    prompt: "{{ input.text }}"
    output: {schema: {relevant: boolean, seen: boolean}}
pipeline: {operations: [keep]}
"""
    pipe = etl.Pipeline.load(spec(tmp_path, s))
    assert pipe.op("keep").boolean_field == "relevant"
    prov = JsonProvider([{"seen": True, "relevant": False}, {"seen": False, "relevant": True}])
    out = runner(root, store, prov).run(pipe, rows())
    assert [r["_page"] for r in out] == [2]


def test_an_extract_is_held_to_the_page(tmp_path, root, store):
    s = """name: e
operations:
  - name: quotes
    type: extract
    prompt: "Quote the fox sentence. {{ input.text }}"
pipeline: {operations: [quotes]}
"""
    pipe = etl.Pipeline.load(spec(tmp_path, s))
    assert pipe.op("quotes").output == {"spans": "list[string]"}
    row = {"_document": "d", "_page": 1, "_page_id": "p", "_block": "b1", "_bbox": [0, 0, 1, 1],
           "text": "the quick brown fox\njumps over",
           "_lines": [["the quick brown fox", [0.1, 0.1, 0.5, 0.03]], ["jumps over", [0.1, 0.15, 0.3, 0.03]]]}
    prov = JsonProvider([{"spans": ["quick brown fox", "a sentence that is not there"]}])
    out = runner(root, store, prov).run(pipe, [row])
    spans = out[0]["_spans"]
    assert spans[0]["found"] and spans[0]["bbox"] == [0.1, 0.1, 0.5, 0.03]
    assert not spans[1]["found"] and spans[1]["bbox"] is None      # kept, and marked: the fabrication signal
    assert etl.locality(out) == (1, 2)
    sc = etl.score(out, [{"_page": 1, "name": "x"}], "name")
    assert sc.locality == 0.5 and sc.recall == 0.0 and "locality 0.50" in sc.describe()


def test_an_operator_may_name_its_own_model(tmp_path, root, store):
    s = SPEC.replace("    type: map\n", "    type: map\n    model: other-model\n")
    pipe = etl.Pipeline.load(spec(tmp_path, s))
    default, other = JsonProvider([{"cases": []}]), JsonProvider([{"cases": []}])
    other.model = "other-model"
    made = []
    r = etl.Runner(root, store, "d", default, provider_for=lambda m: made.append(m) or other)
    r.run(pipe, rows())
    assert made == ["other-model"] and len(other.prompts) == 2 and default.prompts == []
    assert pipe.op("find").version("fake-model") != etl.Operation(**(pipe.op("find").__dict__ | {"model": None})).version("fake-model")


def test_the_docetl_export_speaks_litellm_and_completes_resolve(tmp_path):
    s = SPEC.replace("pipeline:", """  - name: same
    type: resolve
    resolve_key: name
    comparison_prompt: "same? A: {{ input.a }} B: {{ input.b }}"
    output: {schema: {same: boolean}}
  - name: q
    type: extract
    prompt: "quote"
pipeline:""")
    s = s.replace("operations: [find, flatten]", "operations: [find, flatten, same, q]")
    pipe = etl.Pipeline.load(spec(tmp_path, s))
    out = etl.export_docetl(pipe, rows(), tmp_path / "out", "claude-haiku-5-5")
    assert out["default_model"] == "anthropic/claude-haiku-5-5"
    resolve = next(o for o in out["operations"] if o["type"] == "resolve")
    assert "resolution_prompt" in resolve and resolve["output"]["schema"] == {"name": "string"}
    ext = next(o for o in out["operations"] if o["type"] == "extract")
    assert ext["document_keys"] == ["text"] and ext["extraction_key"] == "spans"


@needs_ocr_engine
def test_blocks_and_figures_are_rows_with_a_rectangle(root, store):
    pid = _seed(root, store)
    blocks = etl.rows_for(root, store, "d", "blocks")
    assert blocks and all(r["_page_id"] == pid and r["_block"] and len(r["_bbox"]) == 4 for r in blocks)
    assert {r["role"] for r in blocks} >= {"heading", "paragraph"}
    assert all("_lines" in r for r in blocks) and blocks[0]["agreement"] is not None
    figs = etl.rows_for(root, store, "d", "figures")
    assert len(figs) == 1 and figs[0]["role"] == "figure" and figs[0]["_asset"].endswith(".png")
    assert "quick brown fox" in figs[0]["text"]                     # the page text rides with the picture
    with pytest.raises(FileNotFoundError):
        etl.rows_for(root, store, "d", "extracted:nothing")


@needs_ocr_engine
def test_a_layout_sink_names_the_figure_and_the_caption_survives_a_rerun(root, store, tmp_path):
    from ocrmd.web.jobs import BUNDLED_PIPELINES
    _seed(root, store)
    pipe = etl.Pipeline.load(BUNDLED_PIPELINES / "figures.yaml")
    assert pipe.source == "figures" and pipe.writes_back and pipe.op("name_figure").image

    class SeeingProvider(JsonProvider):
        def complete_json(self, system, user, schema, max_tokens=4000, image_bgr=None):
            self.images = getattr(self, "images", []) + [image_bgr is not None]
            return super().complete_json(system, user, schema, max_tokens)
    prov = SeeingProvider([{"kind": "diagram", "caption": "Figure 1: a circle"}])
    rows_ = etl.rows_for(root, store, "d", "figures")
    out = etl.Runner(root, store, "d", prov).run(pipe, rows_)
    assert prov.images == [True]                                    # the crop went with the question
    counts = etl.write_back(root, store, "d", pipe, out, "name_figure")
    assert counts["blocks"] == 1
    lay = pipeline.page_layout(root, store, store.page(out[0]["_page_id"]))
    f = lay.figures[0]
    assert f.kind == "diagram" and f.caption == "Figure 1: a circle" and f.source == "etl"
    assert store.etl_writes("d")[0]["target"] == "layout"
    # a rerun of the page keeps what the pipeline wrote, like a human's edit
    pipeline.run_page(root, store, store.page(out[0]["_page_id"]), None, "d")
    lay2 = pipeline.load_layout(root, out[0]["_page_id"])
    assert lay2.figures[0].caption == "Figure 1: a circle"
    fm, _ = render.read_page(render.page_path(root.documents / "d", 1))
    assert fm["references"]["figures"][0]["caption"] == "Figure 1: a circle"


@needs_ocr_engine
def test_a_frontmatter_sink_fills_cites_and_verify_keeps_them(root, store):
    from ocrmd.web.jobs import BUNDLED_PIPELINES
    pid = _seed(root, store)
    pipe = etl.Pipeline.load(BUNDLED_PIPELINES / "references.yaml")
    prov = JsonProvider([{"cites": ["Marbury v. Madison"], "tags": ["foxes"]}])
    out = etl.Runner(root, store, "d", prov).run(pipe, etl.rows_for(root, store, "d", "pages"))
    counts = etl.write_back(root, store, "d", pipe, out)
    assert counts["pages"] == 1 and counts["fields"] == 2
    path = render.page_path(root.documents / "d", 1)
    fm, _ = render.read_page(path)
    assert fm["references"]["cites"] == ["Marbury v. Madison"] and fm["references"]["tags"] == ["foxes"]
    r = pipeline.run_page(root, store, store.page(pid), FakeProvider(markdown="# Chapter One\n\nthe quick brown fox", page_number="9"), "d", refresh=True)
    assert "references.cites" in r.preserved
    fm2, _ = render.read_page(r.path)
    assert fm2["references"]["cites"] == ["Marbury v. Madison"] and fm2["references"]["tags"] == ["foxes"]
    # writing again accumulates rather than duplicating
    etl.write_back(root, store, "d", pipe, out)
    assert render.read_page(path)[0]["references"]["cites"] == ["Marbury v. Madison"]


def test_a_document_sink_adds_to_document_yaml_and_never_touches_the_slug(root, store, tmp_path):
    s = """name: about
operations:
  - name: summary
    type: reduce
    prompt: "{{ inputs }}"
    output: {schema: {abstract: string, tags: "list[string]"}}
pipeline:
  operations: [summary]
  output: {type: document, fields: {abstract: abstract, tags: tags}}
"""
    pipe = etl.Pipeline.load(spec(tmp_path, s))
    render.ensure_document_meta(root.documents / "d", "d", title="T")
    out = runner(root, store, JsonProvider([{"abstract": "A book about foxes.", "tags": ["foxes"]}])).run(pipe, rows())
    counts = etl.write_back(root, store, "d", pipe, out)
    meta = render.load_document_meta(root.documents / "d")
    assert counts["document"] == 1 and meta["abstract"] == "A book about foxes." and meta["tags"] == ["foxes"]
    assert meta["slug"] == "d" and meta["title"] == "T"
    with pytest.raises(ValueError, match="may not write"):
        etl.Pipeline.load(spec(tmp_path, s.replace("{abstract: abstract, tags: tags}", "{abstract: slug}")))


def test_the_bundled_library_parses():
    from ocrmd.web.jobs import BUNDLED_PIPELINES
    names = {p.stem for p in BUNDLED_PIPELINES.glob("*.yaml")}
    assert names >= {"figures", "references", "glossary", "receipts", "cases"}
    for p in BUNDLED_PIPELINES.glob("*.yaml"):
        pipe = etl.Pipeline.load(p)
        assert pipe.steps
