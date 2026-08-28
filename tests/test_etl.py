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
    spec_ = etl.export_docetl(etl.Pipeline.load(spec(tmp_path)), rows(2), out, "claude-haiku-4-5")
    assert spec_["default_model"] == "claude-haiku-4-5"
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
