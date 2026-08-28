"""SQLite store (migrations, scoping, threads) and configuration loading."""
from __future__ import annotations
import sqlite3, threading
from pathlib import Path

import pytest

from ocrmd.config import Settings, load_settings
from ocrmd.store import Store

OLD_SCHEMA = """
CREATE TABLE documents(slug TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE pages(id TEXT PRIMARY KEY, document TEXT NOT NULL, sequence INTEGER NOT NULL,
  raw_sha256 TEXT NOT NULL, phash TEXT NOT NULL, captured_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unverified', page_label TEXT, UNIQUE(document, sequence));
CREATE TABLE captures(sha256 TEXT PRIMARY KEY, phash TEXT NOT NULL, page_id TEXT, duplicate_of TEXT,
  captured_at TEXT NOT NULL, source TEXT NOT NULL);
CREATE TABLE stages(page_id TEXT NOT NULL, stage TEXT NOT NULL, version TEXT NOT NULL,
  completed_at TEXT, error TEXT, PRIMARY KEY(page_id, stage));
CREATE TABLE llm_calls(id INTEGER PRIMARY KEY, page_id TEXT, provider TEXT, model TEXT, request_id TEXT,
  tokens_in INTEGER, tokens_out INTEGER, cache_read INTEGER, latency_ms INTEGER, at TEXT);
INSERT INTO documents VALUES('old', '2026-01-01');
INSERT INTO pages(id,document,sequence,raw_sha256,phash,captured_at) VALUES('p1','old',1,'sha','0000','2026-01-01');
"""


def test_v010_database_is_migrated_in_place(tmp_path: Path):
    db = tmp_path / "ocrmd.db"
    con = sqlite3.connect(db); con.executescript(OLD_SCHEMA); con.commit(); con.close()
    st = Store(db)
    cols = {r["name"] for r in st.db.execute("PRAGMA table_info(pages)")}
    assert {"part", "width_px", "advice", "written_fingerprint", "reviewed_by"} <= cols
    assert st.page("p1")["status"] == "unverified"
    assert st.integrity() == "ok"
    st.set_page_meta("p1", 640, "move closer")
    assert st.pages("old")[0]["advice"] == "move closer"
    assert st.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_near_duplicate_is_scoped_to_the_document(tmp_path: Path):
    st = Store(tmp_path / "t.db")
    st.ensure_document("a"); st.ensure_document("b")
    st.add_page("pa", "a", "sha-a", "1" * 64)
    ids = lambda rows: [r["id"] for r in rows]
    assert ids(st.near_duplicates("1" * 64, "a")) == ["pa"]
    assert ids(st.near_duplicates("1" * 64, "b")) == []      # a similar-looking page in another book is new
    assert ids(st.near_duplicates("1" * 62 + "00", "a", max_dist=4)) == ["pa"]
    assert ids(st.near_duplicates("1" * 58 + "000000", "a", max_dist=4)) == []


def test_store_is_safe_across_threads(tmp_path: Path):
    st = Store(tmp_path / "t.db")
    st.ensure_document("doc")
    errors = []

    def work(i):
        try:
            st.add_page(f"p{i}", "doc", f"sha{i}", "0" * 64)
            st.mark_stage(f"p{i}", "ocr", "v")
            st.set_status(f"p{i}", "verified")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    seqs = sorted(p["sequence"] for p in st.pages("doc"))
    assert seqs == list(range(1, 13))                       # sequence allocation never collided
    assert st.counts("doc") == {"verified": 12}


def test_stage_done_with_and_without_version(tmp_path: Path):
    st = Store(tmp_path / "t.db")
    st.ensure_document("d"); st.add_page("p", "d", "s", "0" * 64)
    st.mark_stage("p", "llm", "transcribe@2")
    assert st.stage_done("p", "llm")                         # any completed version
    assert st.stage_done("p", "llm", "transcribe@2")
    assert not st.stage_done("p", "llm", "transcribe@3")     # stale for --refresh
    st.mark_stage("p", "llm", "transcribe@3", error="boom")
    assert not st.stage_done("p", "llm") and st.stage_error("p", "llm") == "boom"


def test_reviewed_and_batches(tmp_path: Path):
    st = Store(tmp_path / "t.db")
    st.ensure_document("d"); st.add_page("p", "d", "s", "0" * 64)
    st.set_reviewed("p", "amr", "reviewed")
    assert st.page("p")["reviewed_by"] == "amr" and st.page("p")["status"] == "reviewed"
    st.add_batch("msgbatch_1", "anthropic", "m", "transcribe@3", ["p"])
    assert [b["id"] for b in st.open_batches()] == ["msgbatch_1"]
    st.end_batch("msgbatch_1", "ended")
    assert st.open_batches() == []


# --- config --------------------------------------------------------------------------------------
def test_env_overrides_and_toml(tmp_path: Path):
    (tmp_path / "ocrmd.toml").write_text('model = "from-toml"\nstable_frames = 5\nfocus = "settle"\n')
    s = load_settings(root=tmp_path, env={"OCRMD_MODEL": "from-env", "OCRMD_RESOLUTION": "1280x960",
                                          "OCRMD_LLM_WORKERS": "3"})
    assert s.model == "from-env"               # env beats toml
    assert s.stable_frames == 5 and s.focus == "settle"
    assert s.resolution == (1280, 960) and s.llm_workers == 3
    assert s.root == tmp_path and s.db_path == tmp_path / "ocrmd.db"


def test_invalid_config_is_a_clear_error(tmp_path: Path):
    with pytest.raises(RuntimeError, match="focus"):
        load_settings(root=tmp_path, env={"OCRMD_FOCUS": "sometimes"})
    with pytest.raises(RuntimeError, match="stable_frames"):
        load_settings(root=tmp_path, env={"OCRMD_STABLE_FRAMES": "eight"})
    (tmp_path / "ocrmd.toml").write_text("model = \n")
    with pytest.raises(RuntimeError, match="cannot read"):
        load_settings(root=tmp_path, env={})


def test_defaults_are_sane():
    s = Settings()
    assert s.prompt_version == "transcribe@5" and s.llm_failure_limit >= 1 and s.occlusion_cover < 0.2
    assert s.focus_bracket in ("auto", "always", "off") and s.focus_bracket_shots >= 2
