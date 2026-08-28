"""SQLite state: documents, pages, captures, stages, llm_calls, batches.

One connection, shared across threads (`scan` runs OCR on one worker and LLM calls on a small pool).
Every method takes the lock, so callers never see a half-applied read-then-write sequence, and the
database runs in WAL mode so a crash mid-write leaves the last committed state intact.
"""
from __future__ import annotations
import sqlite3, threading, datetime as dt
from functools import wraps
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  slug TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pages(
  id TEXT PRIMARY KEY, document TEXT NOT NULL, sequence INTEGER NOT NULL,
  raw_sha256 TEXT NOT NULL, phash TEXT NOT NULL, captured_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unverified', page_label TEXT, part TEXT,
  UNIQUE(document, sequence));
CREATE TABLE IF NOT EXISTS captures(
  sha256 TEXT PRIMARY KEY, phash TEXT NOT NULL, page_id TEXT, duplicate_of TEXT,
  captured_at TEXT NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stages(
  page_id TEXT NOT NULL, stage TEXT NOT NULL, version TEXT NOT NULL,
  completed_at TEXT, error TEXT, PRIMARY KEY(page_id, stage));
CREATE TABLE IF NOT EXISTS llm_calls(
  id INTEGER PRIMARY KEY, page_id TEXT, provider TEXT, model TEXT, request_id TEXT,
  tokens_in INTEGER, tokens_out INTEGER, cache_read INTEGER, latency_ms INTEGER, at TEXT);
CREATE TABLE IF NOT EXISTS batches(
  id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  page_ids TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, ended_at TEXT);
CREATE TABLE IF NOT EXISTS exports(
  page_id TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL, fidelity REAL, backend TEXT,
  at TEXT NOT NULL, PRIMARY KEY(page_id, kind));
CREATE TABLE IF NOT EXISTS etl_writes(
  id INTEGER PRIMARY KEY, document TEXT NOT NULL, page_id TEXT, pipeline TEXT NOT NULL, operation TEXT,
  target TEXT NOT NULL, keys TEXT NOT NULL, at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS pages_doc_seq ON pages(document, sequence);
CREATE INDEX IF NOT EXISTS pages_status ON pages(document, status);
CREATE INDEX IF NOT EXISTS captures_page ON captures(page_id);
CREATE INDEX IF NOT EXISTS llm_calls_page ON llm_calls(page_id);
"""

# Additive migrations for databases created by an older ocrmd. `CREATE TABLE IF NOT EXISTS`
# silently skips new columns, so every column added after v0.1.0 must be listed here.
COLUMNS: dict[str, dict[str, str]] = {
    "pages": {
        "part": "TEXT",
        "width_px": "INTEGER",            # px across the page crop: the quality predictor
        "advice": "TEXT",                 # framing advice from preprocess, for `status`
        "written_fingerprint": "TEXT",    # hash of the human-editable parts as last written by the pipeline
        "reviewed_by": "TEXT",
        "reviewed_at": "TEXT",
        "fidelity": "REAL",               # how closely the last facsimile export matched the crop (0-1)
    },
    "documents": {"title": "TEXT"},
    # A rejected capture has no page, so without its own `document` column there is nothing tying it
    # to the scan it came from — `source` records the camera, not the book.
    "captures": {"rejected": "TEXT", "document": "TEXT"},
}

STATUSES = ("verified", "flagged", "suspect", "illegible", "blank", "unverified", "reviewed", "error")
DONE_STATUSES = ("verified", "reviewed")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _locked(fn):
    @wraps(fn)
    def inner(self, *a, **kw):
        with self.lock:
            return fn(self, *a, **kw)
    return inner


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass  # read-only or network filesystem: fall back to the default journal
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        for table, cols in COLUMNS.items():
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols.items():
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()

    @_locked
    def integrity(self) -> str:
        return str(self.db.execute("PRAGMA integrity_check").fetchone()[0])

    # documents -----------------------------------------------------------
    @_locked
    def ensure_document(self, slug: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO documents(slug, created_at) VALUES(?,?)", (slug, now()))
        self.db.commit()

    @_locked
    def documents(self) -> list[str]:
        return [r["slug"] for r in self.db.execute("SELECT slug FROM documents ORDER BY created_at")]

    @_locked
    def next_sequence(self, slug: str) -> int:
        r = self.db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM pages WHERE document=?", (slug,)).fetchone()
        return int(r[0])

    # captures / dedup ----------------------------------------------------
    @_locked
    def find_capture(self, sha256: str):
        return self.db.execute("SELECT * FROM captures WHERE sha256=?", (sha256,)).fetchone()

    @_locked
    def near_duplicates(self, phash: str, slug: str | None = None, max_dist: int = 4) -> list[sqlite3.Row]:
        """Existing pages within `max_dist` dHash bits, nearest first, *within one document*: two books can
        share a layout, and a global check silently dropped the first page of every new document that
        resembled an old one. The caller confirms candidates against the raw frames."""
        from .fingerprint import hamming
        q, args = "SELECT id, raw_sha256, phash FROM pages", ()
        if slug:
            q, args = q + " WHERE document=?", (slug,)
        rows = [(hamming(r["phash"], phash), r) for r in self.db.execute(q, args)]
        return [r for d, r in sorted(rows, key=lambda x: x[0]) if d <= max_dist]

    @_locked
    def add_capture(self, sha256, phash, page_id, duplicate_of, source, rejected: str | None = None,
                    document: str | None = None) -> None:
        # Named columns, not positional: `captures` gains columns by migration, and `VALUES(?,?,?...)`
        # silently binds to the wrong ones the first time it does.
        self.db.execute("INSERT OR IGNORE INTO captures(sha256, phash, page_id, duplicate_of, captured_at, "
                        "source, rejected, document) VALUES(?,?,?,?,?,?,?,?)",
                        (sha256, phash, page_id, duplicate_of, now(), source, rejected, document))
        self.db.commit()

    @_locked
    def clear_rejection(self, sha256: str) -> None:
        """Forget that a frame was refused, so a deliberate re-ingest of it can make a page. Without
        this the refusal row itself reads as an exact duplicate and the override silently does nothing."""
        self.db.execute("DELETE FROM captures WHERE sha256=? AND rejected IS NOT NULL AND page_id IS NULL",
                        (sha256,))
        self.db.commit()

    @_locked
    def rejected_captures(self, slug: str | None = None) -> list[sqlite3.Row]:
        """Frames refused as unreadable: kept on disk, never made into pages.

        `page_id IS NULL` matters — half of a spread can be refused while the other half becomes a
        page, and that capture is not one the operator has to re-shoot.
        """
        q = ("SELECT sha256, captured_at, source, rejected, document FROM captures "
             "WHERE rejected IS NOT NULL AND page_id IS NULL")
        args: tuple = ()
        if slug:
            q, args = q + " AND document=?", (slug,)
        return self.db.execute(q + " ORDER BY captured_at", args).fetchall()

    @_locked
    def capture_source(self, sha256: str) -> str | None:
        r = self.db.execute("SELECT source FROM captures WHERE sha256=?", (sha256,)).fetchone()
        return r["source"] if r else None

    @_locked
    def duplicates(self, slug: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT c.sha256, c.captured_at, c.source, c.duplicate_of, p.sequence FROM captures c "
            "JOIN pages p ON p.id=c.duplicate_of WHERE p.document=? ORDER BY c.captured_at", (slug,)).fetchall()

    # pages ---------------------------------------------------------------
    @_locked
    def add_page(self, page_id, slug, sha256, phash, part: str | None = None,
                 width_px: int | None = None, advice: str | None = None) -> int:
        seq = self.next_sequence(slug)
        self.db.execute("INSERT INTO pages(id,document,sequence,raw_sha256,phash,captured_at,part,width_px,advice)"
                        " VALUES(?,?,?,?,?,?,?,?,?)",
                        (page_id, slug, seq, sha256, phash, now(), part, width_px, advice))
        self.db.commit()
        return seq

    @_locked
    def page(self, page_id: str):
        return self.db.execute("SELECT * FROM pages WHERE id=?", (page_id,)).fetchone()

    @_locked
    def page_by_sequence(self, slug: str, sequence: int):
        return self.db.execute("SELECT * FROM pages WHERE document=? AND sequence=?", (slug, sequence)).fetchone()

    @_locked
    def pages(self, slug: str | None = None, status: str | None = None):
        q, args, conds = "SELECT * FROM pages", [], []
        if slug: conds.append("document=?"); args.append(slug)
        if status: conds.append("status=?"); args.append(status)
        if conds: q += " WHERE " + " AND ".join(conds)
        return self.db.execute(q + " ORDER BY document, sequence", args).fetchall()

    @_locked
    def previous_page(self, slug: str, sequence: int):
        return self.db.execute("SELECT * FROM pages WHERE document=? AND sequence<? ORDER BY sequence DESC LIMIT 1",
                               (slug, sequence)).fetchone()

    @_locked
    def raw_captures(self, slug: str):
        """Distinct raw captures for a document, in the order their pages first appear."""
        return self.db.execute("SELECT raw_sha256, phash, MIN(sequence) seq FROM pages WHERE document=? "
                               "GROUP BY raw_sha256 ORDER BY seq", (slug,)).fetchall()

    @_locked
    def delete_page(self, page_id: str) -> None:
        self.db.execute("DELETE FROM stages WHERE page_id=?", (page_id,))
        self.db.execute("UPDATE captures SET page_id=NULL WHERE page_id=?", (page_id,))
        self.db.execute("DELETE FROM pages WHERE id=?", (page_id,))
        self.db.commit()

    @_locked
    def pages_by_raw(self, sha256: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM pages WHERE raw_sha256=? ORDER BY sequence", (sha256,)).fetchall()

    @_locked
    def delete_captures_of(self, page_id: str) -> int:
        """Capture rows that produced or duplicated this page. Without this, re-scanning a deleted page
        would be refused as an exact or near duplicate."""
        n = self.db.execute("DELETE FROM captures WHERE page_id=? OR duplicate_of=?", (page_id, page_id)).rowcount
        self.db.commit()
        return n

    @_locked
    def delete_document(self, slug: str) -> int:
        ids = [r["id"] for r in self.db.execute("SELECT id FROM pages WHERE document=?", (slug,))]
        for pid in ids:
            self.db.execute("DELETE FROM stages WHERE page_id=?", (pid,))
            self.db.execute("DELETE FROM captures WHERE page_id=? OR duplicate_of=?", (pid, pid))
        self.db.execute("DELETE FROM pages WHERE document=?", (slug,))
        self.db.execute("DELETE FROM documents WHERE slug=?", (slug,))
        self.db.commit()
        return len(ids)

    @_locked
    def set_capture_page(self, sha256: str, page_id: str) -> None:
        self.db.execute("UPDATE captures SET page_id=? WHERE sha256=?", (page_id, sha256))
        self.db.commit()

    @_locked
    def set_status(self, page_id: str, status: str, page_label: str | None = None) -> None:
        self.db.execute("UPDATE pages SET status=?, page_label=COALESCE(?,page_label) WHERE id=?",
                        (status, page_label, page_id))
        self.db.commit()

    @_locked
    def set_page_meta(self, page_id: str, width_px: int | None, advice: str | None) -> None:
        self.db.execute("UPDATE pages SET width_px=?, advice=? WHERE id=?", (width_px, advice, page_id))
        self.db.commit()

    @_locked
    def set_fingerprint(self, page_id: str, fingerprint: str) -> None:
        self.db.execute("UPDATE pages SET written_fingerprint=? WHERE id=?", (fingerprint, page_id))
        self.db.commit()

    @_locked
    def set_reviewed(self, page_id: str, who: str | None, status: str | None = None) -> None:
        self.db.execute("UPDATE pages SET reviewed_by=?, reviewed_at=?, status=COALESCE(?, status) WHERE id=?",
                        (who, now() if who else None, status, page_id))
        self.db.commit()

    @_locked
    def counts(self, slug: str) -> dict[str, int]:
        return {r["status"]: r["n"] for r in
                self.db.execute("SELECT status, COUNT(*) n FROM pages WHERE document=? GROUP BY status", (slug,))}

    # stages --------------------------------------------------------------
    @_locked
    def stage(self, page_id: str, stage: str):
        return self.db.execute("SELECT * FROM stages WHERE page_id=? AND stage=?", (page_id, stage)).fetchone()

    def stage_done(self, page_id: str, stage: str, version: str | None = None) -> bool:
        """Completed, and (when `version` is given) at that version."""
        r = self.stage(page_id, stage)
        return bool(r and r["completed_at"] and (version is None or r["version"] == version))

    @_locked
    def mark_stage(self, page_id, stage, version, error: str | None = None) -> None:
        self.db.execute("INSERT OR REPLACE INTO stages VALUES(?,?,?,?,?)",
                        (page_id, stage, version, None if error else now(), error))
        self.db.commit()

    @_locked
    def stage_history(self, page_id: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT stage, version, completed_at FROM stages WHERE page_id=? AND completed_at IS NOT NULL "
            "ORDER BY completed_at", (page_id,))]

    @_locked
    def stage_error(self, page_id: str, stage: str = "llm") -> str | None:
        r = self.db.execute("SELECT error FROM stages WHERE page_id=? AND stage=? AND completed_at IS NULL",
                            (page_id, stage)).fetchone()
        return r["error"] if r else None

    # exports ---------------------------------------------------------------
    @_locked
    def add_export(self, page_id: str, kind: str, path: str, fidelity: float | None = None,
                   backend: str | None = None) -> None:
        self.db.execute("INSERT OR REPLACE INTO exports VALUES(?,?,?,?,?,?)",
                        (page_id, kind, path, fidelity, backend, now()))
        if fidelity is not None:
            self.db.execute("UPDATE pages SET fidelity=? WHERE id=?", (fidelity, page_id))
        self.db.commit()

    @_locked
    def exports(self, page_id: str | None = None, slug: str | None = None) -> list[sqlite3.Row]:
        if page_id:
            return self.db.execute("SELECT * FROM exports WHERE page_id=? ORDER BY kind", (page_id,)).fetchall()
        if slug:
            return self.db.execute("SELECT e.*, p.sequence FROM exports e JOIN pages p ON p.id=e.page_id "
                                   "WHERE p.document=? ORDER BY p.sequence, e.kind", (slug,)).fetchall()
        return self.db.execute("SELECT * FROM exports ORDER BY at").fetchall()

    # extraction write-backs ------------------------------------------------
    @_locked
    def log_etl_write(self, document: str, page_id: str | None, pipeline: str, operation: str | None,
                      target: str, keys: list[str]) -> None:
        """Every field an extraction pipeline wrote back into a page, a layout or document.yaml — so
        an enrichment can be found, questioned and undone, like any other write path's output."""
        self.db.execute("INSERT INTO etl_writes(document,page_id,pipeline,operation,target,keys,at) VALUES(?,?,?,?,?,?,?)",
                        (document, page_id, pipeline, operation, target, ",".join(keys), now()))
        self.db.commit()

    @_locked
    def etl_writes(self, document: str, page_id: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM etl_writes WHERE document=?", [document]
        if page_id:
            q += " AND page_id=?"; args.append(page_id)
        return self.db.execute(q + " ORDER BY at DESC, id DESC", args).fetchall()

    # llm accounting ------------------------------------------------------
    @_locked
    def log_llm(self, page_id, provider, model, request_id, usage: dict, latency_ms: int) -> None:
        self.db.execute("INSERT INTO llm_calls(page_id,provider,model,request_id,tokens_in,tokens_out,cache_read,latency_ms,at)"
                        " VALUES(?,?,?,?,?,?,?,?,?)",
                        (page_id, provider, model, request_id, usage.get("input_tokens"), usage.get("output_tokens"),
                         usage.get("cache_read_input_tokens"), latency_ms, now()))
        self.db.commit()

    @_locked
    def costs(self):
        return self.db.execute("SELECT provider, model, COUNT(*) n, SUM(tokens_in) tin, SUM(tokens_out) tout, "
                               "SUM(cache_read) cached, AVG(latency_ms) lat FROM llm_calls GROUP BY provider, model").fetchall()

    # batches -------------------------------------------------------------
    @_locked
    def add_batch(self, batch_id, provider, model, prompt_version, page_ids: list[str]) -> None:
        self.db.execute("INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,?,?)",
                        (batch_id, provider, model, prompt_version, ",".join(page_ids), "in_progress", now(), None))
        self.db.commit()

    @_locked
    def end_batch(self, batch_id: str, status: str) -> None:
        self.db.execute("UPDATE batches SET status=?, ended_at=? WHERE id=?", (status, now(), batch_id))
        self.db.commit()

    @_locked
    def open_batches(self):
        return self.db.execute("SELECT * FROM batches WHERE status='in_progress' ORDER BY created_at").fetchall()
