"""Background jobs for the web UI: verify, build, reprocess, ingest, reconstruct, etl. One worker thread, jobs run in
order (the OCR engine is serial and the store is one connection), each with its own log."""
from __future__ import annotations
import logging, threading, time, uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ..config import Settings
from ..store import Store, DONE_STATUSES
from .. import pipeline, render
from ..session import LLMHealth, make_provider, process_page, summary

log = logging.getLogger("ocrmd.web.jobs")


def pipeline_dirs(s: Settings, slug: str) -> list[Path]:
    """Where extraction pipelines are looked for: the document's own first, then the root's shared ones."""
    return [s.documents / slug / "etl", s.root / "etl" / "pipelines"]


def pipelines(s: Settings, slug: str) -> list[Path]:
    return sorted({p.name: p for d in reversed(pipeline_dirs(s, slug)) if d.exists()
                   for p in sorted(d.glob("*.yaml"))}.values(), key=lambda p: p.name)


def pipeline_path(s: Settings, slug: str, name: str) -> Path | None:
    """A pipeline by file name. Deliberately not an arbitrary path: the browser must not be able to
    point the runner at a YAML file somewhere else on the machine."""
    return next((p for p in pipelines(s, slug) if p.name == name or p.stem == name), None)
KINDS = ("verify", "build", "reprocess", "ingest", "reconstruct", "etl")


@dataclass
class Job:
    id: str
    kind: str
    doc: str | None
    options: dict
    status: str = "queued"                  # queued | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    done: int = 0
    total: int = 0
    log: deque = field(default_factory=lambda: deque(maxlen=1000))
    cancel: threading.Event = field(default_factory=threading.Event)

    def emit(self, line: str, level: str = "info") -> None:
        self.log.append({"at": time.time(), "level": level, "line": line})

    def to_dict(self, with_log: bool = False) -> dict:
        d = {"id": self.id, "kind": self.kind, "doc": self.doc, "options": self.options, "status": self.status,
             "created_at": self.created_at, "started_at": self.started_at, "ended_at": self.ended_at,
             "error": self.error, "done": self.done, "total": self.total, "log_lines": len(self.log)}
        if with_log:
            d["log"] = list(self.log)
        return d


class JobManager:
    def __init__(self, s: Settings, st: Store, scan_running=lambda doc: False):
        self.s, self.st = s, st
        self.scan_running = scan_running
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="ocrmd-jobs", daemon=True)
        self.thread.start()

    def submit(self, kind: str, doc: str | None, options: dict | None = None) -> Job:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind}")
        if kind == "reprocess" and doc and self.scan_running(doc):
            raise ValueError(f"'{doc}' is being scanned; stop the scan before reprocessing")
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, doc=doc, options=options or {})
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
            self.queue.append(job.id)
            if len(self.order) > 200:                       # keep the history bounded
                old = self.order.pop(0)
                self.jobs.pop(old, None)
        self.wake.set()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return [self.jobs[i] for i in reversed(self.order) if i in self.jobs]

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status == "queued":
            with self.lock:
                if job_id in self.queue:
                    self.queue.remove(job_id)
            job.status, job.ended_at = "cancelled", time.time()
            return True
        job.cancel.set()
        return job.status == "running"

    def _loop(self) -> None:
        while True:
            self.wake.wait(1.0)
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    job = self.jobs[self.queue.pop(0)]
                self._run(job)

    def _run(self, job: Job) -> None:
        job.status, job.started_at = "running", time.time()
        try:
            getattr(self, f"_job_{job.kind}")(job)
            job.status = "cancelled" if job.cancel.is_set() else "done"
        except Exception as e:
            job.status, job.error = "failed", str(e)
            job.emit(f"failed: {e}", "error")
            log.exception("job %s failed", job.id)
        finally:
            job.ended_at = time.time()

    # --- job kinds -------------------------------------------------------------------------------
    def _provider(self, job: Job):
        o = job.options
        return make_provider(self.s, bool(o.get("no_llm")), o.get("provider") or self.s.provider,
                             o.get("model") or self.s.model, job.emit)

    def _job_verify(self, job: Job) -> None:
        s, st, o = self.s, self.st, job.options
        prov = self._provider(job)
        pages = st.pages(job.doc, "flagged" if o.get("flagged_only") else None)
        ocr_ver = pipeline.ocr_version(s)          # pages read by an older engine set are re-OCR'd (local, free)
        stale = lambda p: pipeline.ocr_stale(s, st, p, ocr_ver)
        if o.get("refresh"):
            current = pipeline.llm_version(s, prov.model if prov else None)
            targets = [p for p in pages if o.get("flagged_only") or stale(p) or not st.stage_done(p["id"], "llm", current)]
        else:
            targets = [p for p in pages if o.get("flagged_only") or stale(p) or p["status"] not in DONE_STATUSES]
        job.total = len(targets)
        if not targets:
            job.emit("nothing to do", "info")
            return
        if any(stale(p) for p in targets):
            job.emit(f"{sum(stale(p) for p in targets)} page(s) will be re-read with {ocr_ver}", "info")
        job.emit(f"{len(targets)} page(s) to verify" + (" (refresh)" if o.get("refresh") else ""), "info")
        health = LLMHealth(s.llm_failure_limit)
        for page in targets:
            if job.cancel.is_set():
                job.emit("cancelled", "warn"); break
            if prov is not None and health.degraded:
                job.emit("LLM unavailable; finishing the rest OCR-only.", "error")
                prov = None
            process_page(s, st, page["id"], prov, page["document"], refresh=bool(o.get("refresh")), health=health, emit=job.emit)
            job.done += 1
        for slug in {p["document"] for p in targets}:
            render.build_document(s.documents / slug, slug)
            summary(st, slug, job.emit)

    def _job_build(self, job: Job) -> None:
        slugs = [job.doc] if job.doc else self.st.documents()
        job.total = len(slugs)
        for slug in slugs:
            render.ensure_document_meta(self.s.documents / slug, slug)
            out = render.build_document(self.s.documents / slug, slug)
            job.emit(f"built {out}", "ok")
            job.done += 1

    def _job_reprocess(self, job: Job) -> None:
        s, st = self.s, self.st
        slug = job.doc
        if not slug:
            raise ValueError("reprocess needs a document")
        n = len(st.pages(slug))
        ids, removed = pipeline.reprocess_document(s, st, slug)
        job.emit(f"{slug}: {n} old page(s) -> {len(ids)} new page(s); {removed} stale artifact(s) removed", "info")
        job.total = len(ids)
        prov = self._provider(job)
        health = LLMHealth(s.llm_failure_limit)
        for pid in ids:
            if job.cancel.is_set():
                job.emit("cancelled; remaining pages are OCR-less until `verify`", "warn"); break
            process_page(s, st, pid, None if health.degraded else prov, slug, health=health, emit=job.emit)
            job.done += 1
        render.build_document(s.documents / slug, slug)
        summary(st, slug, job.emit)

    def _job_etl(self, job: Job) -> None:
        """Run an extraction pipeline over a document (see `ocrmd/etl.py`).

        A job rather than a request: a pipeline is one LLM call per page per operator, which is minutes
        for a book, and it wants the same cancel button and live log every other long job has.
        """
        from .. import etl as etl_mod
        s, st, o = self.s, self.st, job.options
        slug = job.doc
        if not slug:
            raise ValueError("etl needs a document")
        path = pipeline_path(s, slug, o.get("pipeline") or "")
        if path is None:
            raise ValueError(f"no such pipeline: {o.get('pipeline')!r}")
        pipe = etl_mod.Pipeline.load(path)
        rows = etl_mod.page_rows(s, st, slug, list(o.get("status") or []) or pipe.status or None, o.get("limit"))
        if not rows:
            job.emit("no pages with text to work on", "warn")
            return
        prov = self._provider(job)
        if prov is None and pipe.llm_steps and not o.get("no_llm"):
            raise ValueError(f"no credentials; {len(pipe.llm_steps)} operator(s) need a model")
        job.emit(f"{pipe.name}: {len(rows)} page(s) -> {' -> '.join(pipe.steps)}", "info")
        job.total = len(rows) * max(1, len(pipe.llm_steps))
        runner = etl_mod.Runner(s, st, slug, prov, use_cache=not o.get("no_cache"), dry_run=bool(o.get("dry_run")))
        for name in pipe.steps:                      # step by step, so the log and the cancel work
            if job.cancel.is_set():
                job.emit("cancelled", "warn"); return
            op = pipe.op(name)
            before = len(rows)
            rows = getattr(runner, f"_{op.type}")(op, rows)
            job.done = min(job.total, runner.stats.calls + runner.stats.cached)
            job.emit(f"{op.name} ({op.type}): {before} row(s) -> {len(rows)}", "info")
        if o.get("dry_run"):
            job.emit(f"{len(runner.prompts)} call(s) would be made; nothing was sent", "ok")
            return
        out_path, kind, fields = etl_mod.output_path(s, slug, pipe, None)
        etl_mod.write(rows, out_path, kind, fields)
        for err in runner.stats.errors[:5]:
            job.emit(err, "error")
        job.emit(f"{len(rows)} row(s) -> {out_path.relative_to(s.root) if out_path.is_relative_to(s.root) else out_path}"
                 f"  [{runner.stats.describe()}]", "ok")

    def _job_reconstruct(self, job: Job) -> None:
        """Rebuild pages from their Markdown and score each against the photograph it came from.

        Run as a job rather than inline: rasterising a page takes a second or two, and a book's worth
        of them is a long wait for a request. The score is the point — a page that rebuilds badly is a
        page the transcription lost something on.
        """
        import cv2
        from .. import layout as layout_mod, reconstruct as rc_mod
        from ..ocr import OcrResult
        s, st = self.s, self.st
        slug = job.doc
        if not slug:
            raise ValueError("reconstruct needs a document")
        out = s.root / "reconstructed" / slug
        out.mkdir(parents=True, exist_ok=True)
        if not rc_mod.backends():
            job.emit("no SVG rasteriser (inkscape, rsvg-convert or Chrome): writing files, "
                     "but fidelity cannot be scored", "warn")
        pages = st.pages(slug)
        job.total = len(pages)
        scores = []
        for page in pages:
            if job.cancel.is_set():
                job.emit("cancelled", "warn"); break
            pid = page["id"]
            img = cv2.imread(str(s.processed / f"{pid}.png"))
            ocr_path = s.ocr_dir / f"{pid}.json"
            md_path = render.page_path(s.documents / slug, page["sequence"])
            if img is None or not ocr_path.exists() or not md_path.exists():
                job.emit(f"page {page['sequence']:04d}: missing artifacts, skipped", "warn")
                job.done += 1
                continue
            res = OcrResult.model_validate_json(ocr_path.read_text())
            fm, body = render.read_page(md_path)
            lay = layout_mod.build(res, body, (img.shape[1], img.shape[0]))
            lay.figures = layout_mod.figures(img, res.blocks)
            r = rc_mod.rebuild(img, lay, title=(fm.get("context") or {}).get("chapter"))
            stem = out / f"{page['sequence']:04d}"
            stem.with_suffix(".svg").write_text(r.svg, encoding="utf-8")
            stem.with_suffix(".tex").write_text(r.latex, encoding="utf-8")
            if r.png is not None:
                cv2.imwrite(str(stem.with_suffix(".png")), r.png)
            if r.fidelity is not None:
                scores.append(r.fidelity)
            job.emit(f"page {page['sequence']:04d}: {r.describe()}",
                     "warn" if (r.fidelity or 1) < 0.5 else "ok")
            job.done += 1
        if scores:
            job.emit(f"{len(scores)} page(s) into {out}; mean fidelity {sum(scores) / len(scores):.2f}",
                     "info")

    def _job_ingest(self, job: Job) -> None:
        s, st = self.s, self.st
        slug = job.doc
        files = [Path(p) for p in job.options.get("files", [])]
        if not slug or not files:
            raise ValueError("ingest needs a document and files")
        st.ensure_document(slug)
        render.ensure_document_meta(s.documents / slug, slug)
        prov = self._provider(job)
        job.total = len(files)
        health = LLMHealth(s.llm_failure_limit)
        for f in files:
            if job.cancel.is_set():
                job.emit("cancelled", "warn"); break
            img = cv2.imread(str(f))
            if img is None:
                job.emit(f"unreadable: {f.name}", "error"); job.done += 1; continue
            pids = pipeline.ingest_frame(s, st, img, slug, f"file:{f.name}").pages
            if not pids:
                job.emit(f"duplicate: {f.name}", "warn"); job.done += 1; continue
            job.emit(f"{f.name} -> {len(pids)} page(s)", "info")
            for pid in pids:
                process_page(s, st, pid, None if health.degraded else prov, slug, health=health, emit=job.emit)
            job.done += 1
        render.build_document(s.documents / slug, slug)
        summary(st, slug, job.emit)
