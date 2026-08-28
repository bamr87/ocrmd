"""Scan orchestration shared by the CLI and the web UI.

`process_page` runs the pipeline for one page and reports through an `emit(line, level)` callback
instead of printing, so the same code feeds a terminal, a job log, or a browser. `ScanSession` owns a
scanning run: the capture generator, the OCR worker, the LLM pool, health/degraded mode, the event
buffer, and — for the web UI — the latest HUD frame as JPEG plus a command queue standing in for the
preview window's keyboard.
"""
from __future__ import annotations
import logging, queue, threading, time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import cv2

from .config import Settings
from .store import Store
from . import legibility, pipeline, render
from .hud import ScanState
from .llm import LLMUnavailable, LLMError

log = logging.getLogger("ocrmd.session")
Emit = Callable[[str, str], None]          # (line, level) level: info | ok | warn | error | advice | note
STATUS_LEVEL = {"verified": "ok", "reviewed": "ok", "flagged": "warn", "suspect": "error", "illegible": "error",
                "blank": "info", "unverified": "info", "error": "error"}


def _noop(_line: str, _level: str = "info") -> None:
    pass


class LLMHealth:
    """Consecutive *service* failures (not page failures). Past the limit a scan stops calling the LLM:
    with the network down, every page would otherwise wait out the retries and the queue would grow
    for nothing — the captures are safe and `ocrmd verify` completes them later."""

    def __init__(self, limit: int):
        self.limit, self.failures, self.lock = limit, 0, threading.Lock()

    def record(self, ok: bool) -> None:
        with self.lock:
            self.failures = 0 if ok else self.failures + 1

    @property
    def degraded(self) -> bool:
        return self.limit > 0 and self.failures >= self.limit


def process_page(s: Settings, st: Store, page_id: str, prov, slug: str, state: ScanState | None = None,
                 refresh: bool = False, health: LLMHealth | None = None, emit: Emit = _noop) -> str:
    """Run the pipeline for one page, report the outcome, and never raise for a page-level problem."""
    page = st.page(page_id)
    label = f"{page['sequence']:04d}"

    def on_stage(stage: str) -> None:
        if state is not None:
            state.set_active(label, stage)

    def report(r: pipeline.PageResult, extra: str = "") -> None:
        emit(f"page {label} -> {r.path.name}  [{r.status}{r.detail()}]{extra}", STATUS_LEVEL.get(r.status, "info"))
        if r.advice:
            emit(f"    {r.advice}", "advice")
        if r.continuity:
            emit(f"    {r.continuity}", "note")
        if r.preserved:
            emit(f"    kept your edits: {', '.join(r.preserved)}", "info")
        if state is not None:
            state.note(label, r.status)
            state.advice, state.error = r.continuity or r.advice, None

    try:
        r = pipeline.run_page(s, st, page, prov, slug, on_stage=on_stage, refresh=refresh)
        if prov is not None and health is not None:
            health.record(True)
        report(r)
        return r.status
    except LLMError as e:
        if isinstance(e, LLMUnavailable) and health is not None:
            health.record(False)
        emit(f"page {label}: LLM {'unavailable' if isinstance(e, LLMUnavailable) else 'failed'}: {e}", "error")
        log.warning("page %s llm: %s", page_id, e)
        try:                                   # always leave a page file behind: OCR-only, marked unverified
            report(pipeline.run_page(s, st, page, None, slug, on_stage=on_stage), extra="  (OCR only)")
        except Exception as e2:
            log.error("page %s render after llm failure: %s", page_id, e2)
        if state is not None:
            state.note(label, "error"); state.error = str(e)[:110]
        return "error"
    except Exception as e:
        emit(f"page {label} failed: {e}", "error")
        log.exception("page %s failed", page_id)
        if state is not None:
            state.note(label, "error"); state.error = str(e)[:110]
        return "error"
    finally:
        if state is not None:
            state.set_active(label, None)


def summary(st: Store, slug: str, emit: Emit) -> dict:
    counts = st.counts(slug)
    if not counts:
        return {}
    parts = ", ".join(f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    emit(f"{slug}: {sum(counts.values())} page(s) — {parts}", "info")
    bad = [p["sequence"] for p in st.pages(slug) if p["status"] in ("illegible", "suspect")]
    if bad:
        emit(f"  re-shoot page(s) {', '.join(f'{n:04d}' for n in bad[:12])}{'…' if len(bad) > 12 else ''} "
             "closer or better lit (`ocrmd status` lists why)", "advice")
    if counts.get("flagged"):
        emit(f"  {counts['flagged']} flagged page(s): `ocrmd review --doc {slug}` walks the disagreements", "info")
    if counts.get("unverified") or counts.get("error"):
        emit(f"  {counts.get('unverified', 0) + counts.get('error', 0)} page(s) without an LLM pass: `ocrmd verify --doc {slug}`", "info")
    return counts


@dataclass
class Event:
    at: float
    level: str
    line: str


@dataclass
class ScanOptions:
    doc: str
    camera: str | int = 0
    focus: str = "continuous"
    resolution: tuple[int, int] | None = None
    rotate: int | None = None
    zoom: float | None = None
    duration: float | None = None
    max_pages: int | None = None
    bracket: str | None = None            # focus bracket: auto | always | off (None: the configured default)
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    project: str | None = None


class ScanSession:
    """One scanning run. `run()` executes in the calling thread (the CLI needs the OpenCV window on the
    main thread); `start()` runs it on a thread for the web UI, which reads `latest_jpeg()` and
    `snapshot()` and sends `command()`s instead of key presses."""

    def __init__(self, s: Settings, st: Store, opts: ScanOptions, provider, preview: bool, emit: Emit = _noop):
        self.s, self.st, self.opts, self.provider, self.preview = s, st, opts, provider, preview
        self.state = ScanState()
        self.health = LLMHealth(s.llm_failure_limit)
        self.events: deque[Event] = deque(maxlen=400)
        self._emit_cb = emit
        self.commands: "queue.Queue[str]" = queue.Queue()
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._frame_cond = threading.Condition()
        self.gate: dict = {}
        self.camera_name: str | None = None
        self.camera_index: int | None = None
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- reporting ---------------------------------------------------------------------------------
    def emit(self, line: str, level: str = "info") -> None:
        self.events.append(Event(time.time(), level, line))
        self._emit_cb(line, level)

    @property
    def running(self) -> bool:
        return self.started_at is not None and self.ended_at is None

    def latest_jpeg(self, after_seq: int = -1, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._frame_cond:
            if self._jpeg_seq <= after_seq:
                self._frame_cond.wait(timeout)
            return self._jpeg, self._jpeg_seq

    def snapshot(self) -> dict:
        st = self.state
        return {
            "doc": self.opts.doc, "running": self.running, "error": self.error,
            "camera": {"name": self.camera_name, "index": self.camera_index},
            "started_at": self.started_at, "ended_at": self.ended_at,
            "gate": self.gate,
            "captured": st.captured, "queued": st.queued, "working": st.working, "active": dict(st.active),
            "degraded": st.degraded, "advice": st.advice, "last_error": st.error,
            "recent": [{"label": lbl, "status": s_} for lbl, s_ in st.recent],
            "llm": {"provider": self.provider.name, "model": self.provider.model} if self.provider else None,
            "events": [{"at": e.at, "level": e.level, "line": e.line} for e in list(self.events)[-60:]],
        }

    # --- control -------------------------------------------------------------------------------------
    def command(self, cmd: str) -> None:
        """quit | capture | focus_lock | focus_auto | rotate | rotate_ccw | zoom_in | zoom_out |
        pan_up | pan_down | pan_left | pan_right | reset"""
        self.commands.put(cmd)

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name="ocrmd-scan", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self.command("quit")
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout)

    # --- the run -------------------------------------------------------------------------------------
    def run(self) -> None:
        from .capture import auto_shutter, resolve_camera, camera_name, CameraLost, Frame
        from .camview import load_view
        s, st, o = self.s, self.st, self.opts
        self.started_at = time.time()
        try:
            self.camera_index = s.camera_index = resolve_camera(o.camera)
            self.camera_name = camera_name(self.camera_index)
        except RuntimeError as e:
            self.error = str(e); self.emit(str(e), "error"); self.ended_at = time.time()
            return
        if o.bracket is not None:
            s.focus_bracket = o.bracket
        view = load_view(s.root, self.camera_name)
        if o.rotate is not None: view.rotation = o.rotate % 360
        if o.zoom is not None: view.zoom = max(1.0, o.zoom)
        st.ensure_document(o.doc)
        render.ensure_document_meta(s.documents / o.doc, o.doc, title=o.title, tags=o.tags, project=o.project)
        prov, state, health = self.provider, self.state, self.health
        rejected: list[str] = []
        self.emit(f"scanning into {s.documents / o.doc}  (camera {self.camera_index}: {self.camera_name})"
                  + (f"; llm={prov.name}/{prov.model} x{s.llm_workers}" if prov else "; OCR only"), "info")

        # Capture, OCR and the LLM pass are three separate lanes. Processing inline would suspend the
        # capture generator — no frames read, no key events — so the capture loop only enqueues frames.
        # One worker does preprocess+OCR (Vision is CPU-bound and serial anyway) and renders an OCR-only
        # page immediately; a small pool then runs the LLM pass and re-renders when it lands.
        work: "queue.Queue[Frame | None]" = queue.Queue()
        pool = ThreadPoolExecutor(max_workers=max(1, s.llm_workers), thread_name_prefix="ocrmd-llm") if prov else None
        futures: list[Future] = []

        def pending() -> int:
            return work.qsize() + sum(1 for f in futures if not f.done())

        def llm_job(pid: str) -> None:
            if health.degraded:
                return
            process_page(s, st, pid, prov, o.doc, state=state, health=health, emit=self.emit)
            if health.degraded and not state.degraded:
                state.degraded = True
                self.emit(f"LLM unavailable after {health.failures} consecutive failures — continuing OCR-only. "
                          f"Run `ocrmd verify --doc {o.doc}` when the service is back.", "error")
                log.error("llm degraded after %d failures", health.failures)
            state.queued = pending()

        def worker() -> None:
            while True:
                fr = work.get()
                try:
                    if fr is None:
                        return
                    state.set_active("capture", "preprocess")
                    # `admit` only for automatic captures: a frame the operator forced with the space
                    # bar is a decision, and the shutter's opinion of it is not the deciding one.
                    got = pipeline.ingest_frame(s, st, fr.image, o.doc, f"webcam:{self.camera_name}",
                                                [v.image for v in fr.variants],
                                                admit=s.admit_captures and fr.reason == "auto")
                    state.set_active("capture", None)
                    for why in got.rejected:
                        rejected.append(why)
                        self.emit(f"capture refused: {why} — no page made, the frame is kept; re-shoot it", "warn")
                        state.note("refused", "illegible")
                    if got.duplicate_of:
                        self.emit("duplicate page, skipped", "warn")
                        state.note("dup", "blank")
                        continue
                    if not got.pages:
                        continue
                    for pid in got.pages:
                        process_page(s, st, pid, None, o.doc, state=state, emit=self.emit)      # OCR-only, right now
                        if pool is not None and not health.degraded:
                            futures.append(pool.submit(llm_job, pid))                            # LLM upgrade later
                except Exception as e:                      # a bad page must never kill the worker
                    self.emit(f"processing failed: {e}", "error")
                    log.exception("worker failed")
                    state.error = str(e)[:110]
                finally:
                    work.task_done()
                    state.queued = pending()
                    state.set_active("capture", None)

        def on_frame(disp, info: dict) -> None:
            self.gate = info
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._frame_cond:
                    self._jpeg, self._jpeg_seq = buf.tobytes(), self._jpeg_seq + 1
                    self._frame_cond.notify_all()

        thread = threading.Thread(target=worker, name="ocrmd-worker", daemon=True)
        thread.start()
        try:
            for fr in auto_shutter(s, preview=self.preview, on_status=lambda m: self.emit(m, "info"), view=view,
                                   resolution=o.resolution, focus=o.focus, state=state, duration=o.duration,
                                   max_captures=o.max_pages, on_frame=None if self.preview else on_frame,
                                   commands=self.commands):
                bracket = f", {len(fr.variants)} focus variant(s)" if fr.variants else ""
                self.emit(f"captured ({fr.reason}; sharpness {fr.meta.get('sharpness')}, "
                          f"{fr.meta.get('page_px')}px/page{bracket})", "ok")
                work.put(fr)
                state.queued = pending()
        except KeyboardInterrupt:
            self.emit("interrupted", "warn")
        except (CameraLost, RuntimeError) as e:
            self.error = str(e); self.emit(str(e), "error")
        finally:
            left = pending()
            if left:
                self.emit(f"finishing {left} page(s) still processing...", "info")
            try:
                work.join()
                work.put(None)
                thread.join(timeout=5)
                if pool is not None:
                    pool.shutdown(wait=True)
            except KeyboardInterrupt:
                self.emit(f"abandoning {pending()} page(s); `ocrmd verify --doc {o.doc}` finishes them", "warn")
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
            render.build_document(s.documents / o.doc, o.doc)
            if rejected:
                self.emit(legibility.advice(rejected), "advice")
            summary(st, o.doc, self.emit)
            self.ended_at = time.time()
            with self._frame_cond:
                self._frame_cond.notify_all()


def make_provider(s: Settings, no_llm: bool, provider: str, model: str, emit: Emit = _noop):
    """The configured LLM provider, or None (OCR-only) with a reason emitted."""
    from . import llm as llm_mod
    if no_llm:
        return None
    if not llm_mod.credentials_available(provider):
        emit(f"no credentials for {provider}; running OCR-only. Set ANTHROPIC_API_KEY (or CLAUDE_CODE_OAUTH_TOKEN) "
             "in .env, then `ocrmd verify`.", "warn")
        return None
    if provider == "xai" and model == s.model:
        model = s.xai_model
    try:
        return llm_mod.get_provider(provider, model, timeout=s.llm_timeout, max_retries=s.llm_max_retries, effort=s.llm_effort)
    except LLMUnavailable as e:
        emit(f"{e}; running OCR-only", "warn")
        return None
