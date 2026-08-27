from __future__ import annotations
import logging, os, time
from pathlib import Path
import typer, cv2
from .config import settings as S, setup_logging
from .store import Store, DONE_STATUSES
from . import ocr, pipeline, render, review as review_mod
from .llm import LLMUnavailable
from .session import LLMHealth, ScanOptions, ScanSession, make_provider, process_page, summary

app = typer.Typer(help="ocrmd — incremental scanning + OCR to cross-checked Markdown", no_args_is_help=True)
DEBUG = False
log = logging.getLogger("ocrmd.cli")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
LEVEL_COLOR = {"ok": "green", "warn": "yellow", "error": "red", "advice": "yellow", "note": "magenta", "info": None}
STATUS_COLOR = {"verified": "green", "reviewed": "green", "flagged": "yellow", "suspect": "red", "illegible": "red",
                "blank": "blue", "unverified": "white", "error": "red"}


def emit(line: str, level: str = "info") -> None:
    """Console reporter for the session/pipeline helpers."""
    typer.secho(("  " if line.startswith("page ") or line.startswith("    ") else "") + line, fg=LEVEL_COLOR.get(level))


@app.callback()
def main(debug: bool = typer.Option(False, "--debug", help="print full tracebacks and debug logs"),
         quiet: bool = typer.Option(False, "--quiet", "-q", help="errors only on the console")):
    global DEBUG
    DEBUG = debug
    setup_logging(S, debug=debug, quiet=quiet)
    log.debug("settings: %s", S.model_dump(exclude={"root"}))


def _store() -> Store:
    S.ensure_dirs()
    return Store(S.db_path)


def _provider(no_llm: bool, provider: str, model: str):
    return make_provider(S, no_llm, provider, model, emit)


def _fail(msg: str, code: int = 1):
    typer.secho(msg, fg="red", err=True)
    raise typer.Exit(code)


def _process(st, page_id, prov, slug, refresh: bool = False, health: LLMHealth | None = None) -> str:
    return process_page(S, st, page_id, prov, slug, refresh=refresh, health=health, emit=emit)


def _summary(st, slug: str) -> None:
    summary(st, slug, emit)


# =================================================================================================
@app.command()
def scan(doc: str = typer.Option(..., help="document slug"),
         camera: str = typer.Option(str(S.camera_index), help="index or device-name substring, e.g. c525"),
         no_llm: bool = typer.Option(False, "--no-llm", help="capture + OCR only; fill in the LLM pass later with `ocrmd verify`"),
         provider: str = S.provider, model: str = S.model, preview: bool = True,
         rotate: int | None = typer.Option(None, help="initial rotation 0/90/180/270 (overrides saved view)"),
         zoom: float | None = typer.Option(None, help="initial digital zoom, e.g. 2.0 (overrides saved view)"),
         focus: str = typer.Option(S.focus, help="continuous | locked | settle (lock after the first capture) | off"),
         resolution: str | None = typer.Option(None, help="WxH, e.g. 1280x960; default is the largest mode"),
         title: str | None = typer.Option(None, help="document title (stored in document.yaml)"),
         tag: list[str] = typer.Option([], "--tag", help="tag for document.yaml; repeatable"),
         project: str | None = typer.Option(None, help="project name for document.yaml"),
         duration: float | None = typer.Option(None, help="stop after this many seconds (unattended runs)"),
         max_pages: int | None = typer.Option(None, help="stop after this many captures")):
    """Auto-capture pages from the webcam and process them incrementally.

    The shutter fires when the scene is still, the lens has settled, no hand is over the page, the page
    is new and it is sharp. Preview keys: r rotate, +/- zoom, wasd/arrows pan, f/F focus, 0 reset,
    space force capture, q quit. The view is saved per camera.
    """
    st = _store()
    prov = _provider(no_llm, provider, model)
    res = tuple(int(x) for x in resolution.lower().split("x")) if resolution else S.resolution
    opts = ScanOptions(doc=doc, camera=camera, focus=focus, resolution=res, rotate=rotate, zoom=zoom,
                       duration=duration, max_pages=max_pages, title=title, tags=list(tag), project=project)
    typer.echo("space=force capture, q=quit")
    session = ScanSession(S, st, opts, prov, preview=preview, emit=emit)
    session.run()                                   # in this thread: the preview window needs the main thread
    if session.error:
        raise typer.Exit(1)


def _pdf_images(path: Path, dpi: int = 200):
    try:
        import fitz  # pymupdf
    except ImportError:
        _fail(f"{path.name}: PDF support needs pymupdf — pip install 'ocrmd[pdf]'")
    import numpy as np
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            yield f"{path.name}#p{i + 1}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _input_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_SUFFIXES + (".pdf",)]
        elif p.exists():
            out.append(p)
        else:
            typer.secho(f"  not found: {p}", fg="red")
    return out


def _ingest_file(st, f: Path, doc: str, prov, refresh: bool = False) -> None:
    sources = _pdf_images(f) if f.suffix.lower() == ".pdf" else [(f.name, cv2.imread(str(f)))]
    for name, img in sources:
        if img is None:
            typer.secho(f"  unreadable: {name}", fg="red"); continue
        pids = pipeline.ingest_frame(S, st, img, doc, f"file:{name}")
        if not pids:
            typer.secho(f"  duplicate: {name}", fg="yellow"); continue
        typer.echo(f"{name} -> {len(pids)} page(s)")
        for pid in pids:
            _process(st, pid, prov, doc, refresh=refresh)


@app.command()
def ingest(paths: list[Path], doc: str = typer.Option(..., help="document slug"),
           no_llm: bool = typer.Option(False, "--no-llm", help="capture + OCR only"),
           provider: str = S.provider, model: str = S.model,
           watch: bool = typer.Option(False, "--watch", help="keep watching the directory for new files (Ctrl-C to stop)"),
           title: str | None = typer.Option(None), tag: list[str] = typer.Option([], "--tag"),
           project: str | None = typer.Option(None)):
    """Ingest image files or PDFs (png/jpg/tiff/webp, pdf with the `pdf` extra) as pages."""
    st = _store(); st.ensure_document(doc)
    render.ensure_document_meta(S.documents / doc, doc, title=title, tags=list(tag), project=project)
    prov = _provider(no_llm, provider, model)
    seen: set[Path] = set()
    try:
        while True:
            files = [f for f in _input_files(paths) if f not in seen]
            for f in files:
                seen.add(f)
                _ingest_file(st, f, doc, prov)
            if not watch:
                break
            if files:
                render.build_document(S.documents / doc, doc)
            time.sleep(2.0)
    except KeyboardInterrupt:
        typer.echo("\nstopped watching")
    render.build_document(S.documents / doc, doc)
    _summary(st, doc)


@app.command()
def verify(doc: str | None = None,
           flagged_only: bool = typer.Option(False, "--flagged-only", help="re-run only pages already marked flagged"),
           refresh: bool = typer.Option(False, "--refresh", help="also re-run pages whose LLM pass used an older prompt version (costs tokens)"),
           batch: bool = typer.Option(False, "--batch", help="send the LLM pass through the Batches API (half price, up to hours of latency)"),
           wait: bool = typer.Option(True, help="with --batch: poll until the batch ends and collect it"),
           collect: bool = typer.Option(False, "--collect", help="collect batches submitted earlier with --batch --no-wait"),
           provider: str = S.provider, model: str = S.model):
    """(Re)run missing/stale stages — e.g. the LLM pass for pages scanned with --no-llm or while offline.

    Without --refresh a completed LLM pass is always reused, whatever prompt version produced it: a
    version bump must never silently spend tokens on a whole book.
    """
    st = _store()
    prov = _provider(False, provider, model)
    if collect:
        _collect_batches(st, prov)
        return
    pages = st.pages(doc, "flagged" if flagged_only else None)
    ocr_ver = pipeline.ocr_version(S)      # pages read by an older engine set are re-OCR'd (local, free)
    stale = lambda p: pipeline.ocr_stale(S, st, p, ocr_ver)
    if refresh:
        current = pipeline.llm_version(S, prov.model if prov else None)
        targets = [p for p in pages if flagged_only or stale(p) or not st.stage_done(p["id"], "llm", current)]
    else:
        targets = [p for p in pages if flagged_only or stale(p) or p["status"] not in DONE_STATUSES]
    if any(stale(p) for p in targets):
        typer.echo(f"{sum(stale(p) for p in targets)} page(s) will be re-read with {ocr_ver}")
    if not targets:
        typer.echo("nothing to do")
        return
    if batch:
        if prov is None or prov.name != "anthropic":
            _fail("--batch needs the anthropic provider with credentials")
        _run_batch(st, prov, targets, wait)
    else:
        typer.echo(f"{len(targets)} page(s) to verify" + (" (refresh)" if refresh else ""))
        health = LLMHealth(S.llm_failure_limit)
        for page in targets:
            if prov is not None and health.degraded:
                typer.secho("LLM unavailable; finishing the rest OCR-only. Re-run `ocrmd verify` when the service is back.", fg="red")
                prov = None
            _process(st, page["id"], prov, page["document"], refresh=refresh, health=health)
    for slug in {p["document"] for p in targets}:
        render.build_document(S.documents / slug, slug)
        _summary(st, slug)


def _run_batch(st, prov, targets, wait: bool) -> None:
    # only pages that actually need an LLM pass (or a refresh) go into the batch
    need = [p for p in targets if pipeline.llm_cached(S, st, p, prov.name, True, prov.model) is None]
    if not need:
        typer.echo("all target pages already have a current LLM pass"); return
    typer.echo(f"submitting {len(need)} page(s) to the Batches API...")
    try:
        batch_id, pids = pipeline.submit_batch(S, st, need, prov, lambda p: p["document"])
    except LLMUnavailable as e:
        _fail(str(e))
    typer.echo(f"batch {batch_id}: {len(pids)} page(s) queued")
    if not wait:
        typer.echo("collect later with `ocrmd verify --collect`")
        return
    pipeline.wait_for_batch(prov, batch_id, S.batch_poll_seconds, on_status=typer.echo)
    _collect_one(st, prov, batch_id)


def _collect_one(st, prov, batch_id: str) -> None:
    outcomes = pipeline.collect_batch(S, st, batch_id, prov, on_page=lambda page, r: typer.secho(
        f"  page {page['sequence']:04d} ({page['document']}) -> {r.path.name}  [{r.status}{r.detail()}]",
        fg=STATUS_COLOR.get(r.status, "white")))
    errs = {k: v for k, v in outcomes.items() if v.startswith("error")}
    typer.echo(f"batch {batch_id}: {len(outcomes) - len(errs)} ok, {len(errs)} failed")
    for pid, msg in errs.items():
        typer.secho(f"  {pid}: {msg}", fg="red")


def _collect_batches(st, prov) -> None:
    if prov is None or prov.name != "anthropic":
        _fail("--collect needs the anthropic provider with credentials")
    open_b = st.open_batches()
    if not open_b:
        typer.echo("no open batches"); return
    slugs = set()
    for b in open_b:
        status, counts = prov.batch_status(b["id"])
        typer.echo(f"batch {b['id']} ({b['created_at']}): {status} {counts}")
        if status == "ended":
            _collect_one(st, prov, b["id"])
            slugs |= {st.page(pid)["document"] for pid in b["page_ids"].split(",") if st.page(pid)}
    for slug in slugs:
        render.build_document(S.documents / slug, slug)
        _summary(st, slug)


@app.command()
def reprocess(doc: str | None = None, no_llm: bool = typer.Option(False, "--no-llm"),
              provider: str = S.provider, model: str = S.model,
              yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation prompt")):
    """Re-derive pages from the raw captures with the current preprocessing pipeline.

    Use after upgrading ocrmd: page ids come from the processed image, so improved cropping,
    rotation or spread-splitting produces different pages. Cached OCR/LLM results for the old
    pages are discarded (the LLM pass re-runs and costs tokens); raw captures are never touched.
    Human-edited page files are also replaced — export them first if they matter.
    """
    st = _store()
    slugs = [doc] if doc else st.documents()
    for slug in slugs:
        n = len(st.pages(slug))
        if not yes and not typer.confirm(f"reprocess '{slug}': {n} page(s) will be rebuilt and their cached "
                                          "OCR/LLM results discarded. Continue?", default=True):
            continue
        ids, removed = pipeline.reprocess_document(S, st, slug)
        typer.echo(f"{slug}: {n} old page(s) -> {len(ids)} new page(s); {removed} stale artifact(s) removed")
        prov = _provider(no_llm, provider, model)
        health = LLMHealth(S.llm_failure_limit)
        for pid in ids:
            _process(st, pid, None if health.degraded else prov, slug, health=health)
        render.build_document(S.documents / slug, slug)
        _summary(st, slug)


@app.command()
def review(doc: str = typer.Option(..., help="document slug"),
           who: str = typer.Option(os.environ.get("USER", "reviewer"), help="reviewer name recorded in the page"),
           list_only: bool = typer.Option(False, "--list", help="print the open disagreements and exit")):
    """Step through flagged disagreements: pick the OCR or the LLM reading, type your own, or skip.

    A page with no disagreements left is signed off as `reviewed`; the pipeline then keeps its body
    on later runs. Answers: o=OCR  l=LLM  t=type  k=keep unresolved  s=skip page  q=quit.
    (The web UI — `ocrmd serve` — does the same with the page image alongside.)
    """
    st = _store()
    doc_dir = S.documents / doc
    pages = [p for p in st.pages(doc) if p["status"] == "flagged"]
    if not pages:
        typer.echo("no flagged pages"); return
    done = 0
    for page in pages:
        path = render.page_path(doc_dir, page["sequence"])
        if not path.exists():
            continue
        fm, body = render.read_page(path)
        flags = review_mod.parse_flags(body)
        header = f"page {page['sequence']:04d} (label {page['page_label'] or '?'}): {len(flags)} open"
        if list_only:
            typer.secho(header, fg="yellow")
            for fl in flags:
                typer.echo(f"   #{fl.n} ocr={fl.ocr!r} llm={fl.llm!r}  {fl.context(body, 40)}")
            continue
        typer.secho(f"\n{header}", fg="yellow", bold=True)
        quit_all, skipped = False, False
        for fl in flags:
            typer.echo(f"  {fl.context(body)}")
            typer.echo(f"     ocr: {fl.ocr!r}    llm: {fl.llm!r}")
            ans = typer.prompt("     [o]cr [l]lm [t]ype [k]eep [s]kip page [q]uit", default="k").strip().lower()
            if ans == "q": quit_all = True; break
            if ans == "s": skipped = True; break
            if ans == "k": continue
            chosen = fl.ocr if ans == "o" else fl.llm if ans == "l" else typer.prompt("     text") if ans == "t" else None
            if chosen is None:
                continue
            body = review_mod.resolve(body, fl.n, chosen)
        if quit_all:
            break
        if skipped:
            continue
        left = review_mod.unresolved(body)
        v = fm.setdefault("verification", {})
        v["flagged_spans"] = left
        if left == 0:
            v["reviewed_by"] = {"who": who, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            v["status"] = "reviewed"
            st.set_reviewed(page["id"], who, "reviewed")
            done += 1
        render.write_page(doc_dir, page["sequence"], fm, body)
        st.set_fingerprint(page["id"], review_mod.fingerprint(fm, body))
        typer.secho(f"  saved; {left} left" + ("  -> reviewed" if left == 0 else ""), fg="green")
    if not list_only:
        render.build_document(doc_dir, doc)
        typer.echo(f"{done} page(s) signed off")


@app.command()
def status(doc: str | None = None, all_pages: bool = typer.Option(False, "--all", help="list every page, not just problems")):
    """Per-page status: what needs a re-shoot, a review, or an LLM pass — and why."""
    st = _store()
    pipeline.backfill_page_meta(S, st)
    for slug in ([doc] if doc else st.documents()):
        pages = st.pages(slug)
        if not pages:
            continue
        typer.secho(f"{slug}", bold=True)
        typer.echo(f"  {'seq':>4} {'label':>6} {'status':<10} {'px':>5} {'agree':>5}  note")
        for p in pages:
            problem = p["status"] not in DONE_STATUSES or p["advice"]
            if not all_pages and not problem:
                continue
            agree = ""
            path = render.page_path(S.documents / slug, p["sequence"])
            if path.exists():
                try:
                    v = render.read_page(path)[0].get("verification") or {}
                    agree = f"{v['agreement']:.2f}" if v.get("agreement") is not None else ""
                except Exception:
                    pass
            note = st.stage_error(p["id"], "llm") or p["advice"] or ""
            typer.secho(f"  {p['sequence']:>4} {str(p['page_label'] or ''):>6} {p['status']:<10} "
                        f"{p['width_px'] or '':>5} {agree:>5}  {note[:80]}", fg=STATUS_COLOR.get(p["status"], "white"))
        _summary(st, slug)
        dups = st.duplicates(slug)
        if dups:
            typer.echo(f"  {len(dups)} duplicate capture(s) were skipped")


@app.command()
def delete(doc: str = typer.Option(..., help="document slug"),
           page: list[int] = typer.Option([], "--page", "-p", help="sequence number(s) to delete; repeatable"),
           whole_capture: bool = typer.Option(False, "--whole-capture", help="also delete the other half of each spread"),
           all_pages: bool = typer.Option(False, "--all", help="delete the whole document"),
           keep_raw: bool = typer.Option(False, "--keep-raw", help="keep raw captures that nothing refers to any more"),
           yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation prompt")):
    """Delete pages (or a whole document): database rows, artifacts, page files — and the raw capture
    when no page refers to it any more. Irreversible; the web UI offers the same with confirmations."""
    st = _store()
    if doc not in st.documents():
        _fail(f"no such document: {doc}")
    if all_pages:
        n = len(st.pages(doc))
        if not yes and not typer.confirm(f"delete document '{doc}' with {n} page(s), its files and raw captures?", default=False):
            raise typer.Exit()
        out = pipeline.delete_document(S, st, doc, purge_raw=not keep_raw)
        typer.echo(f"{doc}: {out.summary()}")
        return
    if not page:
        _fail("give --page N (repeatable) or --all")
    ids = [r["id"] for seq in page if (r := st.page_by_sequence(doc, seq)) is not None]
    missing = [seq for seq in page if st.page_by_sequence(doc, seq) is None]
    if missing:
        typer.secho(f"  no such page(s): {missing}", fg="yellow")
    if not ids:
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"delete {len(ids)} page(s) from '{doc}'" + (" and their spread partners" if whole_capture else "") + "?", default=False):
        raise typer.Exit()
    out = pipeline.delete_pages(S, st, doc, ids, purge_raw=not keep_raw, whole_capture=whole_capture)
    render.build_document(S.documents / doc, doc)
    typer.echo(f"{doc}: {out.summary()} (pages {', '.join(f'{n:04d}' for n in out.pages)})")


@app.command()
def build(doc: str | None = None):
    """Regenerate document markdown + nav links from page files (and document.yaml)."""
    st = _store()
    for slug in ([doc] if doc else st.documents()):
        render.ensure_document_meta(S.documents / slug, slug)
        typer.echo(render.build_document(S.documents / slug, slug))


@app.command()
def serve(host: str = typer.Option("127.0.0.1", help="bind address; keep it local, there is no authentication"),
          port: int = typer.Option(8765),
          open_browser: bool = typer.Option(True, "--open/--no-open", help="open the UI in a browser")):
    """Run the web UI: live scanning, page browser, review, jobs, metadata — at http://127.0.0.1:8765."""
    import threading, webbrowser
    import uvicorn
    from .web.app import create_app
    application = create_app(S)
    url = f"http://{host}:{port}"
    typer.echo(f"ocrmd web UI at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command("ocr-bench")
def ocr_bench(doc: str | None = None,
              engines: str | None = typer.Option(None, "--engines", "-e", help="engine set to measure, e.g. apple_vision or apple_vision+rapidocr (default: configured)"),
              compare: bool = typer.Option(False, "--compare", help="measure every installed engine alone and every pair"),
              limit: int | None = typer.Option(None, "--limit", help="only the first N reference pages"),
              verbose: bool = typer.Option(False, "--verbose", "-v", help="one line per page")):
    """Score local OCR against the LLM transcriptions in the store (bag-of-words F1 per page).

    Uses every page that has a legible model transcription; the numbers that chose the default engines
    are in ocr.py. Run it after installing an engine or changing the camera to see whether it paid off.
    """
    from . import bench
    st = _store()
    sets: list[list[str] | None]
    if compare:
        names = ocr.available_engines()
        sets = [[n] for n in names] + [[a, b] for i, a in enumerate(names) for b in names[i + 1:]]
        if not sets:
            _fail("no OCR engine installed")
    else:
        sets = [[x.strip() for x in engines.replace("+", ",").split(",") if x.strip()] if engines else None]
    n_ref = len(bench.reference_pages(S, st, doc))
    if not n_ref:
        _fail("no reference pages: verify some pages with the LLM first (`ocrmd verify`)")
    typer.echo(f"{min(n_ref, limit) if limit else n_ref} reference page(s)" + (f" in {doc}" if doc else ""))
    header = f"{'engines':34s} {'pages':>5s} {'recall':>7s} {'prec':>6s} {'F1':>6s} {'F1<600px':>9s} {'F1>=600':>8s} {'sec/pg':>6s} {'agree':>6s}"
    typer.echo(header)
    for names in sets:
        def show(row):
            if verbose:
                typer.echo(f"  {row.document}/{row.sequence:04d} {row.width:5d}px F1={row.f1:.3f} {row.seconds:.2f}s"
                           + (f" agree={row.agreement:.2f}" if row.agreement is not None else ""))
        try:
            rep = bench.run(S, st, doc, names, limit, on_page=show)
        except RuntimeError as e:
            typer.secho(f"{'+'.join(names or [])}: {e}", fg="red"); continue
        m = rep.summary()
        fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
        typer.echo(f"{m['version']:34s} {m['pages']:5d} {fmt(m['recall']):>7s} {fmt(m['precision']):>6s} {fmt(m['f1']):>6s} "
                   f"{fmt(m['f1_under_600px']):>9s} {fmt(m['f1_600px_plus']):>8s} {m['seconds_per_page'] or 0:6.2f} "
                   f"{fmt(m['engine_agreement']):>6s}")


@app.command()
def doctor(costs: bool = False):
    """Show engines, cameras, credentials, configuration, database health and spend."""
    from .web.health import doctor_report
    rep = doctor_report(S, _store())
    typer.echo(f"root:        {rep['root']}")
    c = rep["config"]
    typer.echo(f"config:      model={c['model']} provider={c['provider']} prompt={c['prompt_version']} focus={c['focus']} "
               f"llm_workers={c['llm_workers']} log={c['log'] or 'off'}")
    typer.echo(f"ocr:         {rep['ocr_engine'] or 'NONE (macOS: pyobjc-framework-Vision; elsewhere pip install ocrmd[rapidocr])'}"
               f"   installed: {', '.join(rep['ocr_engines_installed']) or 'none'}")
    typer.echo(f"anthropic:   {'credentials found' if rep['credentials']['anthropic'] else 'no credentials (ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN in .env)'}")
    typer.echo(f"xai:         {'XAI_API_KEY set' if rep['credentials']['xai'] else 'no credentials'}")
    if not rep["dictionary"]:
        typer.secho("dictionary:  /usr/share/dict/words missing — orientation falls back to weaker evidence", fg="yellow")
    for cam in rep["cameras"]:
        typer.echo(f"camera {cam['index']}:    {cam['name']}")
        if cam.get("format"):
            typer.echo(f"             ocrmd will use {cam['format'][0]}x{cam['format'][1]}; focus modes: {', '.join(cam['focus_modes']) or 'none (fixed focus)'}")
            line = f"             now: focus={cam['focus']} exposure={cam['exposure']}"
            if cam.get("warning"):
                typer.secho(line + f"  <- {cam['warning']}", fg="yellow")
            else:
                typer.echo(line)
    if rep["camera_authorized"] is False:
        typer.secho("camera:      NOT authorized — System Settings > Privacy & Security > Camera, enable your terminal/IDE", fg="red")
    else:
        typer.echo(f"openable:    {[c['index'] for c in rep['cameras']] or 'none'}")
    typer.secho(f"database:    {rep['db_path']} integrity={rep['db_integrity']}", fg=None if rep["db_integrity"] == "ok" else "red")
    for d in rep["documents"]:
        typer.echo(f"  {d['slug']}: {d['pages']} page(s) " + ", ".join(f"{n} {k}" for k, n in sorted(d["counts"].items())))
    m = rep["missing_artifacts"]
    if m["raw"] or m["processed"]:
        typer.secho(f"artifacts:   {len(m['raw'])} page(s) without raw capture, {len(m['processed'])} without "
                    "processed image (run `ocrmd verify` to re-derive from raw)", fg="yellow")
    if rep["orphan_artifacts"]:
        typer.echo(f"artifacts:   {rep['orphan_artifacts']} orphan file(s) in processed/ocr/llm (safe to delete)")
    if rep["open_batches"]:
        typer.echo(f"batches:     {rep['open_batches']} open — `ocrmd verify --collect`")
    if costs:
        for r in rep["costs"]:
            typer.echo(f"  {r['provider']}/{r['model']}: {r['n']} calls, in={r['tokens_in']} out={r['tokens_out']} "
                       f"cached={r['cache_read']} avg {r['avg_latency_ms']} ms")


if __name__ == "__main__":
    app()
