from __future__ import annotations
import json, logging, os, time
from pathlib import Path
import typer, cv2, yaml
from .config import settings as S, setup_logging
from .store import Store, DONE_STATUSES
from . import library as library_mod, ocr, pipeline, render, review as review_mod, sinks as sinks_mod
from .export import FORMATS, export_document
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


def _build(st, slug: str) -> Path:
    """Build the document file and mirror document.yaml into the inventory (library.sync_document)."""
    out = render.build_document(S.documents / slug, slug)
    library_mod.sync_document(S, st, slug)
    return out


# =================================================================================================
@app.command()
def scan(doc: str = typer.Option(..., help="document slug"),
         camera: str = typer.Option(str(S.camera_index), help="index or device-name substring, e.g. c525"),
         no_llm: bool = typer.Option(False, "--no-llm", help="capture + OCR only; fill in the LLM pass later with `ocrmd verify`"),
         provider: str = S.provider, model: str = S.model, preview: bool = True,
         rotate: int | None = typer.Option(None, help="initial rotation 0/90/180/270 (overrides saved view)"),
         zoom: float | None = typer.Option(None, help="initial digital zoom, e.g. 2.0 (overrides saved view)"),
         focus: str = typer.Option(S.focus, help="continuous | locked | settle (lock after the first capture) | off"),
         admit: bool = typer.Option(S.admit_captures, "--admit/--no-admit",
                                    help="refuse automatic captures no OCR engine can read, instead of making "
                                         "a page of them (forced captures are always kept)"),
         focus_bracket: str = typer.Option(S.focus_bracket, "--focus-bracket",
                                           help="shoot each page at several lens positions: auto (only when the frame "
                                                "is not the sharpest the scene has shown) | always | off"),
         resolution: str | None = typer.Option(None, help="WxH, e.g. 1280x960; default is the largest mode"),
         title: str | None = typer.Option(None, help="document title (stored in document.yaml)"),
         tag: list[str] = typer.Option([], "--tag", help="tag for document.yaml; repeatable"),
         project: str | None = typer.Option(None, help="project name for document.yaml"),
         duration: float | None = typer.Option(None, help="stop after this many seconds (unattended runs)"),
         max_pages: int | None = typer.Option(None, help="stop after this many captures")):
    """Auto-capture pages from the webcam and process them incrementally.

    The shutter fires when the *scene* has settled — still, focused, unobstructed, and with the page
    outline holding its position — and the page is new and sharp. After a hand or a moved page it
    waits longer, so a page turn is not captured halfway through. A capture no OCR engine can read is
    refused rather than made into a page (--no-admit keeps the old behaviour; `ocrmd status` lists
    what was refused). A soft capture is then re-shot at several lens positions (--focus-bracket)
    and the extra exposures travel with the page, so text no single focus resolves is re-read from
    another one instead of guessed at. Preview keys: r rotate, +/- zoom, wasd/arrows pan, f/F focus,
    0 reset, space force capture, q quit. The view is saved per camera.
    """
    if focus_bracket not in ("auto", "always", "off"):
        _fail("--focus-bracket must be auto | always | off")
    st = _store()
    prov = _provider(no_llm, provider, model)
    res = tuple(int(x) for x in resolution.lower().split("x")) if resolution else S.resolution
    S.admit_captures = admit
    opts = ScanOptions(doc=doc, camera=camera, focus=focus, resolution=res, rotate=rotate, zoom=zoom,
                       duration=duration, max_pages=max_pages, bracket=focus_bracket, title=title,
                       tags=list(tag), project=project)
    typer.echo("space=force capture, q=quit")
    session = ScanSession(S, st, opts, prov, preview=preview, emit=emit)
    session.run()                                   # in this thread: the preview window needs the main thread
    if session.error:
        raise typer.Exit(1)


@app.command()
def annotate(doc: str = typer.Option(..., help="document slug"),
             camera: str = typer.Option(str(S.camera_index), help="index or device-name substring"),
             no_llm: bool = typer.Option(False, "--no-llm", help="skip handwriting; marks still sync free"),
             provider: str = S.provider, model: str = S.model,
             dry_run: bool = typer.Option(False, "--dry-run", help="report what would change, write nothing"),
             preview: bool = True, duration: float | None = typer.Option(None),
             max_pages: int | None = typer.Option(None, help="stop after this many looks")):
    """Watch already-scanned pages and fold what you write on the paper into the Markdown.

    Hold a scanned page under the camera and underline, highlight, strike through or write a margin
    note. The page is recognised, the new marks are diffed against the stored crop, and the page file
    is updated. Underlines and highlights cost nothing — the word positions are already on disk — so
    only handwriting reaches the model, and only when there is some.
    """
    from .capture import CameraLost, auto_shutter, camera_name, resolve_camera
    st = _store()
    prov = _provider(no_llm, provider, model)
    try:
        S.camera_index = resolve_camera(camera)
    except RuntimeError as e:
        _fail(str(e))
    name = camera_name(S.camera_index)
    typer.echo(f"watching {doc} on camera {S.camera_index} ({name}) — underline, highlight or write on "
               "the page; q to quit")
    seen: dict[str, int] = {}
    try:
        for fr in auto_shutter(S, preview=preview, on_status=lambda m: emit(m, "info"),
                               duration=duration, max_captures=max_pages, require_new=False):
            page = pipeline.identify_page(S, st, fr.image, doc)
            if page is None:
                emit("that page has not been scanned into this document yet", "warn")
                continue
            r = pipeline.sync_annotations(S, st, page, fr.image, prov, apply=not dry_run)
            level = "ok" if r.changes else ("warn" if r.reason else "info")
            emit(r.describe() + (f"  [{r.llm_calls} llm call(s)]" if r.llm_calls else ""), level)
            if r.changes:
                seen[r.page_id] = seen.get(r.page_id, 0) + len(r.changes)
    except KeyboardInterrupt:
        typer.echo("\nstopped")
    except (CameraLost, RuntimeError) as e:
        _fail(str(e))
    if seen:
        _build(st, doc)
    typer.echo(f"{sum(seen.values())} change(s) across {len(seen)} page(s)"
               + ("  (dry run: nothing written)" if dry_run else ""))


@app.command()
def etl(doc: str = typer.Option(..., help="document slug"),
        pipeline_file: Path = typer.Option(..., "--pipeline", "-p", help="the pipeline spec (YAML)"),
        out: Path | None = typer.Option(None, help="write here instead of documents/<doc>/extracted/"),
        limit: int | None = typer.Option(None, help="only the first N pages (try a pipeline cheaply)"),
        status: list[str] = typer.Option([], help="only pages with this status; repeatable"),
        dry_run: bool = typer.Option(False, "--dry-run", help="show the prompts and the call count, spend nothing"),
        no_cache: bool = typer.Option(False, "--no-cache", help="ignore cached answers and ask again"),
        export_docetl: Path | None = typer.Option(None, help="write a DocETL pipeline + dataset here instead of running"),
        no_llm: bool = typer.Option(False, "--no-llm", help="run only the operators that cost nothing"),
        source: str | None = typer.Option(None, help="pages | blocks | figures | extracted:<name> (overrides the spec)"),
        score: Path | None = typer.Option(None, help="a JSON list of expected rows; report precision/recall on --score-key"),
        score_key: str = typer.Option("name", help="the field --score matches rows on"),
        provider: str = S.provider, model: str = S.model):
    """Extract structured data from a scanned document with a declarative pipeline.

    map / filter / extract / reduce / resolve / split / unnest / sample over the document's pages,
    blocks or figures, each an LLM prompt and an output schema (DocETL's operator model — see
    ocrmd/etl.py). Every output row carries the page — and, from blocks, the rectangle — it came
    from; every operator's answers are cached and versioned by its prompt, so changing one operator
    re-runs one operator. A pipeline may write back into the page files, the page model or
    document.yaml instead of a spreadsheet (`output.type: frontmatter | layout | document`).

    PIPELINE may be a file, or the name of a bundled one: figures, references, glossary, receipts, cases.
    """
    from . import etl as etl_mod
    from .web.jobs import BUNDLED_PIPELINES
    st = _store()
    if not pipeline_file.exists():
        cand = BUNDLED_PIPELINES / (pipeline_file.name if pipeline_file.suffix == ".yaml" else f"{pipeline_file.name}.yaml")
        if cand.exists():
            pipeline_file = cand
    try:
        pipe = etl_mod.Pipeline.load(pipeline_file)
    except (ValueError, KeyError, yaml.YAMLError, OSError) as e:
        _fail(f"{pipeline_file}: {e}")
    try:
        rows = etl_mod.rows_for(S, st, doc, source or pipe.source, list(status) or pipe.status or None, limit)
    except (FileNotFoundError, ValueError) as e:
        _fail(str(e))
    if not rows:
        _fail(f"{doc} has no {source or pipe.source} with text to work on (has it been verified and built?)")
    typer.echo(f"{pipe.name}: {len(rows)} {source or pipe.source} row(s) -> {' -> '.join(pipe.steps)}")
    if export_docetl:
        prov_model = S.xai_model if provider == "xai" else model
        etl_mod.export_docetl(pipe, rows, export_docetl, prov_model)
        try:
            import docetl                                       # noqa: F401
            typer.echo(f"wrote {export_docetl}/pipeline.yaml and pages.json — run it with "
                       f"`docetl run {export_docetl}/pipeline.yaml`")
        except ImportError:
            typer.echo(f"wrote {export_docetl}/pipeline.yaml and pages.json — `pip install 'ocrmd[docetl]'`, "
                       f"then `docetl run {export_docetl}/pipeline.yaml`")
        return
    prov = _provider(no_llm or dry_run, provider, model)
    if prov is None and pipe.llm_steps and not dry_run and not no_llm:
        _fail(f"no credentials for {provider}; {len(pipe.llm_steps)} operator(s) need a model")
    runner = etl_mod.Runner(S, st, doc, prov, use_cache=not no_cache, dry_run=dry_run,
                            provider_for=lambda m: _provider(False, provider, m))
    try:
        result = runner.run(pipe, rows)
    except (ValueError, KeyError) as e:
        _fail(str(e))
    if dry_run:
        for name, prompt in runner.prompts[:3]:
            typer.echo(f"\n--- {name} ---\n{prompt[:1200]}")
        if pipe.writes_back:
            typer.echo(f"\nwould write {pipe.output.get('type')} fields {', '.join(pipe.output.get('fields', {}).values())}")
        typer.echo(f"\n{len(runner.prompts)} call(s) would be made ({runner.stats.cached} already cached); "
                   "nothing was sent")
        return
    for err in runner.stats.errors[:5]:
        emit(err, "error")
    found, quoted = etl_mod.locality(result)
    if quoted:
        emit(f"locality: {found}/{quoted} quoted span(s) found on the page", "ok" if found == quoted else "warn")
    if score is not None:
        try:
            truth = json.loads(score.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            _fail(f"{score}: {e}")
        emit(etl_mod.score(result, truth, score_key).describe(), "info")
    if pipe.writes_back and out is None:
        counts = etl_mod.write_back(S, st, doc, pipe, result, pipe.steps[-1] if pipe.steps else None)
        etl_mod.write(result, S.documents / doc / "extracted" / f"{pipe.name}.json", "json")
        emit(f"wrote {counts['fields']} field(s) into {counts['pages']} page(s)"
             + (f", {counts['blocks']} block(s)" if counts['blocks'] else "")
             + (", document.yaml" if counts['document'] else "") + f"  [{runner.stats.describe()}]",
             "warn" if runner.stats.errors else "ok")
        return
    path, kind, fields = etl_mod.output_path(S, doc, pipe, out)
    if kind in ("frontmatter", "layout", "document"):
        kind = "json"
    etl_mod.write(result, path, kind, fields)
    emit(f"{len(result)} row(s) -> {path}  [{runner.stats.describe()}]",
         "warn" if runner.stats.errors else "ok")


@app.command()
def read(doc: str = typer.Option(..., help="document slug"),
         page: int = typer.Option(..., help="page sequence"),
         region: str = typer.Option("0,0,1,1", help="x,y,w,h as fractions of the page (0-1)"),
         llm: bool = typer.Option(False, "--llm", help="also ask the model about this region (costs tokens)"),
         ask: str | None = typer.Option(None, help="what to ask about the region, instead of 'transcribe it'"),
         save: Path | None = typer.Option(None, help="write the enlarged crop here, as it was read"),
         provider: str = S.provider, model: str = S.model):
    """Read one region of a page harder than the page itself was read.

    The same thing the web UI does when you drag a box on a page image: the rectangle is cut out of the
    flat rendering, enlarged on its own terms and re-read. Nothing is stored — this answers "what does
    that word say", and the answer goes to you, not into the page file.
    """
    st = _store()
    row = st.page_by_sequence(doc, page)
    if row is None:
        _fail(f"{doc} has no page {page}")
    try:
        bbox = [float(v) for v in region.split(",")]
        if len(bbox) != 4:
            raise ValueError
    except ValueError:
        _fail(f"--region wants four numbers 'x,y,w,h' as fractions of the page, not {region!r}")
    prov = _provider(not llm, provider, model) if llm else None
    if llm and prov is None:
        _fail(f"no credentials for {provider}; nothing was sent to a model")
    try:
        out = pipeline.read_region(S, st, row, bbox, prov, ask)
    except (ValueError, FileNotFoundError) as e:
        _fail(str(e))
    if save:
        import cv2
        from .ocr import upscale_for_ocr
        crop = pipeline.region_crop(S, row["id"], out.bbox, flat=True)
        cv2.imwrite(str(save), upscale_for_ocr(crop, S.ocr_min_side, max_scale=pipeline.REGION_MAX_SCALE))
        typer.echo(f"crop written to {save}")
    typer.echo(f"{out.size[0]}x{out.size[1]}px enlarged {out.scale}x · {out.engine or 'no OCR engine'}"
               + (f" conf {out.confidence:.2f}" if out.text else ""))
    typer.echo(out.describe())
    if out.llm_error:
        emit(out.llm_error, "error")


@app.command()
def send(doc: str = typer.Option(..., help="document slug"),
         page: int = typer.Option(..., help="page sequence"),
         to: str = typer.Option("", "--to", help="destination name (see `ocrmd sinks`); may be omitted when only one is configured"),
         option: list[str] = typer.Option([], "--option", "-o", help="key=value for one of the receiver's options; repeatable"),
         region: str | None = typer.Option(None, help="x,y,w,h fractions: read this box first and send that reading"),
         dry_run: bool = typer.Option(False, "--dry-run", help="show the envelope that would be sent, send nothing")):
    """Hand a page you have read to a destination — a study platform, a knowledge base, anything that
    speaks the sink contract (ocrmd/sinks.py; `ocrmd sinks` lists what is configured).

    The text goes out, the photograph does not: the receiver gets ocrmd's *reconciled* reading, the
    page's provenance (document, page, page id, both agreement scores, a link back to the image) and
    the document's catalogue entry. Nothing here is written and no stage is marked — the same contract
    as `ocrmd read`. `--region` reads a box first and sends *that* reading, which is what to reach for
    when one item on a crowded page came out badly.
    """
    st = _store()
    row = st.page_by_sequence(doc, page)
    if row is None:
        _fail(f"{doc} has no page {page}")
    try:
        sink = sinks_mod.get(S, to)
        options = sinks_mod.parse_option_args(list(option))
    except sinks_mod.SinkError as e:
        _fail(str(e))
    bbox, text = None, None
    if region is not None:
        try:
            bbox = [float(v) for v in region.split(",")]
            if len(bbox) != 4:
                raise ValueError
        except ValueError:
            _fail(f"--region wants four numbers 'x,y,w,h' as fractions of the page, not {region!r}")
        # The region is read the way the web UI's drag-a-box reads it, model included: a selection is a
        # fraction of a page, so the same pixels buy several times the glyph height.
        prov = _provider(False, S.provider, S.model)
        try:
            out = pipeline.read_region(S, st, row, bbox, prov)
        except (ValueError, FileNotFoundError) as e:
            _fail(str(e))
        text = out.llm or out.text
        if not (text or "").strip():
            _fail("the region read produced no text — nothing to send")
        emit(f"region read: {out.size[0]}x{out.size[1]}px enlarged {out.scale}x · "
             f"{out.engine or 'no OCR engine'}{'  + model' if out.llm else ''}")
    try:
        if text is None:
            text, _fm = sinks_mod.page_text(S, doc, row)
        flags = sinks_mod.unresolved_flags(text)
        if flags and not dry_run:
            emit(f"{flags} unresolved disagreement(s) in this text — the marks travel with it, and so does the count", "warn")
        d = sinks_mod.send_page(S, doc, row, sink, options, text=text, bbox=bbox, dry_run=dry_run)
    except sinks_mod.SinkError as e:
        _fail(str(e))
    if dry_run:
        typer.echo(f"{d.summary} ({getattr(sink, 'url', '')})")
        typer.echo(json.dumps({k: v for k, v in d.raw.items() if k != "text"}, indent=2, default=str))
        typer.echo("--- text ---")
        typer.echo(d.raw.get("text", ""))
        return
    emit(d.describe(), "ok" if d.items else "warn")


@app.command()
def sinks(probe: bool = typer.Option(True, "--probe/--no-probe", help="ask each receiver for its descriptor")):
    """List the destinations a page can be sent to, and the options each one takes."""
    have = sinks_mod.registry(S)
    if not have:
        typer.echo("no destinations configured — add a [[sinks]] table to ocrmd.toml, or set OCRMD_SINKS "
                   "(see ocrmd.example.toml)")
        return
    for sink in have:
        d = sink.describe(probe=probe) if isinstance(sink, sinks_mod.HttpSink) else sink.describe()
        state = "" if not probe else ("  reachable" if d.get("reachable") else f"  {d.get('error') or 'unreachable'}")
        typer.secho(f"{sink.name}  {getattr(sink, 'url', '')}{state}", fg=None if d.get("reachable", True) else "yellow")
        if d.get("description"):
            typer.echo(f"    {d['description']}")
        for o in d.get("options") or []:
            choices = f" ({', '.join(map(str, o['choices']))})" if o.get("choices") else ""
            default = f" [default {o['default']}]" if o.get("default") not in (None, "") else ""
            typer.echo(f"    --option {o['key']}=…  {o.get('type', 'string')}{choices}{default}  {o.get('help') or o.get('label') or ''}")
        if d.get("defaults"):
            typer.echo(f"    defaults from ocrmd.toml: {json.dumps(d['defaults'])}")


@app.command()
def library(as_json: bool = typer.Option(False, "--json", help="machine-readable"),
            all_docs: bool = typer.Option(False, "--all", help="also documents with no pages yet")):
    """The inventory: every document scanned into this root, what is known about it, how far along it is."""
    st = _store()
    docs = [d for d in library_mod.catalog(S, st) if all_docs or d.pages]
    if as_json:
        typer.echo(json.dumps([d.to_dict() for d in docs], indent=1, default=str))
        return
    if not docs:
        typer.echo("nothing scanned yet — `ocrmd scan` or `ocrmd ingest` to begin")
        return
    typer.echo(f"{'slug':<22} {'title':<38} {'pages':>13} {'done':>9}")
    for d in docs:
        colour = "green" if d.pages and d.done == d.pages else ("yellow" if d.needs_attention else None)
        typer.secho(d.describe(), fg=colour)
        extras = []
        if d.page_labels:
            extras.append(f"printed {d.page_labels}")
        if d.tags:
            extras.append("tags " + ", ".join(d.tags))
        if d.exports:
            extras.append("exported " + ", ".join(sorted(d.exports)))
        if d.last_captured_at:
            extras.append(f"last capture {d.last_captured_at[:16]}")
        if extras:
            typer.echo("    " + " · ".join(extras))
    typer.echo(f"{len(docs)} document(s), {sum(d.pages for d in docs)} page(s), root {S.root}")


@app.command()
def search(query: str = typer.Argument(..., help="words to find; quote a phrase; word* for a prefix"),
           doc: str | None = typer.Option(None, help="only this document"),
           limit: int = typer.Option(20, help="most hits to show"),
           as_json: bool = typer.Option(False, "--json")):
    """Full-text search across every page ever scanned, or one document. Each hit names its page."""
    st = _store()
    hits = library_mod.search(S, st, query, doc, limit)
    if as_json:
        typer.echo(json.dumps([h.to_dict() for h in hits], indent=1, ensure_ascii=False))
        return
    if not hits:
        typer.echo(f"no page matches {query!r}  ({'fts5' if st.fts else 'substring scan — this SQLite has no FTS5'})")
        return
    for h in hits:
        where = f"{h.document}/{h.sequence:04d}" + (f" p.{h.page_label}" if h.page_label else "")
        typer.secho(f"{where:<28} {h.status or '':<10}", fg=STATUS_COLOR.get(h.status or "", None), nl=False)
        typer.echo(f" {h.snippet[:160]}")
    typer.echo(f"{len(hits)} hit(s)")


@app.command()
def export(doc: str | None = typer.Option(None, help="document slug; default every document"),
           fmt: str = typer.Option("md", "--format", "-f", help="md | txt | json | jsonl"),
           out: Path | None = typer.Option(None, help="write here instead of <root>/export/<slug>/<slug>.<fmt> (one document only)")):
    """Write a document in another shape: the built Markdown, plain text, one JSON object, or one JSON
    object per page. Every shape keeps the page ids, so a consumer can always come back to the photograph."""
    st = _store()
    if fmt not in FORMATS:
        _fail(f"--format must be one of {', '.join(FORMATS)}")
    slugs = [doc] if doc else st.documents()
    if out is not None and len(slugs) != 1:
        _fail("--out needs exactly one --doc")
    for slug in slugs:
        if slug not in st.documents():
            _fail(f"no such document: {slug}")
        path = export_document(S, st, slug, fmt, out)
        typer.echo(f"{slug} -> {path}  ({path.stat().st_size:,} B)")


@app.command()
def index(doc: str | None = typer.Option(None, help="document slug; default every document"),
          force: bool = typer.Option(False, "--force", help="re-read every page file, not just the changed ones")):
    """Rebuild the search index from the page files. Search does this on its own for pages whose file
    changed; this is for after restoring a backup, or to be sure."""
    st = _store()
    n = library_mod.reindex(S, st, doc, force)
    typer.echo(f"{n} page(s) indexed ({'fts5' if st.fts else 'substring scan: this SQLite has no FTS5'}); "
               f"{st.indexed_count()} in the index")


@app.command()
def reconstruct(doc: str = typer.Option(..., help="document slug"),
                page: int | None = typer.Option(None, help="one page sequence; default every page"),
                out: Path = typer.Option(Path("reconstructed"), help="directory for the .svg/.tex/.png"),
                pdf: bool = typer.Option(False, "--pdf", help="also compile the LaTeX (needs a TeX engine)"),
                review: bool = typer.Option(False, "--review", help="ask the model what the rebuild lost (costs tokens)"),
                provider: str = S.provider, model: str = S.model):
    """Rebuild pages from their Markdown as SVG + LaTeX, and score them against the original crop.

    The round trip is the check: anything the transcription lost — a heading flattened, a table turned
    to prose, a figure nobody mentioned — shows up as a difference between the rebuilt page and the
    photograph it came from. `fidelity` is that difference as one number, 0 to 1.
    """
    from . import reconstruct as rc_mod
    st = _store()
    pages = [p for p in st.pages(doc) if page is None or p["sequence"] == page]
    if not pages:
        _fail(f"no pages in {doc}" + (f" at sequence {page}" if page else ""))
    if not rc_mod.backends():
        typer.secho("no SVG rasteriser found (inkscape, rsvg-convert or Chrome) — writing files, "
                    "but fidelity cannot be scored", fg="yellow")
    out.mkdir(parents=True, exist_ok=True)
    prov = _provider(not review, provider, model) if review else None
    scores = []
    for p in pages:
        pid = p["id"]
        img = cv2.imread(str(S.processed / f"{pid}.png"))
        md_path = render.page_path(S.documents / doc, p["sequence"])
        if img is None or not md_path.exists():
            typer.secho(f"  {p['sequence']:04d}: missing artifacts, skipped", fg="yellow"); continue
        try:
            lay = pipeline.page_layout(S, st, p)
        except Exception as e:
            typer.secho(f"  {p['sequence']:04d}: no layout ({e}), skipped", fg="yellow"); continue
        fm, _body = render.read_page(md_path)
        r = rc_mod.rebuild(img, lay, title=(fm.get("context") or {}).get("chapter"))
        stem = out / f"{p['sequence']:04d}"
        stem.with_suffix(".svg").write_text(r.svg, encoding="utf-8")
        stem.with_suffix(".tex").write_text(r.latex, encoding="utf-8")
        if r.png is not None:
            cv2.imwrite(str(stem.with_suffix(".png")), r.png)
        if pdf:
            made = rc_mod.compile_pdf(r.latex, stem.with_suffix(".pdf"))
            if made is None:
                typer.secho("    no TeX engine; .tex written but not compiled", fg="yellow")
        if r.fidelity is not None:
            scores.append(r.fidelity)
        colour = ("green" if (r.fidelity or 0) >= 0.8 else "yellow" if (r.fidelity or 0) >= 0.6 else "red")
        typer.secho(f"  {p['sequence']:04d}: {r.describe()}", fg=colour if r.fidelity is not None else None)
        if review and prov is not None and r.png is not None:
            _review_rebuild(prov, img, r.png, p["sequence"])
    if scores:
        typer.echo(f"{len(scores)} page(s) rebuilt into {out}; mean fidelity "
                   f"{sum(scores) / len(scores):.2f}")


def _review_rebuild(prov, original, rebuilt, seq: int) -> None:
    """Ask the model what the rebuild lost. A pixel score says *how much* differs, not *what*."""
    from .llm import FocusViews
    note = ("The first image is a photograph of a book page. The second is a reconstruction of it "
            "built from the transcription this tool produced. List, briefly, what the reconstruction "
            "got WRONG about the page — text that is missing, headings flattened into body text, "
            "tables turned into prose, figures absent. Do not list styling or font differences; only "
            "content and structure. If it is faithful, say so in one line.")
    try:
        res = prov.transcribe(original, "", None, FocusViews(images=[rebuilt], note=note))
        msg = (res.transcription.notes or res.transcription.markdown or "").strip()
        if msg:
            typer.secho(f"      review: {msg[:300]}", fg="magenta")
    except Exception as e:
        typer.secho(f"      review unavailable: {e}", fg="yellow")


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
    def landed(name: str, res: pipeline.Ingested) -> None:
        if not res.pages:
            typer.secho(f"  duplicate: {name}", fg="yellow"); return
        typer.echo(f"{name} -> {len(res.pages)} page(s)")
        for pid in res.pages:
            _process(st, pid, prov, doc, refresh=refresh)
    if f.suffix.lower() == ".pdf":
        try:
            pipeline.ingest_pdf(S, st, f, doc, on_page=landed)
        except RuntimeError as e:
            _fail(str(e))
        return
    img = cv2.imread(str(f))
    if img is None:
        typer.secho(f"  unreadable: {f.name}", fg="red"); return
    landed(f.name, pipeline.ingest_frame(S, st, img, doc, f"file:{f.name}"))


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
                _build(st, doc)
            time.sleep(2.0)
    except KeyboardInterrupt:
        typer.echo("\nstopped watching")
    _build(st, doc)
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
        _build(st, slug)
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
        _build(st, slug)
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
        _build(st, slug)
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
        _build(st, doc)
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
        rejected = st.rejected_captures(slug)
        if rejected:
            typer.secho(f"  {len(rejected)} capture(s) refused as unreadable — the frames are kept, "
                        "no pages were made:", fg="yellow")
            for r in rejected[:10]:
                typer.echo(f"    raw/{r['sha256'][:12]}…jpg  {r['captured_at'][11:19]}  {r['rejected'][:60]}")
            if len(rejected) > 10:
                typer.echo(f"    ... and {len(rejected) - 10} more")
            typer.echo(f"    force one in with: ocrmd ingest raw/<sha>.jpg --doc {slug}")


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
    _build(st, doc)
    typer.echo(f"{doc}: {out.summary()} (pages {', '.join(f'{n:04d}' for n in out.pages)})")


@app.command()
def build(doc: str | None = None):
    """Regenerate document markdown + nav links from page files (and document.yaml)."""
    st = _store()
    for slug in ([doc] if doc else st.documents()):
        render.ensure_document_meta(S.documents / slug, slug)
        typer.echo(_build(st, slug))


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
               f"bracket={c['focus_bracket']} llm_workers={c['llm_workers']} log={c['log'] or 'off'}")
    typer.echo(f"ocr:         {rep['ocr_engine'] or 'NONE (macOS: pyobjc-framework-Vision; elsewhere pip install ocrmd[rapidocr])'}"
               f"   installed: {', '.join(rep['ocr_engines_installed']) or 'none'}")
    typer.echo(f"anthropic:   {'credentials found' if rep['credentials']['anthropic'] else 'no credentials (ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN in .env)'}")
    typer.echo(f"xai:         {'XAI_API_KEY set' if rep['credentials']['xai'] else 'no credentials'}")
    if not rep["dictionary"]:
        typer.secho("dictionary:  /usr/share/dict/words missing — orientation falls back to weaker evidence", fg="yellow")
    for cam in rep["cameras"]:
        typer.echo(f"camera {cam['index']}:    {cam['name']}")
        if cam.get("format"):
            typer.echo(f"             ocrmd will use {cam['format'][0]}x{cam['format'][1]}; focus modes: {', '.join(cam['focus_modes']) or 'none (fixed focus)'}"
                       + ("; focus bracket available" if cam.get("lens_position") else "; no custom lens position (focus bracket unavailable)"))
            if cam.get("best_sharpness"):
                typer.echo(f"             sharpest this rig has focused: {cam['best_sharpness']:.0f} "
                           "— anything far below that during a scan is treated as out of focus")
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
    lib = rep["library"]
    typer.echo(f"library:     {lib['documents']} document(s), {lib['pages']} page(s), {lib['indexed']} indexed for search "
               f"({'fts5' if lib['fts'] else 'substring scan'}); exports under {lib['export_dir']}")
    for d in rep["documents"]:
        typer.echo(f"  {d['slug']}: {d['pages']} page(s) " + ", ".join(f"{n} {k}" for k, n in sorted(d["counts"].items())))
    typer.echo("sinks:       " + (", ".join(f"{x['name']} ({x['url']})" for x in rep["sinks"]) if rep["sinks"]
                                 else "none configured (see ocrmd.example.toml [[sinks]])"))
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
