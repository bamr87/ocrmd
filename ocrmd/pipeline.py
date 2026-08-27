"""Stage runner: ingest a frame, then run missing stages for a page.

Stages: preprocess (at ingest — it decides page identity) -> ocr -> llm -> reconcile+render. Each is
cached on disk and recorded in SQLite, so `run_page` on an already-processed page is a no-op render,
and a page scanned with --no-llm is completed later by `verify` without redoing OCR.
"""
from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field
from pathlib import Path
import cv2, numpy as np
from .config import Settings
from .store import Store
from .fingerprint import sha256_bytes, dhash, similarity, detail_similarity
from . import preprocess, ocr, reconcile as rc, render, review
from .reconcile import token_overlap, _content_tokens
from .llm import Provider, LLMResult, LLMError, PageHint, Transcription

log = logging.getLogger("ocrmd.pipeline")
PRE_VER = preprocess.PIPELINE_VERSION


@dataclass
class PageResult:
    path: Path
    status: str
    agreement: float | None = None
    overlap: float | None = None
    advice: str | None = None
    continuity: str | None = None
    illegible_reason: str | None = None
    preserved: list[str] = field(default_factory=list)

    def detail(self) -> str:
        d = f" agreement={self.agreement:.2f} overlap={self.overlap or 0:.2f}" if self.agreement is not None else ""
        if self.illegible_reason:
            d += f"  ({self.illegible_reason[:50]})"
        return d


# --- ingest ------------------------------------------------------------------------------------
def ingest_frame(s: Settings, st: Store, img: np.ndarray, slug: str, source: str) -> list[str]:
    """Store raw frame, dedup, preprocess into 1..2 pages, register them. Returns page ids ([] if duplicate)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    raw = buf.tobytes()
    sha = sha256_bytes(raw)
    ph = dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if st.find_capture(sha):
        return []
    parts = preprocess.process(img, s.min_page_width)
    ocr_results: list[ocr.OcrResult | None] = [None] * len(parts)
    dup = find_duplicate(s, st, img, ph, slug, parts, ocr_results)
    if dup:
        st.add_capture(sha, ph, None, dup, source)
        log.info("duplicate capture %s of page %s", sha[:8], dup)
        return []
    (s.raw / f"{sha}.jpg").write_bytes(raw)
    st.ensure_document(slug)
    ids = derive_pages(s, st, img, slug, sha, ph, parts, ocr_results)
    st.add_capture(sha, ph, ids[0] if ids else None, None, source)
    return ids


def text_duplicate(a: str, b: str, threshold: float) -> bool | None:
    """Do two OCR readings describe the same page? None when there is too little text to say."""
    wa, wb = a.split(), b.split()
    if len(wa) < 10 or len(wb) < 10 or not _content_tokens(wa) or not _content_tokens(wb):
        return None
    ratio = len(wa) / len(wb)
    return token_overlap(wa, wb) >= threshold and 0.7 <= ratio <= 1.4


def find_duplicate(s: Settings, st: Store, img: np.ndarray, ph: str, slug: str, parts, ocr_results) -> str | None:
    """A page this frame re-captures, if any.

    dHash prefilter, then frame correlation against the candidate's raw capture, then — because a
    dHash and a correlation cannot tell two pages of one book apart when both are mostly blank —
    the OCR text of the new crops against the candidate's OCR text. OCR runs here only when a
    candidate exists, and the result is kept as the new page's OCR stage if the page is new.
    """
    cands = []
    for cand in st.near_duplicates(ph, slug, s.near_duplicate_distance):
        raw = s.raw / f"{cand['raw_sha256']}.jpg"
        other = cv2.imread(str(raw)) if raw.exists() else None
        if other is None or other.shape[:2] != img.shape[:2]:
            continue
        sim = similarity(img, other)
        log.debug("dedup candidate %s similarity=%.3f", cand["id"], sim)
        if sim >= s.near_duplicate_similarity:
            cands.append((cand, other))
    if not cands or not parts:
        return None
    if ocr.available_engines():
        rec = ocr.recognizer(s)
        for i, (color, _binar, _) in enumerate(parts):
            if ocr_results[i] is None:
                ocr_results[i] = rec.run(color)
    for cand, other in cands:
        cand_text = None
        try:                                                # the candidate's OCR, computed now if it has none yet
            cand_text = run_ocr(s, st, st.page(cand["id"])).text
        except Exception as e:                              # missing crop, no engine: image fallback below
            log.debug("dedup: no OCR text for candidate %s (%s)", cand["id"], e)
        verdicts = []
        for res in ocr_results:
            v = text_duplicate(res.text, cand_text, s.near_duplicate_text) if (res is not None and cand_text is not None) else None
            if v is None:                                   # no text to compare: fall back to the image
                v = detail_similarity(img, other) >= s.near_duplicate_detail
            verdicts.append(v)
        if verdicts and any(verdicts):
            return cand["id"]
    return None


def derive_pages(s: Settings, st: Store, img: np.ndarray, slug: str, sha: str, ph: str,
                 parts=None, ocr_results=None) -> list[str]:
    """Preprocess one capture into 1..2 pages. The page id is the hash of the *processed* image,
    so changing the preprocess pipeline necessarily changes page identity (see `ocrmd reprocess`)."""
    ids = []
    parts = parts if parts is not None else preprocess.process(img, s.min_page_width)
    for i, (color, binar, meta) in enumerate(parts):
        ok, pbuf = cv2.imencode(".png", color)
        pid = sha256_bytes(pbuf.tobytes())[:16]
        if st.page(pid):
            log.info("page %s already exists (identical crop); skipping", pid)
            continue
        (s.processed / f"{pid}.png").write_bytes(pbuf.tobytes())
        cv2.imwrite(str(s.processed / f"{pid}.bin.png"), binar)
        (s.processed / f"{pid}.json").write_text(json.dumps(meta))
        st.add_page(pid, slug, sha, ph, meta.get("part"), meta.get("page_width_px"), meta.get("advice"))
        st.mark_stage(pid, "preprocess", PRE_VER)
        res = ocr_results[i] if ocr_results else None
        if res is not None:                                  # OCR already done for dedup: keep it
            (s.ocr_dir / f"{pid}.json").write_text(res.model_dump_json(indent=1))
            st.mark_stage(pid, "ocr", ocr.recognizer(s).version)
        ids.append(pid)
    return ids


def _drop_page_artifacts(s: Settings, pid: str) -> int:
    n = 0
    for f in (list(s.processed.glob(f"{pid}.*")) + list(s.ocr_dir.glob(f"{pid}.json"))
              + list(s.llm_dir.glob(f"{pid}.*.json"))):
        f.unlink(); n += 1
    return n


def reprocess_document(s: Settings, st: Store, slug: str) -> tuple[list[str], int]:
    """Re-derive every page of a document from its raw captures using the current pipeline.
    Returns (new page ids, artifacts removed). Cached OCR/LLM results are dropped: they belong
    to the old crops and the new pages have different ids."""
    raws = st.raw_captures(slug)
    removed = 0
    for page in st.pages(slug):
        removed += _drop_page_artifacts(s, page["id"])
        st.delete_page(page["id"])
    pages_dir = s.documents / slug / "pages"
    if pages_dir.exists():
        for f in pages_dir.glob("*.md"):
            f.unlink()
    ids: list[str] = []
    for r in raws:
        raw = s.raw / f"{r['raw_sha256']}.jpg"
        if not raw.exists():
            log.warning("raw capture missing for %s; skipped", r["raw_sha256"][:12])
            continue
        img = cv2.imread(str(raw))
        new = derive_pages(s, st, img, slug, r["raw_sha256"], r["phash"])
        if new:
            st.set_capture_page(r["raw_sha256"], new[0])
        ids += new
    return ids, removed


# --- stages ------------------------------------------------------------------------------------
def run_preprocess(s: Settings, st: Store, page) -> dict:
    """Preprocessing happens at ingest (it decides page identity). Here we only re-derive artifacts
    that went missing — e.g. the processed/ directory was cleaned but the database kept the page."""
    pid = page["id"]
    meta_p = s.processed / f"{pid}.json"
    if meta_p.exists() and (s.processed / f"{pid}.png").exists():
        return json.loads(meta_p.read_text())
    raw = s.raw / f"{page['raw_sha256']}.jpg"
    if not raw.exists():
        raise FileNotFoundError(f"raw capture missing for page {pid}: {raw}")
    parts = preprocess.process(cv2.imread(str(raw)), s.min_page_width)
    want = page["part"]
    match = next((p for p in parts if p[2].get("part") == want), parts[0])
    color, binar, meta = match
    cv2.imwrite(str(s.processed / f"{pid}.png"), color)
    cv2.imwrite(str(s.processed / f"{pid}.bin.png"), binar)
    meta_p.write_text(json.dumps(meta))
    st.mark_stage(pid, "preprocess", PRE_VER)
    return meta


def ocr_version(s: Settings) -> str | None:
    """The current OCR stage version, or None when no engine is installed."""
    try:
        return ocr.recognizer(s).version
    except RuntimeError:
        return None


def ocr_stale(s: Settings, st: Store, page, version: str | None = None) -> bool:
    """The page's OCR was made by a different engine set / OCR version than the current one."""
    version = version or ocr_version(s)
    return version is not None and not st.stage_done(page["id"], "ocr", version)


def run_ocr(s: Settings, st: Store, page) -> ocr.OcrResult:
    """Cached per (page, engine set, OCR version): a new engine or a merge change re-reads the page the
    next time it is processed — local and free, unlike the LLM stage."""
    pid = page["id"]
    out = s.ocr_dir / f"{pid}.json"
    rec = ocr.recognizer(s)
    if st.stage_done(pid, "ocr", rec.version) and out.exists():
        return ocr.OcrResult.model_validate_json(out.read_text())
    img = cv2.imread(str(s.processed / f"{pid}.png"))
    if img is None:
        raise FileNotFoundError(f"processed image missing for page {pid}")
    res = rec.run(img)
    out.write_text(res.model_dump_json(indent=1))
    st.mark_stage(pid, "ocr", rec.version)
    log.debug("ocr %s: %s %.2fs agreement=%s", pid, res.engine, sum(res.timings.values()), res.agreement)
    return res


def page_hint(s: Settings, st: Store, page, slug: str) -> PageHint:
    """Where this page sits: half of a spread, and what the previous page established."""
    meta = render.load_document_meta(s.documents / slug)
    hint = PageHint(part=page["part"], document_title=meta.get("title"), kind=meta.get("kind"))
    prev = st.previous_page(slug, page["sequence"])
    if prev:
        hint.prev_page_label = prev["page_label"]
        pp = render.page_path(s.documents / slug, prev["sequence"])
        if pp.exists():
            try:
                ctx = (render.read_page(pp)[0].get("context") or {})
                hint.prev_chapter, hint.prev_section = ctx.get("chapter"), ctx.get("section")
                hint.prev_running_head = ctx.get("running_head")
            except Exception:  # a hand-edited page with broken YAML must not block this one
                log.debug("could not read previous page %s for context", pp)
    return hint


def llm_artifact(s: Settings, pid: str, provider_name: str) -> Path:
    return s.llm_dir / f"{pid}.{provider_name}.json"


def llm_version(s: Settings, model: str | None) -> str:
    """Stage version for the LLM pass: the prompt *and* the model, so `verify --refresh` re-runs pages
    made by an older prompt or a different model, and nothing else does."""
    return f"{s.prompt_version}:{model}" if model else s.prompt_version


def llm_cached(s: Settings, st: Store, page, provider_name: str, refresh: bool, model: str | None = None) -> LLMResult | None:
    """A completed LLM stage is reused as-is; only `refresh` insists on the current prompt+model version
    (re-running costs tokens, so a version bump must never silently trigger it)."""
    out = llm_artifact(s, page["id"], provider_name)
    if st.stage_done(page["id"], "llm", llm_version(s, model) if refresh else None) and out.exists():
        return LLMResult.model_validate_json(out.read_text())
    return None


def store_llm(s: Settings, st: Store, pid: str, res: LLMResult, model: str | None = None) -> None:
    llm_artifact(s, pid, res.provider).write_text(res.model_dump_json(indent=1))
    st.log_llm(pid, res.provider, res.model, res.request_id, res.usage, res.latency_ms)
    st.mark_stage(pid, "llm", llm_version(s, model or res.model))


def run_llm(s: Settings, st: Store, page, provider: Provider, ocr_res: ocr.OcrResult,
            hint: PageHint | None = None, refresh: bool = False) -> LLMResult:
    pid = page["id"]
    cached = llm_cached(s, st, page, provider.name, refresh, provider.model)
    if cached is not None:
        return cached
    img = cv2.imread(str(s.processed / f"{pid}.png"))
    try:
        res = provider.transcribe(img, ocr_res.text, hint)
    except LLMError as e:  # recorded, page stays unverified
        st.mark_stage(pid, "llm", llm_version(s, provider.model), error=str(e)[:500])
        raise
    store_llm(s, st, pid, res, provider.model)
    return res


def _ocr_only_markdown(res: ocr.OcrResult) -> str:
    """Without an LLM: paragraphs from OCR line boxes (vertical gap > 1.6x median line height),
    de-hyphenated. Headings/lists are not inferred."""
    blocks = [b for b in res.blocks if b.text.strip()]
    if not blocks:
        return ""
    heights = sorted(b.bbox[3] for b in blocks)
    med_h = heights[len(heights) // 2] or 0.01
    paras, buf, prev_bottom = [], "", None
    for b in blocks:
        top = b.bbox[1]
        if prev_bottom is not None and top - prev_bottom > 0.6 * med_h:
            paras.append(buf); buf = ""
        line = b.text.strip()
        if buf.endswith("-") and line[:1].islower():
            buf = buf[:-1] + line
        else:
            buf = (buf + " " + line).strip()
        prev_bottom = top + b.bbox[3]
    if buf:
        paras.append(buf)
    return "\n\n".join(paras)


def continuity_advice(st: Store, slug: str, sequence: int, label: str | None) -> str | None:
    """Printed page numbers on a textbook run in order; a jump or a repeat means a page was missed
    or captured twice, and that is far cheaper to hear now than to discover on review."""
    if not label:
        return None
    prev = st.previous_page(slug, sequence)
    if not prev or not prev["page_label"]:
        return None
    a, b = render._roman_or_int(prev["page_label"]), render._roman_or_int(label)
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    if b == a:
        return f"printed page {label} repeats the previous page — duplicate capture?"
    if 1 < b - a <= 20:
        return f"printed page jumped {prev['page_label']} → {label} — {b - a - 1} page(s) may have been missed"
    if b < a:
        return f"printed page went backwards {prev['page_label']} → {label}"
    return None


def run_page(s: Settings, st: Store, page, provider: Provider | None, slug: str, on_stage=None,
             refresh: bool = False) -> PageResult:
    note = on_stage or (lambda _stage: None)
    note("preprocess")
    meta = run_preprocess(s, st, page)
    note("ocr")
    ocr_res = run_ocr(s, st, page)
    llm_res: LLMResult | None = None
    if provider is not None:
        note("llm")
        llm_res = run_llm(s, st, page, provider, ocr_res, page_hint(s, st, page, slug), refresh)
    else:
        llm_res = llm_cached(s, st, page, s.provider, refresh=False)   # keep an earlier LLM pass if there is one
    note("render")
    return render_page(s, st, page, slug, meta, ocr_res, llm_res)


def render_page(s: Settings, st: Store, page, slug: str, meta: dict, ocr_res: ocr.OcrResult,
                llm_res: LLMResult | None) -> PageResult:
    pid = page["id"]
    doc_dir = s.documents / slug
    source = st.capture_source(page["raw_sha256"]) or ""
    fm = {
        "ocrmd": 1, "id": pid, "document": slug, "sequence": page["sequence"],
        "captured_at": page["captured_at"], "kind": "other", "page_label": None, "language": None,
        "context": {"title": None, "author": [], "chapter": None, "section": None, "running_head": None,
                    "citation_key": None, "tags": [], "project": None},
        "references": {"cites": [], "figures": [], "footnotes": 0, "related": []},
        "source": {"type": "webcam" if source.startswith("webcam") else "file",
                   "device": source.split(":", 1)[1] if ":" in source else None,
                   "raw": f"raw/{page['raw_sha256']}.jpg",
                   "sha256": page["raw_sha256"], "phash": page["phash"], "part": page["part"]},
        "processing": dict(meta),
        "ocr": {"engine": ocr_res.engine, "confidence": round(ocr_res.mean_confidence, 3),
                "agreement": round(ocr_res.agreement, 3) if ocr_res.agreement is not None else None,
                "artifact": f"ocr/{pid}.json"},
        "llm": None,
        "verification": {"status": "unverified", "agreement": None, "overlap": None, "cer": None,
                         "flagged_spans": 0, "arbiter": None, "reviewed_by": None},
    }
    result = PageResult(path=render.page_path(doc_dir, page["sequence"]), status="unverified",
                        advice=meta.get("advice"))
    if llm_res is None:
        body = _ocr_only_markdown(ocr_res)
        status = "unverified"
        err = st.stage_error(pid, "llm")
        if err:
            fm["verification"]["llm_error"] = err[:200]
    else:
        tr: Transcription = llm_res.transcription
        rec = rc.reconcile(ocr_res.text, ocr_res.words, tr.markdown, s.ocr_trust, s.ocr_distrust,
                           s.flag_threshold, s.suspect_threshold)
        body, status = rec.markdown, rec.status
        fm.update(kind=tr.kind, page_label=tr.page_number, language=tr.language)
        fm["context"].update(title=tr.title, chapter=tr.context.chapter, section=tr.context.section,
                             running_head=tr.context.running_head)
        fm["references"].update(cites=tr.context.cites, figures=tr.context.figures)
        fm["llm"] = {"provider": llm_res.provider, "model": llm_res.model, "prompt": s.prompt_version,
                     "request_id": llm_res.request_id,
                     "tokens": {"in": llm_res.usage.get("input_tokens"), "out": llm_res.usage.get("output_tokens"),
                                "cache_read": llm_res.usage.get("cache_read_input_tokens")}}
        fm["verification"].update(status=status, agreement=rec.agreement, overlap=rec.overlap, cer=rec.cer,
                                  flagged_spans=sum(1 for f in rec.flags if f.chosen is None))
        result.agreement, result.overlap = rec.agreement, rec.overlap
        if tr.uncertain:
            fm["verification"]["llm_uncertain"] = [u.text for u in tr.uncertain]
        if tr.notes:
            fm["verification"]["llm_notes"] = tr.notes
        if not tr.legible:
            fm["verification"]["illegible_reason"] = result.illegible_reason = tr.illegible_reason
            if len(tr.markdown.strip()) < 40 and status != "blank":
                # the model honestly declined and OCR only produced noise: the page needs re-shooting,
                # which is a different problem from two sources disagreeing about readable text
                status = "illegible"
            elif status not in ("blank", "suspect"):
                # hard to read, but the two sources still agree: keep the usable transcription, flag it
                status = "flagged"
            fm["verification"]["status"] = status
        result.continuity = continuity_advice(st, slug, page["sequence"], tr.page_number)
        if result.continuity:
            fm["processing"]["continuity"] = result.continuity
    fm["history"] = st.stage_history(pid)

    # human edits: if the file changed since the pipeline last wrote it, keep what the human wrote
    if result.path.exists():
        try:
            old_fm, old_body = render.read_page(result.path)
        except Exception as e:
            log.warning("page %s unreadable (%s); overwriting", result.path.name, e)
            old_fm, old_body = None, ""
        recorded = page["written_fingerprint"] if "written_fingerprint" in page.keys() else None
        reviewer = (old_fm or {}).get("verification", {}).get("reviewed_by") if old_fm else None
        if old_fm and (reviewer or recorded is None or review.fingerprint(old_fm, old_body) != recorded):
            fm, body, kept = review.merge_human_edits(fm, body, old_fm, old_body)
            if kept:
                result.preserved = kept
                status = fm["verification"]["status"]
                log.info("page %s: kept human edits %s", pid, kept)
            if reviewer and not page["reviewed_by"]:
                who = reviewer.get("who") if isinstance(reviewer, dict) else str(reviewer)
                st.set_reviewed(pid, who, "reviewed")
    result.status = status
    fm["verification"]["status"] = status
    render.write_page(doc_dir, page["sequence"], fm, body)
    st.set_fingerprint(pid, review.fingerprint(fm, body))
    st.set_status(pid, status, fm["page_label"])
    st.set_page_meta(pid, meta.get("page_width_px"), meta.get("advice"))
    return result


# --- batches -----------------------------------------------------------------------------------
def submit_batch(s: Settings, st: Store, pages, provider, slug_of) -> tuple[str | None, list[str]]:
    """Queue the LLM stage for `pages` through the provider's batch API. Returns (batch id, page ids)."""
    items = []
    for page in pages:
        pid = page["id"]
        run_preprocess(s, st, page)
        ocr_res = run_ocr(s, st, page)
        img = cv2.imread(str(s.processed / f"{pid}.png"))
        items.append((pid, img, ocr_res.text, page_hint(s, st, page, slug_of(page))))
    if not items:
        return None, []
    batch_id = provider.batch_submit(items)
    st.add_batch(batch_id, provider.name, provider.model, s.prompt_version, [i[0] for i in items])
    log.info("submitted batch %s with %d pages", batch_id, len(items))
    return batch_id, [i[0] for i in items]


def collect_batch(s: Settings, st: Store, batch_id: str, provider, on_page=None) -> dict[str, str]:
    """Fetch results of an ended batch, store the LLM artifacts and re-render. Returns {page id: outcome}."""
    outcomes: dict[str, str] = {}
    for pid, res in provider.batch_collect(batch_id).items():
        page = st.page(pid)
        if page is None:
            outcomes[pid] = "page gone"; continue
        if isinstance(res, Exception):
            st.mark_stage(pid, "llm", llm_version(s, provider.model), error=str(res)[:500])
            outcomes[pid] = f"error: {res}"
            continue
        store_llm(s, st, pid, res, provider.model)
        r = run_page(s, st, page, None, page["document"])
        outcomes[pid] = r.status
        if on_page:
            on_page(page, r)
    st.end_batch(batch_id, "ended")
    return outcomes


def wait_for_batch(provider, batch_id: str, poll_seconds: int, on_status=None, timeout: float | None = None) -> str:
    t0 = time.monotonic()
    while True:
        status, counts = provider.batch_status(batch_id)
        if on_status:
            on_status(f"batch {batch_id}: {status} {counts}")
        if status == "ended":
            return status
        if timeout is not None and time.monotonic() - t0 > timeout:
            return status
        time.sleep(poll_seconds)


# --- deletion ----------------------------------------------------------------------------------
@dataclass
class Deleted:
    pages: list[int] = field(default_factory=list)      # sequences removed
    files: int = 0                                      # artifacts removed
    raws: int = 0                                       # raw captures purged

    def summary(self) -> str:
        return f"{len(self.pages)} page(s), {self.files} artifact(s), {self.raws} raw capture(s) removed"


def delete_pages(s: Settings, st: Store, slug: str, page_ids: list[str], purge_raw: bool = True,
                 whole_capture: bool = False) -> Deleted:
    """Remove pages: database rows, processed/ocr/llm artifacts, the page file, and — when nothing
    else refers to the frame and `purge_raw` is set — the raw capture. `whole_capture` extends the
    set to every page derived from the same frames (both halves of a spread)."""
    out = Deleted()
    ids = list(dict.fromkeys(page_ids))
    if whole_capture:
        for pid in list(ids):
            page = st.page(pid)
            if page:
                ids += [r["id"] for r in st.pages_by_raw(page["raw_sha256"]) if r["id"] not in ids]
    raws: set[str] = set()
    for pid in ids:
        page = st.page(pid)
        if page is None or page["document"] != slug:
            continue
        out.files += _drop_page_artifacts(s, pid)
        md = render.page_path(s.documents / slug, page["sequence"])
        if md.exists():
            md.unlink(); out.files += 1
        raws.add(page["raw_sha256"])
        st.delete_captures_of(pid)
        st.delete_page(pid)
        out.pages.append(page["sequence"])
        log.info("deleted page %s (%s #%d)", pid, slug, page["sequence"])
    if purge_raw:
        for sha in raws:
            if not st.pages_by_raw(sha) and not st.find_capture(sha):
                raw = s.raw / f"{sha}.jpg"
                if raw.exists():
                    raw.unlink(); out.raws += 1
    return out


def delete_document(s: Settings, st: Store, slug: str, purge_raw: bool = True) -> Deleted:
    """Remove a document entirely: its pages and artifacts, page files, document.yaml, the built file."""
    import shutil
    out = delete_pages(s, st, slug, [p["id"] for p in st.pages(slug)], purge_raw=purge_raw)
    st.delete_document(slug)
    doc_dir = s.documents / slug
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    inbox = s.root / "inbox" / slug
    if inbox.exists():
        shutil.rmtree(inbox)
    log.info("deleted document %s: %s", slug, out.summary())
    return out


# --- housekeeping ------------------------------------------------------------------------------
def backfill_page_meta(s: Settings, st: Store) -> int:
    """Pages from before 0.2 have no width/advice in the database; read it from processed/<id>.json."""
    n = 0
    for page in st.pages():
        if page["width_px"] is not None:
            continue
        meta_p = s.processed / f"{page['id']}.json"
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text())
            except ValueError:
                continue
            st.set_page_meta(page["id"], meta.get("page_width_px"), meta.get("advice"))
            n += 1
    return n


def missing_artifacts(s: Settings, st: Store) -> dict[str, list[str]]:
    """Pages whose on-disk artifacts are gone (a cleaned directory, a moved root)."""
    out = {"raw": [], "processed": []}
    for page in st.pages():
        if not (s.raw / f"{page['raw_sha256']}.jpg").exists():
            out["raw"].append(page["id"])
        if not (s.processed / f"{page['id']}.png").exists():
            out["processed"].append(page["id"])
    return out


def orphan_artifacts(s: Settings, st: Store) -> list[Path]:
    """Files in processed/ocr/llm that no page refers to (left behind by an interrupted reprocess)."""
    ids = {p["id"] for p in st.pages()}
    out = []
    for d in (s.processed, s.ocr_dir, s.llm_dir):
        for f in d.glob("*"):
            if f.name.split(".")[0] not in ids:
                out.append(f)
    return out
