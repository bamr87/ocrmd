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
from . import preprocess, ocr, reconcile as rc, render, review, multifocus, legibility, annotate
from . import layout as layout_mod, pdfsrc
from .reconcile import token_overlap, _content_tokens
from .llm import Provider, LLMResult, LLMError, PageHint, Transcription, FocusViews

log = logging.getLogger("ocrmd.pipeline")
PRE_VER = preprocess.PIPELINE_VERSION
LAYOUT_VER = layout_mod.LAYOUT_VERSION


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
    focus: str | None = None

    def detail(self) -> str:
        d = f" agreement={self.agreement:.2f} overlap={self.overlap or 0:.2f}" if self.agreement is not None else ""
        if self.focus:
            d += f"  {self.focus}"
        if self.illegible_reason:
            d += f"  ({self.illegible_reason[:50]})"
        return d


# --- ingest ------------------------------------------------------------------------------------
@dataclass
class Ingested:
    """What became of one frame: the pages it produced, and — when it produced none — why not.

    A frame can now come to nothing for two different reasons, and the operator needs to be told
    which: a duplicate is a page already scanned, a rejection is a page that still needs shooting.
    """
    pages: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    rejected: list[str] = field(default_factory=list)     # reasons, one per crop refused as unreadable


def ingest_frame(s: Settings, st: Store, img: np.ndarray, slug: str, source: str,
                 variants: list[np.ndarray] | None = None, admit: bool | None = None) -> Ingested:
    """Store raw frame, dedup, preprocess into 1..2 pages, register them.

    `variants` are the focus bracket's other exposures of the same frame. They are stored beside the
    capture as `raw/<sha>.v<n>.jpg` and are as immutable as it is — the page identity, the dedup
    decision and every downstream stage still come from the primary frame alone; the variants only
    ever add evidence about text the primary could not resolve.

    `admit` runs the legibility test (`legibility.assess`) over each crop and declines to create a
    page for one no OCR engine could read. It defaults off, and the *scan* path turns it on for
    automatic captures: a file the operator handed over, or a frame they forced with the space bar,
    is a decision, not a guess the shutter made. Nothing is deleted either way — the raw frame stays
    and the capture row records why it produced no page.
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    raw = buf.tobytes()
    sha = sha256_bytes(raw)
    ph = dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    admitting = bool(admit)
    prior = st.find_capture(sha)
    if prior is not None:
        if prior["rejected"] and prior["page_id"] is None and not admitting:
            # a deliberate re-ingest of a frame the shutter refused: the operator overrules it
            st.clear_rejection(sha)
            log.info("re-admitting previously refused capture %s", sha[:8])
        else:
            return Ingested(duplicate_of=prior["duplicate_of"] or prior["page_id"])
    parts = preprocess.process(img, s.min_page_width, probe=s.orientation_probe)
    ocr_results: list[ocr.OcrResult | None] = [None] * len(parts)
    dup = find_duplicate(s, st, img, ph, slug, parts, ocr_results)
    if dup:
        st.add_capture(sha, ph, None, dup, source, document=slug)
        log.info("duplicate capture %s of page %s", sha[:8], dup)
        return Ingested(duplicate_of=dup)
    rejects = admission(s, parts, ocr_results) if admitting else []
    if rejects and all(rejects):
        # nothing on this frame is readable: keep the raw capture, create no page, spend no tokens
        (s.raw / f"{sha}.jpg").write_bytes(raw)
        st.ensure_document(slug)
        st.add_capture(sha, ph, None, None, source, rejected="; ".join(rejects), document=slug)
        log.info("rejected capture %s: %s", sha[:8], "; ".join(rejects))
        return Ingested(rejected=list(rejects))
    (s.raw / f"{sha}.jpg").write_bytes(raw)
    for i, v in enumerate(variants or [], 1):
        ok_v, vbuf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok_v:
            (s.raw / f"{sha}.v{i}.jpg").write_bytes(vbuf.tobytes())
    st.ensure_document(slug)
    keep = [i for i in range(len(parts)) if not (rejects[i] if rejects else None)]
    ids = derive_pages(s, st, img, slug, sha, ph, parts, ocr_results, keep=keep)
    st.add_capture(sha, ph, ids[0] if ids else None, None, source,
                   rejected="; ".join(r for r in rejects if r) or None, document=slug)
    return Ingested(pages=ids, rejected=[r for r in rejects if r])


def admission(s: Settings, parts, ocr_results) -> list[str]:
    """Read each crop and judge whether it is worth a page. Returns one reason per crop, "" to admit.

    The OCR pass is not extra work: `derive_pages` keeps the result as the page's OCR stage, so an
    admitted crop is read exactly once either way. A refused one costs that pass and saves an LLM
    call — the reason the test lives here rather than after the model has already been paid.
    """
    if not ocr.available_engines():
        return []
    rec = ocr.recognizer(s)
    out = []
    for i, (color, _binar, _meta) in enumerate(parts):
        if ocr_results[i] is None:
            ocr_results[i] = rec.run(color)
        v = legibility.assess(ocr_results[i], s.admit_min_words, s.admit_min_confidence)
        out.append("" if v.admit else v.reason)
    return out


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


def variant_raws(s: Settings, sha: str) -> list[Path]:
    """The focus bracket's other exposures of one capture, in bracket order."""
    return sorted(s.raw.glob(f"{sha}.v*.jpg"), key=lambda f: f.name)


def variant_crops(s: Settings, pid: str) -> list[Path]:
    return sorted(s.processed.glob(f"{pid}.v*.png"), key=lambda f: f.name)


def flat_path(s: Settings, pid: str) -> Path:
    return s.processed / f"{pid}.flat.png"


def ensure_flat(s: Settings, pid: str) -> Path | None:
    """The page's legible rendering (`preprocess.legible`), made on first use and kept beside the crop.

    Derived, not decided: it is a pure function of `<pid>.png`, so it carries no stage version, does not
    enter the page id, and can be regenerated for a page scanned long before this existed simply by
    asking for it. Delete the file and the next request rebuilds it.
    """
    out = flat_path(s, pid)
    if out.exists():
        return out
    img = cv2.imread(str(s.processed / f"{pid}.png"))
    if img is None:
        return None
    cv2.imwrite(str(out), preprocess.legible(img))
    return out


ASPECT_TOLERANCE = 0.04    # how far a variant crop may differ in shape from the page's own crop


def derive_variant_crops(s: Settings, pid: str, primary: np.ndarray, sha: str, part: str | None,
                         orientation: int | None) -> int:
    """Crop each raw focus variant the way this page was cropped, and write `processed/<pid>.v<n>.png`.

    The crops are made independently — page detection, spread split and deskew all run again — but
    with the *page's* orientation forced, and the result is then scaled onto the page crop's exact
    pixel dimensions so that normalised boxes from the two are directly comparable. A variant whose
    own detection landed somewhere materially different (a hand at the edge of that one exposure, a
    shadow that moved the sheet outline) has a different aspect ratio, and is dropped rather than
    stretched into place: a mis-cropped variant would line its text up against the wrong primary
    lines, which is worse than having no variant at all.
    """
    n = 0
    h0, w0 = primary.shape[:2]
    for i, raw in enumerate(variant_raws(s, sha), 1):
        img = cv2.imread(str(raw))
        if img is None:
            continue
        try:
            parts = preprocess.process(img, s.min_page_width, orientation=orientation)
        except Exception as e:                       # a variant is optional; never fail the page for one
            log.warning("focus variant %s could not be preprocessed (%s); skipped", raw.name, e)
            continue
        match = next((p for p in parts if p[2].get("part") == part), None)
        if match is None:
            log.debug("focus variant %s has no %s part; skipped", raw.name, part)
            continue
        crop = match[0]
        h, w = crop.shape[:2]
        if abs(w / max(h, 1) - w0 / max(h0, 1)) / max(w0 / max(h0, 1), 1e-6) > ASPECT_TOLERANCE:
            log.info("focus variant %s cropped to a different shape (%dx%d vs %dx%d); skipped", raw.name, w, h, w0, h0)
            continue
        if (h, w) != (h0, w0):
            crop = cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_AREA if w > w0 else cv2.INTER_CUBIC)
        cv2.imwrite(str(s.processed / f"{pid}.v{i}.png"), crop)
        n += 1
    return n


def derive_pages(s: Settings, st: Store, img: np.ndarray, slug: str, sha: str, ph: str,
                 parts=None, ocr_results=None, keep: list[int] | None = None) -> list[str]:
    """Preprocess one capture into 1..2 pages. The page id is the hash of the *processed* image,
    so changing the preprocess pipeline necessarily changes page identity (see `ocrmd reprocess`).

    `keep` limits which crops become pages — half of a spread can be readable while the other half is
    not, and refusing the whole frame for that would lose a good page."""
    ids = []
    parts = parts if parts is not None else preprocess.process(img, s.min_page_width, probe=s.orientation_probe)
    for i, (color, binar, meta) in enumerate(parts):
        if keep is not None and i not in keep:
            continue
        ok, pbuf = cv2.imencode(".png", color)
        pid = sha256_bytes(pbuf.tobytes())[:16]
        if st.page(pid):
            log.info("page %s already exists (identical crop); skipping", pid)
            continue
        (s.processed / f"{pid}.png").write_bytes(pbuf.tobytes())
        cv2.imwrite(str(s.processed / f"{pid}.bin.png"), binar)
        cv2.imwrite(str(flat_path(s, pid)), preprocess.legible(color))
        (s.processed / f"{pid}.json").write_text(json.dumps(meta))
        st.add_page(pid, slug, sha, ph, meta.get("part"), meta.get("page_width_px"), meta.get("advice"))
        st.mark_stage(pid, "preprocess", PRE_VER)
        variants = derive_variant_crops(s, pid, color, sha, meta.get("part"), meta.get("orientation"))
        res = ocr_results[i] if ocr_results else None
        # OCR already done for dedup: keep it — unless this page has focus variants, whose whole
        # purpose is to be read alongside the primary crop. That has to happen in `run_ocr`.
        if res is not None and not variants:
            (s.ocr_dir / f"{pid}.json").write_text(res.model_dump_json(indent=1))
            st.mark_stage(pid, "ocr", ocr.recognizer(s).version)
        ids.append(pid)
    return ids


def _drop_page_artifacts(s: Settings, pid: str) -> int:
    n = 0
    for f in (list(s.processed.glob(f"{pid}.*")) + list(s.ocr_dir.glob(f"{pid}.json"))
              + list(s.ocr_dir.glob(f"{pid}.focus.json")) + list(s.ocr_dir.glob(f"{pid}.pdf.json"))
              + list(s.llm_dir.glob(f"{pid}.*.json"))
              + list(s.layout_dir.glob(f"{pid}.json")) + list(s.documents.glob(f"*/figures/{pid}-*"))
              + list(s.export_dir.glob(f"*/*/{pid}.*"))):
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


def ingest_pdf(s: Settings, st: Store, path: Path, slug: str, on_page=None) -> list[tuple[str, Ingested]]:
    """Ingest a PDF page by page, keeping what the file already knows (see `pdfsrc`).

    The raster is stored as the immutable capture and *is* the processed crop — no page detection,
    no deskew, no spread split: a PDF page is already a page. The text layer, when there is one, is
    written beside the OCR artifacts as `ocr/<pid>.pdf.json` and becomes the primary reading of the
    OCR stage (`run_ocr`), with the local engines voting on it. Placed images are recorded in the
    page's processing meta so the layout stage takes their boxes rather than hunting for ink.
    Duplicates are exact only (same raster bytes): a book can legitimately repeat a page.
    """
    out: list[tuple[str, Ingested]] = []
    for pg in pdfsrc.pages(path):
        source = f"file:{pg.name}"
        ok, buf = cv2.imencode(".jpg", pg.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        raw = buf.tobytes()
        sha = sha256_bytes(raw)
        gray = cv2.cvtColor(pg.image, cv2.COLOR_BGR2GRAY)
        ph = dhash(gray)
        prior = st.find_capture(sha)
        if prior is not None:
            res = Ingested(duplicate_of=prior["duplicate_of"] or prior["page_id"])
            out.append((pg.name, res))
            if on_page: on_page(pg.name, res)
            continue
        den = cv2.fastNlMeansDenoising(gray, h=7)
        binar = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        meta = {"pipeline": PRE_VER, "page_detected": True, "clipped_edges": [], "orientation": 0, "part": None,
                "deskew": 0.0, **preprocess.quality(gray), "page_width_px": int(pg.image.shape[1]),
                "source": "pdf", "pdf": pg.meta()}
        (s.raw / f"{sha}.jpg").write_bytes(raw)
        st.ensure_document(slug)
        ids = derive_pages(s, st, pg.image, slug, sha, ph, parts=[(pg.image, binar, meta)])
        for pid in ids:
            if pg.text_layer:
                (s.ocr_dir / f"{pid}.pdf.json").write_text(
                    ocr.OcrResult(text=pg.text, blocks=pg.blocks, mean_confidence=1.0, engine=pdfsrc.version(),
                                  words=[(w, 1.0) for b in pg.blocks for w in b.text.split()],
                                  engines=[pdfsrc.version()]).model_dump_json(indent=1))
        st.add_capture(sha, ph, ids[0] if ids else None, None, source, document=slug)
        res = Ingested(pages=ids)
        out.append((pg.name, res))
        if on_page: on_page(pg.name, res)
    return out


def pdf_text_path(s: Settings, pid: str) -> Path:
    return s.ocr_dir / f"{pid}.pdf.json"


def pdf_text(s: Settings, pid: str) -> ocr.OcrResult | None:
    """The PDF's own text layer for this page, or None when the page did not come from one / had none."""
    p = pdf_text_path(s, pid)
    if not p.exists():
        return None
    try:
        return ocr.OcrResult.model_validate_json(p.read_text())
    except ValueError:
        return None


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
    parts = preprocess.process(cv2.imread(str(raw)), s.min_page_width, probe=s.orientation_probe)
    want = page["part"]
    match = next((p for p in parts if p[2].get("part") == want), parts[0])
    color, binar, meta = match
    cv2.imwrite(str(s.processed / f"{pid}.png"), color)
    cv2.imwrite(str(s.processed / f"{pid}.bin.png"), binar)
    cv2.imwrite(str(flat_path(s, pid)), preprocess.legible(color))
    meta_p.write_text(json.dumps(meta))
    derive_variant_crops(s, pid, color, page["raw_sha256"], want, meta.get("orientation"))
    st.mark_stage(pid, "preprocess", PRE_VER)
    return meta


def ocr_version(s: Settings) -> str | None:
    """The current OCR stage version, or None when no engine is installed."""
    try:
        return ocr.recognizer(s).version
    except RuntimeError:
        return None


def page_ocr_version(s: Settings, pid: str, version: str | None = None) -> str | None:
    """The OCR stage version this page should be at: the engine set's, with the PDF text layer named
    first when the page has one — it is the primary reading there, and a change of engines still
    re-votes on it."""
    version = version if version is not None else ocr_version(s)
    if pdf_text_path(s, pid).exists():
        names = version.split(":", 1)[1] if version else ""
        return f"{ocr.OCR_VERSION}:{pdfsrc.ENGINE}" + (f"+{names}" if names else "")
    return version


def ocr_stale(s: Settings, st: Store, page, version: str | None = None) -> bool:
    """The page's OCR was made by a different engine set / OCR version than the current one."""
    version = page_ocr_version(s, page["id"], version or ocr_version(s))
    return version is not None and not st.stage_done(page["id"], "ocr", version)


def focus_path(s: Settings, pid: str) -> Path:
    return s.ocr_dir / f"{pid}.focus.json"


def focus_analysis(s: Settings, pid: str) -> multifocus.MultiFocus | None:
    """What the focus bracket found for this page, written by the OCR stage. None: no bracket."""
    f = focus_path(s, pid)
    if not f.exists():
        return None
    try:
        return multifocus.MultiFocus.model_validate_json(f.read_text())
    except ValueError as e:                     # an artifact from an older/partial run must not block the page
        log.warning("focus analysis for %s unreadable (%s); ignored", pid, e)
        return None


def run_multifocus(s: Settings, pid: str, rec: ocr.Recognizer, res: ocr.OcrResult) -> ocr.OcrResult:
    """Read the page again at each bracketed focus and let the variants answer for what it read badly.

    The variants are read with the *same* recogniser as the page — every configured engine, the same
    preparation, the same merge — because `resolve` compares their confidences with the page's own,
    and two numbers produced by different procedures are not comparable.

    Every variant is read, including when the page's own lines are all confident. Skipping those was
    tried first and is wrong: measured on a page whose top half is out of focus (520 px wide, five
    blur radii), the two captures that gained the most were both read *confidently* by the primary —
    it had simply not emitted the unreadable lines at all, and confidence says nothing about a line an
    engine never produced. Correct words recovered, primary vs bracket: 55 -> 165 and 101 -> 146 on
    those two; 117 -> 138 and 64 -> 165 on the two that were visibly doubtful; and on the sharp
    capture, where there was nothing to gain, the bracket changed nothing. The bill is one OCR pass per
    variant (~1.2 s here with two engines), on the worker thread, for pages the shutter was already
    unhappy enough about to bracket.
    """
    crops = variant_crops(s, pid)
    if not crops:
        focus_path(s, pid).unlink(missing_ok=True)
        return res
    pairs = []
    for f in crops:
        img = cv2.imread(str(f))
        if img is None:
            continue
        v = rec.run(img)
        pairs.append((multifocus.FocusReading(label=f.name.split(".")[-2], lines=len(v.blocks),
                                              mean_confidence=round(v.mean_confidence, 3)), v))
    res, mf = multifocus.resolve(res, pairs, s.ocr_trust, s.focus_swap_margin)
    focus_path(s, pid).write_text(mf.model_dump_json(indent=1))
    log.info("focus %s: %s", pid, mf.summary())
    return res


def run_ocr(s: Settings, st: Store, page) -> ocr.OcrResult:
    """Cached per (page, engine set, OCR version): a new engine or a merge change re-reads the page the
    next time it is processed — local and free, unlike the LLM stage."""
    pid = page["id"]
    out = s.ocr_dir / f"{pid}.json"
    layer = pdf_text(s, pid)
    try:
        rec = ocr.recognizer(s)
    except RuntimeError:
        if layer is None:
            raise
        rec = None                                  # no engine, but the PDF spelled the page out itself
    version = page_ocr_version(s, pid, rec.version if rec else f"{ocr.OCR_VERSION}:")
    if st.stage_done(pid, "ocr", version) and out.exists():
        return ocr.OcrResult.model_validate_json(out.read_text())
    img = cv2.imread(str(s.processed / f"{pid}.png"))
    if img is None:
        raise FileNotFoundError(f"processed image missing for page {pid}")
    if layer is not None:
        res = merge_pdf_text(layer, rec.run(img) if rec else None)
    else:
        res = rec.run(img)
        res = run_multifocus(s, pid, rec, res)
    out.write_text(res.model_dump_json(indent=1))
    st.mark_stage(pid, "ocr", version)
    log.debug("ocr %s: %s %.2fs agreement=%s", pid, res.engine, sum(res.timings.values()), res.agreement)
    return res


def merge_pdf_text(layer: ocr.OcrResult, engines: ocr.OcrResult | None) -> ocr.OcrResult:
    """The PDF's text layer as the primary reading, the engines voting on it.

    Confidence 1.0 is what the file claims, not what was measured, so the vote is the measurement:
    a word no engine confirms is doubted (`ocr.DOUBTED`) exactly as a doubted Vision word would be,
    and the reconcile stage flags it against the model's reading. A text layer that is not a
    reading of the pixels at all — a bad OCR layer on a scan, a watermark, a wrong page — comes out
    with a low agreement, which is the point of running the engines on a PDF at all.
    """
    if engines is None or not engines.blocks:
        return layer
    others = {engines.engine: engines.blocks}
    blocks, words, agreement = ocr.merge(layer.blocks, others, fill=True, swap=False)
    conf = float(np.mean([b.confidence for b in blocks])) if blocks else 0.0
    return ocr.OcrResult(text="\n".join(b.text for b in blocks), blocks=blocks, mean_confidence=conf,
                         engine="+".join([layer.engine] + engines.engines), words=words,
                         engines=[layer.engine] + engines.engines, agreement=agreement, timings=engines.timings)


def focus_views(s: Settings, pid: str) -> FocusViews | None:
    """The extra exposures and the analysis to send with the LLM request, or None when the page needs
    neither: no bracket, or every region resolved locally. Images cost tokens; regions cost nothing."""
    mf = focus_analysis(s, pid)
    if mf is None or not mf.unresolved:
        return None
    wanted = multifocus.variant_ranking(mf)[:max(0, s.focus_llm_images)]
    images, labels = [], []
    for label in wanted:
        img = cv2.imread(str(s.processed / f"{pid}.{label}.png"))
        if img is not None:
            images.append(img); labels.append(label)
    note = multifocus.describe(mf, labels)
    return FocusViews(images=images, note=note) if note else None


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
    focus = focus_views(s, pid)
    if focus is not None:
        log.info("page %s: sending %d focus variant(s) to the model", pid, len(focus.images))
    try:
        res = provider.transcribe(img, ocr_res.text, hint, focus)
    except LLMError as e:  # recorded, page stays unverified
        st.mark_stage(pid, "llm", llm_version(s, provider.model), error=str(e)[:500])
        raise
    store_llm(s, st, pid, res, provider.model)
    return res


# --- layout: the page model ---------------------------------------------------------------------
def layout_path(s: Settings, pid: str) -> Path:
    return s.layout_dir / f"{pid}.json"


def figures_dir(s: Settings, slug: str) -> Path:
    """Figure crops live beside the page files, so the Markdown's `![…](../figures/…)` is portable."""
    return s.documents / slug / "figures"


def load_layout(s: Settings, pid: str) -> layout_mod.Layout | None:
    p = layout_path(s, pid)
    if not p.exists():
        return None
    try:
        return layout_mod.Layout.model_validate_json(p.read_text())
    except ValueError as e:                     # an artifact from an older run must not block the page
        log.warning("layout for %s unreadable (%s); rebuilding", pid, e)
        return None


def save_layout(s: Settings, st: Store, lay: layout_mod.Layout) -> Path:
    p = layout_path(s, lay.page_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(lay.model_dump_json(indent=1))
    tmp.replace(p)
    st.mark_stage(lay.page_id, "layout", LAYOUT_VER)
    return p


def write_figure_assets(s: Settings, slug: str, pid: str, img: np.ndarray, lay: layout_mod.Layout) -> int:
    """Cut every figure out of the crop into `documents/<slug>/figures/<pid>-<n>.png` and name it on the
    block. A figure that a human placed keeps its asset if it already has one."""
    out = figures_dir(s, slug)
    n = 0
    for i, f in enumerate(lay.figures, 1):
        if f.asset and (out / f.asset).exists() and f.source == "human":
            continue
        name = f"{pid}-{i}.png"
        out.mkdir(parents=True, exist_ok=True)
        region = layout_mod.crop(img, f.bbox)
        if region.size == 0:
            continue
        cv2.imwrite(str(out / name), region)
        f.asset = name
        n += 1
    return n


def run_layout(s: Settings, st: Store, page, ocr_res: ocr.OcrResult, body: str, slug: str,
               img: np.ndarray | None = None) -> tuple[layout_mod.Layout, list[str]]:
    """The layout stage: join OCR geometry to the finished body, find figures, write the model.

    Runs on every render — it is a pure function of artifacts already on disk and costs no tokens —
    and is versioned like every other stage so `verify` rebuilds one made by an older join. Returns
    the model and the ids of blocks carried over from a human's edits to the previous one.
    """
    pid = page["id"]
    if img is None:
        img = cv2.imread(str(s.processed / f"{pid}.png"))
    if img is None:
        raise FileNotFoundError(f"processed image missing for page {pid}")
    meta_p = s.processed / f"{pid}.json"
    boxes = None
    if meta_p.exists():
        try:
            boxes = (json.loads(meta_p.read_text()).get("pdf") or {}).get("images")
        except ValueError:
            boxes = None
    lay = layout_mod.assemble(ocr_res, body, img, page_id=pid, figure_boxes=boxes)
    kept = layout_mod.merge_human_layout(lay, load_layout(s, pid))
    write_figure_assets(s, slug, pid, img, lay)
    save_layout(s, st, lay)
    return lay, kept


def page_layout(s: Settings, st: Store, page) -> layout_mod.Layout:
    """The page model, from disk when the stage is current, rebuilt from the artifacts when not."""
    pid = page["id"]
    lay = load_layout(s, pid) if st.stage_done(pid, "layout", LAYOUT_VER) else None
    if lay is not None:
        return lay
    md_path = render.page_path(s.documents / page["document"], page["sequence"])
    body = render.read_page(md_path)[1] if md_path.exists() else ""
    lay, _ = run_layout(s, st, page, run_ocr(s, st, page), body, page["document"])
    return lay


def figure_ref(pid: str, asset: str, caption: str | None) -> str:
    return f"![{caption or 'figure'}](../figures/{asset})"


def insert_figure_refs(body: str, lay: layout_mod.Layout) -> str:
    """Put `![caption](../figures/<asset>)` into the body where each figure sits in reading order.

    Only for figures the body does not already reference; a reference that is there — from a
    previous render, or typed by hand — is left exactly as it is. The line is placed after the last
    text block that reads before the figure, so a figure between two paragraphs lands between them.
    """
    blocks = sorted(lay.blocks, key=lambda b: b.order)
    missing = [f for f in lay.figures if f.asset and f"figures/{f.asset}" not in body]
    if not missing:
        return body
    paras = body.rstrip().split("\n\n") if body.strip() else []
    text_blocks = [b for b in blocks if b.role != "figure" and b.text.strip()]
    # map each text block to the paragraph that carries its first words, so an insertion point in
    # block space becomes one in paragraph space
    def para_of(b: layout_mod.Block) -> int | None:
        head = " ".join(b.text.split()[:4])
        for i, p in enumerate(paras):
            if head and head in " ".join(p.split()):
                return i
        return None
    inserts: list[tuple[int, str]] = []
    for f in missing:
        before = [b for b in text_blocks if b.order < f.order]
        at = len(paras)
        for b in reversed(before):
            i = para_of(b)
            if i is not None:
                at = i + 1
                break
        else:
            at = 0
        inserts.append((at, figure_ref(lay.page_id, f.asset, f.caption)))
    for at, ref in sorted(inserts, key=lambda x: -x[0]):
        paras.insert(at, ref)
    return "\n\n".join(paras).rstrip() + "\n"


def locate_flags(lay: layout_mod.Layout, body: str) -> list[dict]:
    """Where each open `==[…]==` flag sits on the page: the line whose reconciled words contain the
    flagged reading, so the image pane can draw it. A flag with no line gets no box, not a guess."""
    out = []
    norm = lambda t: " ".join(layout_mod._norm(w) for w in t.split())
    for f in review.parse_flags(body):
        key = norm(f.llm) or norm(f.ocr)
        hit = None
        if key:
            for b in lay.blocks:
                for ln in b.lines:
                    if key in norm(ln.text):
                        hit = (b, ln); break
                if hit:
                    break
        out.append({"n": f.n, "llm": f.llm, "ocr": f.ocr,
                    "bbox": hit[1].bbox if hit else None, "block": hit[0].id if hit else None})
    return out


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
        "source": {"type": "webcam" if source.startswith("webcam") else "pdf" if meta.get("source") == "pdf" else "file",
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
    mf = focus_analysis(s, pid)
    if mf is not None and mf.variants:
        fm["ocr"]["focus"] = {"variants": len(mf.variants), "improved": mf.improved, "filled": mf.filled,
                              "unresolved": len(mf.unresolved), "artifact": f"ocr/{pid}.focus.json"}
        if mf.useful:
            result.focus = f"focus: {mf.improved}+{mf.filled} lines, {len(mf.unresolved)} unresolved"
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
        if tr.columns > 1 or tr.has_math or tr.tables:
            fm["processing"]["structure"] = {"columns": tr.columns, "math": tr.has_math, "tables": tr.tables}
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
    # human edits: if the file changed since the pipeline last wrote it, keep what the human wrote
    frozen = False                      # a signed-off body is never touched, not even to add a figure
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
            frozen = bool(reviewer)

    # the page model: geometry joined to the finished body, figures cut out, captions matched
    try:
        lay, kept_layout = run_layout(s, st, page, ocr_res, body, slug)
    except FileNotFoundError as e:
        log.warning("page %s: no layout (%s)", pid, e)
        lay, kept_layout = None, []
    if lay is not None:
        tr_ = llm_res.transcription if llm_res is not None else None
        llm_caps = ([f.caption for f in tr_.figures] or list(tr_.context.figures or [])) if tr_ is not None else []
        for f, cap in zip(lay.figures, llm_caps, strict=False):
            if not f.caption and cap:
                f.caption = cap
        if not frozen:
            body = insert_figure_refs(body, lay)
        if kept_layout:
            result.preserved += [f"layout.{b}" for b in kept_layout]
        save_layout(s, st, lay)
        fm["processing"]["layout"] = {"version": lay.version, "blocks": len(lay.blocks), "columns": lay.columns,
                                      "figures": len(lay.figures), "artifact": f"layout/{pid}.json"}
        fm["references"]["figures"] = [{"asset": f.asset, "caption": f.caption, "bbox": [round(v, 4) for v in f.bbox]}
                                       for f in lay.figures]
    fm["history"] = st.stage_history(pid)
    result.status = status
    fm["verification"]["status"] = status
    render.write_page(doc_dir, page["sequence"], fm, body)
    st.set_fingerprint(pid, review.fingerprint(fm, body))
    st.set_status(pid, status, fm["page_label"])
    st.set_page_meta(pid, meta.get("page_width_px"), meta.get("advice"))
    return result


# --- annotation sync ------------------------------------------------------------------------------
def identify_page(s: Settings, st: Store, img: np.ndarray, slug: str) -> dict | None:
    """Which already-scanned page is this a fresh view of? None when it is a page we have not seen.

    The same two-step the deduplicator uses — dHash prefilter, then correlation against the stored raw
    capture — because it is the same question asked for the opposite purpose: dedup wants to *drop* a
    re-capture, annotation sync wants to *find* it.
    """
    ph = dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    best, best_sim = None, 0.0
    for cand in st.near_duplicates(ph, slug, s.near_duplicate_distance):
        raw = s.raw / f"{cand['raw_sha256']}.jpg"
        other = cv2.imread(str(raw)) if raw.exists() else None
        if other is None or other.shape[:2] != img.shape[:2]:
            continue
        sim = similarity(img, other)
        if sim > best_sim:
            best, best_sim = cand, sim
    if best is None or best_sim < s.annotate_match:
        return None
    return st.page(best["id"])


def sync_annotations(s: Settings, st: Store, page, frame: np.ndarray, provider=None,
                     apply: bool = True) -> annotate.Sync:
    """Diff a fresh view of a scanned page against the stored crop and fold the marks into the page.

    Nothing here re-reads the printed text. The line boxes are already on disk, so an underline or a
    highlight is resolved by geometry alone and costs nothing; only handwriting reaches a model, and
    only when there is some.
    """
    pid = page["id"]
    out = annotate.Sync(page_id=pid, sequence=page["sequence"])
    ref = cv2.imread(str(s.processed / f"{pid}.png"))
    if ref is None:
        out.reason = "processed crop missing"; return out
    parts = preprocess.process(frame, s.min_page_width, probe=s.orientation_probe)
    want = page["part"]
    match = next((p for p in parts if p[2].get("part") == want), parts[0] if parts else None)
    if match is None:
        out.reason = "no page found in the frame"; return out
    try:
        res = run_ocr(s, st, page)                     # cached: this does not re-read anything
        blocks = [b for b in res.blocks if b.text.strip()]
    except Exception as e:
        out.reason = f"no stored OCR for this page ({e})"; return out
    d = annotate.compare(ref, match[0], blocks)
    out.diff = d
    if not d.usable:
        out.reason = d.reason or "alignment failed"; return out
    if not d.marks:
        return out
    for m in d.needs_llm():
        if provider is None or not hasattr(provider, "read_handwriting"):
            continue
        try:
            m.text = provider.read_handwriting(annotate.crop_mark(ref, m.bbox))
            out.llm_calls += 1
        except LLMError as e:
            log.warning("handwriting note not read: %s", e)
    if not apply:
        out.changes = [m.describe() for m in d.marks]
        return out
    path = render.page_path(s.documents / page["document"], page["sequence"])
    if not path.exists():
        out.reason = "page file missing"; return out
    fm, body = render.read_page(path)
    new_body, changes = annotate.apply_marks(body, d.marks)
    if changes:
        # written *without* refreshing the fingerprint, exactly like a browser edit: what someone
        # wrote on the paper is a human edit and must survive the next `verify`
        render.write_page(s.documents / page["document"], page["sequence"], fm, new_body)
        st.set_status(pid, fm.get("verification", {}).get("status", page["status"]), fm.get("page_label"))
    out.changes = changes
    return out


REGION_PAD = 0.006       # of the page, grown around a hand-drawn selection so nothing is clipped
MIN_REGION = 0.0004      # a selection this small is a stray click, not a request
REGION_MAX_SCALE = 6     # a region is a fraction of a page: it can take more enlargement than one


@dataclass
class RegionRead:
    """What one hand-selected region of a page says."""
    page_id: str
    sequence: int
    bbox: list[float]
    text: str = ""                      # local OCR, read at region scale
    confidence: float = 0.0
    engine: str = ""
    scale: float = 1.0                  # how much the crop was enlarged before it was read
    size: tuple[int, int] = (0, 0)      # the crop's own pixels, before enlarging
    llm: str | None = None              # the model's reading, when one was asked for
    llm_error: str | None = None

    def describe(self) -> str:
        out = f"{self.text.strip() or '(nothing read)'}"
        if self.llm:
            out += f"\n\nmodel: {self.llm}"
        return out


def region_crop(s: Settings, pid: str, bbox: list[float], flat: bool = False) -> np.ndarray | None:
    """The page crop inside `bbox` (x, y, w, h as fractions), padded, optionally as the flat rendering."""
    src = ensure_flat(s, pid) if flat else (s.processed / f"{pid}.png")
    img = cv2.imread(str(src)) if src is not None else None
    if img is None:
        return None
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = int(max(0.0, x - REGION_PAD) * w), int(max(0.0, y - REGION_PAD) * h)
    x1, y1 = int(min(1.0, x + bw + REGION_PAD) * w), int(min(1.0, y + bh + REGION_PAD) * h)
    return img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]


def read_region(s: Settings, st: Store, page, bbox: list[float], provider: Provider | None = None,
                question: str | None = None) -> RegionRead:
    """Read one region of a page harder than the page itself was read.

    Nothing about the page changes and no stage is marked — the only file this can create is the flat
    rendering it reads from, which is a cache of the crop. This is a person pointing at a word the
    pipeline got wrong and asking what it says. Two things make the second look better. The crop is
    enlarged on its own terms — a region is a fraction of a page, so the same pixel budget buys several
    times the glyph height, which is exactly what Vision is short of on webcam pages (`ocr.MIN_SIDE`) —
    and it is read as the flat rendering rather than the raw crop, which is what `preprocess.legible`
    was measured for.

    The model, when asked, sees the same crop and the OCR draft as a hint, and answers about the region
    alone. Its answer is returned beside the OCR one rather than replacing it: this path has no second
    reading to reconcile against, so the operator is the reconciler.
    """
    pid = page["id"]
    x, y, w, h = [float(v) for v in bbox]
    if w * h < MIN_REGION:
        raise ValueError(f"selection is too small to read ({w * h:.5f} of the page)")
    out = RegionRead(page_id=pid, sequence=page["sequence"], bbox=[x, y, w, h])
    crop = region_crop(s, pid, out.bbox, flat=True)
    if crop is None:
        raise FileNotFoundError(f"processed image missing for page {pid}")
    out.size = (crop.shape[1], crop.shape[0])
    big = ocr.upscale_for_ocr(crop, s.ocr_min_side, max_scale=REGION_MAX_SCALE)
    out.scale = round(big.shape[1] / max(1, crop.shape[1]), 2)
    try:
        res = ocr.recognizer(s).run(big)
        out.text, out.confidence, out.engine = res.text, res.mean_confidence, res.engine
    except RuntimeError as e:                      # no engine installed: the model can still answer
        log.info("region OCR unavailable (%s)", e)
    if provider is not None and hasattr(provider, "read_region"):
        try:
            # the enlarged crop, not the original: the provider resizes to its own long-side budget
            # (`llm._jpeg_b64`), so sending the big one is what makes a small selection fill it
            out.llm = provider.read_region(big, out.text, question)
        except LLMError as e:
            out.llm_error = str(e)[:300]
    return out


# --- batches -----------------------------------------------------------------------------------
def submit_batch(s: Settings, st: Store, pages, provider, slug_of) -> tuple[str | None, list[str]]:
    """Queue the LLM stage for `pages` through the provider's batch API. Returns (batch id, page ids)."""
    items = []
    for page in pages:
        pid = page["id"]
        run_preprocess(s, st, page)
        ocr_res = run_ocr(s, st, page)
        img = cv2.imread(str(s.processed / f"{pid}.png"))
        items.append((pid, img, ocr_res.text, page_hint(s, st, page, slug_of(page)), focus_views(s, pid)))
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
                for raw in [s.raw / f"{sha}.jpg", *variant_raws(s, sha)]:
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
    for d in (s.processed, s.ocr_dir, s.llm_dir, s.layout_dir):
        for f in d.glob("*"):
            if f.name.split(".")[0] not in ids:
                out.append(f)
    return out
