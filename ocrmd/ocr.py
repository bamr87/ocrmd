"""Local OCR: engines, image preparation, and a cross-engine merge.

Engines — each takes a BGR page crop and returns lines with normalised, top-left-origin boxes:
  apple_vision  VNRecognizeTextRequest called directly through pyobjc (macOS).
  rapidocr      PP-OCR detection + recognition on ONNX Runtime; any platform (`pip install 'ocrmd[rapidocr]'`).
  tesseract     the tesseract binary via pytesseract (`ocrmd[tesseract]`); a last resort on camera images.

Measured on 35 textbook pages shot with a webcam, scored against the LLM transcription (bag-of-words
F1): Vision 0.74, RapidOCR 0.66, Tesseract 0.39, Apple Live Text 0.72. Background flattening, CLAHE,
unsharp masking and upscaling beyond 1400 px all scored at or below plain cubic upscaling, so
preparation is deliberately minimal. Turning Vision's language correction off costs 6 points.

With more than one engine the primary's lines are kept and the others *vote*: a word another engine
read identically is confirmed (its confidence rises to at least `CONFIRMED`), a word another engine
read differently is doubted (confidence scaled by `DOUBTED`), a line the primary missed is filled in
from the best other reading, and a line the primary read badly is swapped for a confident reading of
the same line. Reconciliation with the LLM then trusts confirmed words and flags doubted ones.
"""
from __future__ import annotations
import logging, math, platform, re, shutil, threading, time
from difflib import SequenceMatcher
from typing import TYPE_CHECKING
from pydantic import BaseModel
import numpy as np

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger("ocrmd.ocr")

OCR_VERSION = "ocr@2"      # bump when preparation or the merge changes: it is part of the stage version
MIN_SIDE = 1400            # Vision needs ~12px+ text height; webcam page crops are often ~600px wide
PRIORITY = ("apple_vision", "rapidocr", "tesseract")
CONFIRMED = 0.90           # floor for a word two engines agree on (>= Settings.ocr_trust: OCR may win a diff)
DOUBTED = 0.60             # factor for a word another engine read differently (falls below ocr_trust)
FILL_CONFIDENCE = 0.85     # a line only a secondary engine saw is kept above this confidence
SWAP_BELOW = 0.50          # a primary line under this confidence may be replaced by a >= CONFIRMED reading


class Block(BaseModel):
    text: str
    confidence: float
    bbox: list[float]           # x, y, w, h normalised (0-1, origin top-left)
    engine: str = ""            # which engine produced the text
    agreed: float | None = None  # share of the words another engine confirmed (None: single engine)


class OcrResult(BaseModel):
    text: str
    blocks: list[Block]
    mean_confidence: float
    engine: str                              # "apple_vision@26.5+rapidocr@3.9.2"
    words: list[tuple[str, float]] = []      # (word, confidence) in reading order
    engines: list[str] = []                  # every engine that ran, with version
    agreement: float | None = None           # share of primary words another engine confirmed
    timings: dict[str, float] = {}           # seconds per engine


# --- preparation -------------------------------------------------------------------------------
def upscale_for_ocr(bgr: np.ndarray, min_side: int = MIN_SIDE) -> np.ndarray:
    import cv2
    short = min(bgr.shape[:2])
    f = min(3, math.ceil(min_side / short)) if short < min_side else 1
    return cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else bgr


def flatten(bgr: np.ndarray) -> np.ndarray:
    """Divide out slow illumination (page curl, lamp fall-off). Helps Tesseract (+5 F1), hurts Vision."""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(31, (min(gray.shape) // 8) | 1)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(closed, (k, k), 0)
    return cv2.divide(gray, bg, scale=255)


def _sorted(blocks: list[Block]) -> list[Block]:
    return sorted(blocks, key=lambda b: (round(b.bbox[1], 2), b.bbox[0]))


# --- engines -----------------------------------------------------------------------------------
class Engine:
    name = ""

    def version(self) -> str:
        return self.name

    def lines(self, bgr: np.ndarray) -> list[Block]:
        raise NotImplementedError


def vision_lines(bgr: np.ndarray, level: str = "accurate", languages: list[str] | None = None,
                 correction: bool = True, vocabulary: list[str] | None = None) -> list[Block]:
    """One VNRecognizeTextRequest. The 'fast' level does not auto-orient (the accurate one partly does),
    which is what makes it usable as an upright-vs-upside-down probe in preprocess."""
    import cv2, objc, Vision
    from Foundation import NSData
    ok, buf = cv2.imencode(".png", bgr)
    out: list[Block] = []
    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(1 if level == "fast" else 0)
        req.setUsesLanguageCorrection_(bool(correction))
        if languages:
            req.setRecognitionLanguages_(list(languages))
        if vocabulary:
            req.setCustomWords_(list(vocabulary))
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        ret = handler.performRequests_error_([req], None)
        if isinstance(ret, tuple) and not ret[0]:
            raise RuntimeError(f"Vision request failed: {ret[1]}")
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            bb = obs.boundingBox()
            out.append(Block(text=str(cand[0].string()), confidence=float(cand[0].confidence()),
                             bbox=[float(bb.origin.x), float(1 - bb.origin.y - bb.size.height),
                                   float(bb.size.width), float(bb.size.height)], engine="apple_vision"))
    return _sorted(out)


class AppleVision(Engine):
    name = "apple_vision"

    def __init__(self, languages: list[str] | None = None, correction: bool = True, min_side: int = MIN_SIDE):
        self.languages, self.correction, self.min_side = languages, correction, min_side

    def version(self) -> str:
        return f"apple_vision@{platform.mac_ver()[0]}"

    def lines(self, bgr: np.ndarray) -> list[Block]:
        return vision_lines(upscale_for_ocr(bgr, self.min_side), "accurate", self.languages, self.correction)


class RapidOcr(Engine):
    """PP-OCR (PaddleOCR's models) through the rapidocr package on ONNX Runtime. Models are fetched
    on first use. Single instance: the runtime sessions are heavy and thread-safe."""
    name = "rapidocr"
    _engine = None
    _lock = threading.Lock()

    def __init__(self, min_side: int = MIN_SIDE):
        self.min_side = min_side

    def version(self) -> str:
        from importlib.metadata import version
        return f"rapidocr@{version('rapidocr')}"

    @classmethod
    def _get(cls):
        with cls._lock:
            if cls._engine is None:
                from rapidocr import RapidOCR      # the import installs an INFO console logger; quieten it
                for name in ("rapidocr", "RapidOCR"):
                    lg = logging.getLogger(name)
                    lg.setLevel(logging.WARNING)
                    for h in lg.handlers:
                        h.setLevel(logging.WARNING)
                cls._engine = RapidOCR()
            return cls._engine

    def lines(self, bgr: np.ndarray) -> list[Block]:
        img = upscale_for_ocr(bgr, self.min_side)
        r = self._get()(img)
        h, w = img.shape[:2]
        out = []
        if r.txts is None or r.boxes is None:
            return []
        for t, s, box in zip(list(r.txts), list(r.scores), list(r.boxes), strict=False):
            xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
            out.append(Block(text=str(t), confidence=float(s), engine=self.name,
                             bbox=[min(xs) / w, min(ys) / h, (max(xs) - min(xs)) / w, (max(ys) - min(ys)) / h]))
        return _sorted(out)


class Tesseract(Engine):
    name = "tesseract"

    def __init__(self, min_side: int = MIN_SIDE):
        self.min_side = min_side

    def version(self) -> str:
        import pytesseract
        return f"tesseract@{pytesseract.get_tesseract_version()}"

    def lines(self, bgr: np.ndarray) -> list[Block]:
        import pytesseract
        img = upscale_for_ocr(flatten(bgr), self.min_side)
        d = pytesseract.image_to_data(img, config="--psm 3", output_type=pytesseract.Output.DICT)
        h, w = img.shape[:2]
        lines: dict[tuple, list] = {}
        for i, t in enumerate(d["text"]):
            if not t.strip() or int(d["conf"][i]) < 0:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append((t, int(d["conf"][i]) / 100, d["left"][i], d["top"][i],
                                              d["width"][i], d["height"][i]))
        out = []
        for ws in lines.values():
            x0, y0 = min(x[2] for x in ws), min(x[3] for x in ws)
            x1, y1 = max(x[2] + x[4] for x in ws), max(x[3] + x[5] for x in ws)
            out.append(Block(text=" ".join(x[0] for x in ws), confidence=float(np.mean([x[1] for x in ws])),
                             bbox=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h], engine=self.name))
        return _sorted(out)


def _importable(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def available_engines() -> list[str]:
    """Installed engines in priority order."""
    out = []
    if platform.system() == "Darwin" and _importable("Vision"):
        out.append("apple_vision")
    if _importable("rapidocr") and _importable("onnxruntime"):
        out.append("rapidocr")
    if _importable("pytesseract") and shutil.which("tesseract"):
        out.append("tesseract")
    return out


def available_engine() -> str | None:
    """The primary engine, or None when nothing is installed."""
    eng = available_engines()
    return eng[0] if eng else None


def make_engine(name: str, languages: list[str] | None = None, min_side: int = MIN_SIDE) -> Engine:
    if name == "apple_vision":
        return AppleVision(languages, True, min_side)
    if name == "rapidocr":
        return RapidOcr(min_side)
    if name == "tesseract":
        return Tesseract(min_side)
    raise ValueError(f"unknown OCR engine {name!r} (choose from {', '.join(PRIORITY)})")


# --- merge -------------------------------------------------------------------------------------
def _norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def _same_line(p: Block, o: Block) -> bool:
    """Boxes on the same text line: half of the shorter box's height overlaps, and they overlap in x."""
    py0, py1 = p.bbox[1], p.bbox[1] + p.bbox[3]
    oy0, oy1 = o.bbox[1], o.bbox[1] + o.bbox[3]
    v = max(0.0, min(py1, oy1) - max(py0, oy0)) / max(1e-6, min(p.bbox[3], o.bbox[3]))
    hx = min(p.bbox[0] + p.bbox[2], o.bbox[0] + o.bbox[2]) - max(p.bbox[0], o.bbox[0])
    return v >= 0.5 and hx > 0


def merge(primary: list[Block], others: dict[str, list[Block]], fill: bool = True, swap: bool = True
          ) -> tuple[list[Block], list[tuple[str, float]], float | None]:
    """Vote the secondary engines' readings onto the primary's lines.

    Returns (blocks, words with confidence, share of primary words confirmed)."""
    if not others:
        words = [(w, b.confidence) for b in primary for w in b.text.split()]
        return primary, words, None
    used: dict[str, set[int]] = {name: set() for name in others}
    out: list[tuple[Block, list[tuple[str, float]]]] = []     # block with its per-word confidences
    confirmed_total = total = 0
    for p in primary:
        toks = p.text.split()
        norm = [_norm_word(w) for w in toks]
        confirmed: set[int] = set()
        doubted: set[int] = set()
        best_alt: Block | None = None
        for name, blocks in others.items():
            same = [(i, o) for i, o in enumerate(blocks) if _same_line(p, o)]
            if not same:
                continue
            same.sort(key=lambda io: io[1].bbox[0])
            used[name].update(i for i, _ in same)
            alt_text = " ".join(o.text for _, o in same)
            alt_conf = min(o.confidence for _, o in same)
            alt = [_norm_word(w) for w in alt_text.split()]
            sm = SequenceMatcher(None, norm, alt, autojunk=False)
            for op, i1, i2, _j1, _j2 in sm.get_opcodes():
                if op == "equal":
                    confirmed.update(range(i1, i2))
                elif op in ("replace", "delete"):
                    doubted.update(range(i1, i2))
            if alt_conf >= CONFIRMED and len(alt) >= 3 and 0.7 <= len(alt) / max(1, len(norm)) <= 1.4:
                if best_alt is None or alt_conf > best_alt.confidence:
                    best_alt = Block(text=alt_text, confidence=alt_conf, bbox=p.bbox, engine=name)
        doubted -= confirmed
        total += len(toks)
        confirmed_total += len(confirmed)
        if swap and best_alt is not None and p.confidence < SWAP_BELOW and len(confirmed) < len(toks) / 2:
            # the primary barely read this line and another engine is sure of it: take that reading
            b = best_alt.model_copy(update={"agreed": 0.0})
            out.append((b, [(w, b.confidence) for w in b.text.split()]))
            continue
        conf = [max(p.confidence, CONFIRMED) if i in confirmed else p.confidence * DOUBTED if i in doubted
                else p.confidence for i in range(len(toks))]
        agreed = len(confirmed) / len(toks) if toks else None
        b = p.model_copy(update={"agreed": agreed, "confidence": float(np.mean(conf)) if conf else p.confidence})
        out.append((b, list(zip(toks, conf, strict=True))))
    if fill:
        for name, blocks in others.items():
            for i, o in enumerate(blocks):
                if i in used[name] or o.confidence < FILL_CONFIDENCE or len(o.text.split()) < 2:
                    continue
                if any(_same_line(q, o) for q, _ in out):
                    continue
                b = o.model_copy(update={"agreed": 0.0})
                out.append((b, [(w, b.confidence) for w in b.text.split()]))
                used[name].add(i)
        out.sort(key=lambda e: (round(e[0].bbox[1], 2), e[0].bbox[0]))
    agreement = confirmed_total / total if total else None
    return [b for b, _ in out], [w for _, ws in out for w in ws], agreement


# --- recogniser --------------------------------------------------------------------------------
class Recognizer:
    """The configured engine set. `version` is the OCR stage version: it changes when the engines,
    the preparation or the merge change, and `run_ocr` re-reads a page whose stage is older."""

    def __init__(self, names: list[str], languages: list[str] | None = None, min_side: int = MIN_SIDE,
                 fill: bool = True, swap: bool = True):
        if not names:
            raise RuntimeError("No OCR engine: on macOS install pyobjc-framework-Vision; elsewhere "
                               "pip install 'ocrmd[rapidocr]' (or [tesseract] with the tesseract binary)")
        self.engines = [make_engine(n, languages, min_side) for n in names]
        self.fill, self.swap = fill, swap

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.engines]

    @property
    def version(self) -> str:
        return f"{OCR_VERSION}:{'+'.join(self.names)}"

    def run(self, bgr: np.ndarray) -> OcrResult:
        results: dict[str, list[Block]] = {}
        versions, timings = [], {}
        for e in self.engines:
            t0 = time.monotonic()
            try:
                results[e.name] = e.lines(bgr)
            except Exception as ex:
                if e is self.engines[0]:
                    raise
                log.warning("OCR engine %s failed (%s); continuing without it", e.name, ex)
                continue
            timings[e.name] = round(time.monotonic() - t0, 3)
            versions.append(e.version())
        primary = self.engines[0].name
        others = {n: b for n, b in results.items() if n != primary}
        blocks, words, agreement = merge(results[primary], others, self.fill, self.swap)
        conf = float(np.mean([b.confidence for b in blocks])) if blocks else 0.0
        return OcrResult(text="\n".join(b.text for b in blocks), blocks=blocks, mean_confidence=conf,
                         engine="+".join(versions), words=words, engines=versions, agreement=agreement,
                         timings=timings)


def recognizer(s: "Settings | None" = None) -> Recognizer:
    """Engines from settings (`ocr_engines`), or everything installed in priority order."""
    avail = available_engines()
    if s is not None and s.ocr_engines:
        missing = [n for n in s.ocr_engines if n not in avail]
        if missing:
            raise RuntimeError(f"OCR engine(s) not installed: {', '.join(missing)} (installed: {', '.join(avail) or 'none'})")
        names = list(s.ocr_engines)
    else:
        names = avail[:2] if s is None or s.ocr_ensemble else avail[:1]
    languages = list(s.ocr_languages) if s is not None and s.ocr_languages else None
    return Recognizer(names, languages, s.ocr_min_side if s is not None else MIN_SIDE)


def recognize(bgr: np.ndarray, s: "Settings | None" = None) -> OcrResult:
    return recognizer(s).run(bgr)
