"""Page detection (bright region on dark desk) + rotated-rect warp, spread split, orientation, deskew, binarise."""
from __future__ import annotations
import cv2, numpy as np


MIN_PAGE_WIDTH = 900   # below this, body text is too few pixels tall for reliable OCR
PIPELINE_VERSION = "preprocess@3"   # bump when crops/rotation/splitting change: page ids derive from the output


def quality(gray: np.ndarray) -> dict:
    return {"blur": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "glare": round(float((gray > 250).mean()), 3)}


def find_page_rect(img: np.ndarray):
    """The sheet of paper in the frame, as a rotated rect, or None if nothing dominates.

    Merges every comparably-sized bright blob rather than taking the largest: a book with a deep
    gutter shadow thresholds into two separate blobs, and keeping only the biggest silently drops a
    whole page from the scan.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thr = cv2.morphologyEx(cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cv2.contourArea(c) for c in cnts)
    if biggest < 0.12 * img.shape[0] * img.shape[1]:
        return None
    peers = [c for c in cnts if cv2.contourArea(c) >= 0.45 * biggest]
    return cv2.minAreaRect(np.vstack(peers))


def find_page_quad(img: np.ndarray) -> np.ndarray | None:
    """The detected page as 4 corner points in frame coordinates, for the preview overlay."""
    rect = find_page_rect(img)
    return None if rect is None else cv2.boxPoints(rect)


def clipped_edges(quad: np.ndarray, shape: tuple[int, ...], margin: int = 6,
                  img: np.ndarray | None = None, ink_frac: float = 0.02) -> list[str]:
    """Frame edges where the page is being cut *and text is actually being lost*.

    Paper touching the frame border is common and usually harmless — it is the blank margin that
    runs off. Reporting that would cry wolf on nearly every capture, so when the frame is available
    the edge strip is checked for ink and only edges carrying text are reported.
    """
    h, w = shape[:2]
    xs, ys = quad[:, 0], quad[:, 1]
    touching = []
    if xs.min() <= margin: touching.append("left")
    if xs.max() >= w - 1 - margin: touching.append("right")
    if ys.min() <= margin: touching.append("top")
    if ys.max() >= h - 1 - margin: touching.append("bottom")
    if img is None or not touching:
        return touching
    band = max(6, int(0.012 * max(h, w)))
    strips = {"left": img[:, :band], "right": img[:, -band:], "top": img[:band, :], "bottom": img[-band:, :]}
    out = []
    for edge in touching:
        strip = strips[edge]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
        # ink = markedly darker than the paper around it, ignoring the dark desk outside the sheet
        paper = np.percentile(gray, 80)
        if paper < 60:                       # strip is mostly desk, not page
            continue
        if float(np.mean(gray < paper * 0.6)) >= ink_frac:
            out.append(edge)
    return out


SPREAD_RATIO = 1.15   # width/height above which a crop is treated as a two-page spread


def estimated_page_width(img: np.ndarray, rect) -> int:
    """Pixels across a single page of the detected sheet, so framing can be judged before capturing.

    Deliberately mirrors what `process` will do rather than guessing from the quad: a two-page
    spread and a single portrait page have almost the same aspect ratio, so the sheet's shape alone
    cannot tell them apart — only its shape *after* the text has been turned upright can.
    """
    crop = warp_rect(img, rect)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if line_direction_score(gray) < line_direction_score(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)):
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    h, w = crop.shape[:2]
    return int(w / 2) if w / max(h, 1) >= SPREAD_RATIO else int(w)


def warp_rect(img: np.ndarray, rect) -> np.ndarray:
    (cx, cy), (w, h), ang = rect
    w, h = int(w), int(h)
    box = cv2.boxPoints(rect).astype("float32")
    # order: tl, tr, br, bl relative to the rect's own axes
    s, d = box.sum(1), np.diff(box, axis=1).ravel()
    src = np.array([box[np.argmin(s)], box[np.argmin(d)], box[np.argmax(s)], box[np.argmax(d)]], dtype="float32")
    ww = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    hh = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    dst = np.array([[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (ww, hh))


def split_spread(img: np.ndarray) -> list[np.ndarray]:
    """If wide (book spread), split at the gutter: darkest vertical band in the middle third."""
    h, w = img.shape[:2]
    if w / h < SPREAD_RATIO:
        return [img]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col = cv2.GaussianBlur(gray, (1, 51), 0).mean(axis=0)
    lo, hi = w // 3, 2 * w // 3
    x = lo + int(np.argmin(col[lo:hi]))
    return [img[:, :x], img[:, x:]]


_DICT: set[str] | None = None


def _dictionary() -> set[str]:
    global _DICT
    if _DICT is None:
        try:
            _DICT = {w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore") if len(w.strip()) >= 3}
        except OSError:
            _DICT = set()
    return _DICT


def line_direction_score(gray: np.ndarray, size: int = 800, p_min: int = 6, p_max: int = 90) -> float:
    """Fraction of the row-profile's power that sits at text-line spacing.

    A page of text is *periodic* across the lines, so its profile has energy at periods of roughly
    6-90 px once normalised. Plain profile variance was tried first and is not safe here: a single
    strong band — a gutter shadow, a page edge — outscores a whole page of real text lines, which
    is enough to rotate a spread on its side and lose the two-page split entirely.
    """
    g = cv2.resize(gray, (size // 2, size), interpolation=cv2.INTER_AREA)
    thr = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    prof = thr.mean(axis=1).astype(np.float64)
    prof -= cv2.blur(prof.reshape(-1, 1), (1, 51)).ravel()          # detrend: drop low frequency
    spec = np.abs(np.fft.rfft(prof * np.hanning(len(prof)))) ** 2
    freqs = np.arange(len(spec))
    band = (freqs >= size / p_max) & (freqs <= size / p_min)
    return float(spec[band].sum() / max(spec.sum(), 1e-9))


def _dictionary_hits(bgr: np.ndarray, target: int = 2000) -> int:
    """Readable words found by a fast OCR pass. The fast recogniser does not auto-orient (the accurate
    one does), which is what makes it usable as an upright-vs-upside-down test."""
    from . import ocr
    if "apple_vision" not in ocr.available_engines():
        return 0
    import re
    h, w = bgr.shape[:2]
    scale = target / max(h, w)
    im = cv2.resize(bgr, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA) if scale != 1 else bgr
    res = ocr.vision_lines(im, level="fast")
    D = _dictionary()
    toks = [x.lower() for b in res for x in re.findall(r"[A-Za-z]{4,}", b.text)]
    return sum(t in D for t in toks) if D else len(toks)


def ink_asymmetry(gray: np.ndarray) -> float:
    """Latin text puts more ink above each line's centre than below (capitals and ascenders outnumber
    descenders). Positive means upright. Works on pages far too small for any OCR engine to read,
    which is exactly when the dictionary probe has nothing to say."""
    thr = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    rows = thr.mean(axis=1)
    band_on = rows > rows.mean()
    scores, start = [], None
    for i, v in enumerate(list(band_on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            band = rows[start:i]
            if len(band) >= 4:
                centroid = (band * np.arange(len(band))).sum() / max(band.sum(), 1e-6)
                scores.append(centroid / len(band) - 0.5)
            start = None
    return -float(np.mean(scores)) if scores else 0.0


def orientation_probe(bgr: np.ndarray) -> int:
    """Rotation (0/90/180/270, clockwise) that makes the text upright.

    Three stages, strongest evidence first, because no single test is reliable: the projection profile
    knows whether text runs horizontally but not which way up; the OCR probe knows which way up but
    returns nothing on pages whose type is too small to read; ink asymmetry always has an opinion but
    is the weakest. Geometry narrows to two candidates, OCR decides when it can, ink breaks the tie.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    horizontal = line_direction_score(gray)
    vertical = line_direction_score(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE))
    candidates = (0, 180) if horizontal >= vertical else (90, 270)
    hits = {rot: _dictionary_hits(apply_rotation(bgr, rot)) for rot in candidates}
    best, other = sorted(candidates, key=lambda r: -hits[r])
    # require a real margin: a one-word lead on a barely-readable page is noise
    if hits[best] >= 3 and hits[best] >= hits[other] + 2:
        return best
    # the page is too small to read: fall back to the shape of the ink itself
    asym = {rot: ink_asymmetry(cv2.cvtColor(apply_rotation(bgr, rot), cv2.COLOR_BGR2GRAY)) for rot in candidates}
    return max(candidates, key=lambda r: asym[r])


def apply_rotation(img: np.ndarray, rot: int) -> np.ndarray:
    return {0: img, 90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 180: cv2.rotate(img, cv2.ROTATE_180),
            270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)}[rot]


def deskew_angle(gray: np.ndarray) -> float:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle += 90
    if angle > 45: angle -= 90
    return float(angle) if abs(angle) < 10 else 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.2:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def process(img: np.ndarray, min_page_width: int = MIN_PAGE_WIDTH) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """-> list of (colour crop for LLM, binarised for OCR, meta); 2 entries for a book spread (left, right)."""
    base: dict = {"pipeline": PIPELINE_VERSION, "page_detected": False}
    rect = find_page_rect(img)
    clipped: list[str] = []
    if rect is not None:
        clipped = clipped_edges(cv2.boxPoints(rect), img.shape, img=img)
        img = warp_rect(img, rect)
        base["page_detected"] = True
    base["clipped_edges"] = clipped
    rot = orientation_probe(img)
    img = apply_rotation(img, rot)
    base["orientation"] = rot
    parts = split_spread(img)
    out = []
    for i, part in enumerate(parts):
        meta = dict(base, part=(["left", "right"][i] if len(parts) == 2 else None))
        gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
        angle = deskew_angle(gray)
        part, gray = rotate(part, angle), rotate(gray, angle)
        meta["deskew"] = round(angle, 2)
        den = cv2.fastNlMeansDenoising(gray, h=7)
        binar = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        meta.update(quality(gray))
        meta["page_width_px"] = int(part.shape[1])
        if clipped:
            meta["advice"] = (f"page runs past the {' and '.join(clipped)} edge of the frame — text is being "
                              "cut off; pull the camera back or re-centre the book")
        elif part.shape[1] < min_page_width:
            meta["advice"] = (f"page is only {part.shape[1]}px wide; body text needs ~{min_page_width}px+ "
                              "— move the camera closer or fill the frame with one page")
        out.append((part, binar, meta))
    return out
