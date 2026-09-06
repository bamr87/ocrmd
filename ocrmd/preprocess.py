"""Page detection (bright region on dark desk) + rotated-rect warp, spread split, orientation, deskew, binarise."""
from __future__ import annotations
import cv2, numpy as np


MIN_PAGE_WIDTH = 900   # below this, body text is too few pixels tall for reliable OCR
PIPELINE_VERSION = "preprocess@4"   # bump when crops/rotation/splitting change: page ids derive from the output


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


SPREAD_RATIO = 1.15   # width/height below which a crop is never considered a spread (cheap pre-filter)
# ...and how far the gutter column's detail must fall below the page body's. Calibrated on the 60
# split captures in this store against what OCR actually reads: for the 36 where splitting helps the
# depth runs a median 0.527, for the 24 where it hurts 0.225 (AUC 0.797), and 0.30 maximises the net
# expected correctly-read words (+2652 against +2161 for splitting everything and +0 for nothing).
GUTTER_DEPTH = 0.30
# ...with text on *both* sides of it: a binding separates two pages, a margin does not. Deliberately
# loose — uneven light across a spread is the commonest complaint on this rig and it can leave the far
# page carrying a quarter of the near page's ink; this has to reject an *empty* side, not a dim one.
GUTTER_BALANCE = 0.25
GUTTER_BAND = (1 / 3, 2 / 3)   # where in the sheet to look, so a page's own blank margin cannot win


def estimated_page_width(img: np.ndarray, rect) -> int:
    """Pixels across a single page of the detected sheet, so framing can be judged before capturing.

    Deliberately mirrors what `process` will do rather than guessing from the quad: a two-page
    spread and a single portrait page have almost the same aspect ratio, so the sheet's shape alone
    cannot tell them apart — only its shape *after* the text has been turned upright can, and then
    only if it actually has a binding. The gutter test has to be the same one, or the HUD promises
    900 px a page on a sheet the pipeline is about to keep whole (or the reverse).
    """
    crop = warp_rect(img, rect)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if line_direction_score(gray) < line_direction_score(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)):
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    h, w = crop.shape[:2]
    if w / max(h, 1) < SPREAD_RATIO:
        return int(w)
    x, depth = find_gutter(crop)
    # the *narrower* half when it really is a spread: that is the page that will read worst, and
    # framing advice is only useful if it warns about the page at risk rather than the comfortable one
    return int(min(x, w - x)) if depth >= GUTTER_DEPTH else int(w)


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


def ink_columns(img: np.ndarray) -> np.ndarray:
    """Per-column amount of fine detail — text — as a fraction of full scale.

    High-pass energy (the frame minus a heavily blurred copy of itself), not a threshold. Two things
    have to be true at once and no threshold manages both. It must ignore *exposure*: a desk lamp to
    one side puts a smooth brightness ramp across the sheet, and on a global threshold that ramp is
    the strongest signal in the image — which is exactly how the gutter search used to end up on the
    dark side of the page instead of on the binding. And it must survive *blur*: measured against a
    page softened until OCR reads a third of it, an adaptive threshold keeps 13 % of the signal and a
    Sobel magnitude 16 %, while the high pass keeps 39 % — and all three read blank paper as zero.
    Under a threshold, a spread with one page out of focus stops looking like a spread at all.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    detail = np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 12))
    prof = detail.mean(axis=0) / 255.0
    return cv2.GaussianBlur(prof.reshape(-1, 1), (1, 31), 0).ravel()


def find_gutter(img: np.ndarray) -> tuple[int, float]:
    """Where a book's binding falls, and how convincing it is: (column, depth).

    `depth` is how much emptier of ink that column is than the page body, 0 (as inked as the text) to
    1 (completely blank), and it is reported as 0 unless there is comparable text on *both* sides of
    the column. Emptiness alone is not a binding: a page with a ragged right margin, a wide figure or
    a column break has a near-empty stripe too, and cutting there turns one page into a page and an
    offcut. Two facing pages of a book carry comparable amounts of ink; a page and its own margin
    do not.
    """
    w = img.shape[1]
    prof = ink_columns(img)
    lo, hi = int(w * GUTTER_BAND[0]), int(w * GUTTER_BAND[1])
    if hi - lo < 2:
        return w // 2, 0.0
    x = lo + int(np.argmin(prof[lo:hi]))
    inner = prof[int(0.05 * w):int(0.95 * w)]
    body = float(np.median(inner)) if inner.size else 0.0
    if body <= 1e-6:
        return x, 0.0
    # Any column in the blank band is a valid place to cut, so take its middle rather than the
    # argmin. The band is the binding plus the two inner margins, and those are rarely the same
    # width, which puts the lowest point off to one side; the midpoint keeps the most clearance from
    # text on both pages and does not move with noise in the profile.
    edge = body * GUTTER_DEPTH
    left = right = x
    while left > lo and prof[left - 1] <= edge:
        left -= 1
    while right < hi - 1 and prof[right + 1] <= edge:
        right += 1
    mid = (left + right) // 2
    before, after = prof[int(0.05 * w):mid], prof[mid:int(0.95 * w)]
    if not before.size or not after.size:
        return mid, 0.0
    a, b = float(before.mean()), float(after.mean())
    if min(a, b) < GUTTER_BALANCE * max(a, b):
        return mid, 0.0
    return mid, (body - float(prof[x])) / body


def split_spread(img: np.ndarray) -> list[np.ndarray]:
    """Split a book spread at its gutter — and only if there is one.

    The gutter is found by *absence of ink*, not by darkness. Measured on the 60 split captures in
    this repository's store, the old darkest-column rule put the split line on ink 0.109 where the
    page body measures 0.117: it was cutting straight through text. The least-inked column measures
    0.012, and 40 of the 60 would have been split somewhere else. Under a side lamp the darkest
    column in the search band is simply the far edge of the band, which is why 18 of them split at
    almost exactly 1/3 or 2/3 of the sheet and produced a page and a 440-px offcut.

    The depth test is what stops a single page being cut in two at all, and it is calibrated rather
    than guessed. Every one of those 60 captures was read three ways — whole, and split at the
    detail minimum — and scored on expected correctly-read words (words x mean confidence, the only
    fair way to compare one page against two). Splitting genuinely helps 36 of them and hurts 24;
    `GUTTER_DEPTH` is the cut that maximises the net gain across the corpus. Where the trough is
    shallow the unsplit sheet reads both longer and far more confidently — 371 words at 0.88 whole
    against 294 at 0.67 split — and where it is deep, splitting wins: 1118 at 0.94 against 1031 at
    0.91.
    """
    h, w = img.shape[:2]
    if w / h < SPREAD_RATIO:
        return [img]
    x, depth = find_gutter(img)
    if depth < GUTTER_DEPTH:
        return [img]                      # wide, but no binding: one page photographed at an angle
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


ORIENT_TARGET = 2200   # px on the long side for the orientation probe; below this it stops working
ORIENT_MARGIN = 3      # dictionary hits the winner must lead by before it is believed


def _probe_scale(bgr: np.ndarray, target: int = ORIENT_TARGET) -> np.ndarray:
    """Downscale once, before rotating. Rotation and scaling commute, so rotating the full-size crop
    four times and resizing each copy does the same work four times over — on a 2800x1800 crop that
    is ~60 MB of copies and four large resizes per page, and it is enough extra pressure to take the
    process down inside another library's `cv2.resize`."""
    h, w = bgr.shape[:2]
    scale = target / max(h, w)
    if scale >= 1:
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _dictionary_hits(bgr: np.ndarray, target: int = ORIENT_TARGET, level: str = "accurate") -> int:
    """Real words an OCR pass finds. The orientation probe's only reliable signal.

    Two settings here were measured against 68 captures from this repository's store and neither is
    free to change. The **level** was `fast`, chosen because the fast recogniser does not auto-orient;
    but on real webcam pages it reads *nothing at all* — 0 hits at every rotation on the pages the
    cascade actually gets wrong — so it cannot decide anything. The accurate level only partly
    auto-orients and does return a decisive winner. And the **size** is what makes it work: dropping
    the long side from 2200 px to 1500 px agrees with the 2200-px reading on only 54 % of pages, and
    probing a central crop instead is worse still (34-37 %). There is no cheap version of this test.
    """
    from . import ocr
    if "apple_vision" not in ocr.available_engines():
        return 0
    import re
    h, w = bgr.shape[:2]
    scale = target / max(h, w)
    im = cv2.resize(bgr, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA) if scale != 1 else bgr
    res = ocr.vision_lines(im, level=level, languages=["en-US"])
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


def orientation_probe(bgr: np.ndarray, probe: bool = True) -> int:
    """Rotation (0/90/180/270, clockwise) that makes the text upright.

    **The OCR probe decides, over all four rotations, at both recogniser levels.** It used to be third
    in a cascade, run at one level over two candidates chosen by a projection-profile test — which is
    where the whole thing came apart. Measured against 68 captures from this store scored by what OCR can actually
    read from them:

        projection profile picks the right axis    57 %   (chance is 50 %)
        ink asymmetry, all four rotations          25 %   (chance is 25 %)
        ink asymmetry, given the right axis        53 %   (chance is 50 %)

    Every stage but the OCR probe is a coin toss on this corpus, and the cascade multiplied them: an
    expected 30 % end to end, with 10 % of the crops in the store left with their text running
    vertically. The projection profile is also near-tied — within 10 % — on 46 % of pages, so its
    narrowing was frequently arbitrary *and* binding, locking the one test that works out of the
    answer. It is now only a tie-break for when the probe abstains, which is the job it can do.

    The cost is four fast OCR passes, and four accurate ones on top for the pages the fast level
    cannot read — about 4 s for those on the development machine. `ocrmd reprocess` on a book pays it
    once per page, and there is no cheaper version of the accurate pass (see `_dictionary_hits`).

    **It ships off, and that is not a judgement about whether it works.** Turning it on makes enough
    extra Apple Vision calls to reliably segfault the process later, inside RapidOCR's ONNX detector,
    in OpenCV's libdispatch-based `parallel_for_`. That reproduces outside pytest, survives
    `cv2.setNumThreads(1)`, and disappears entirely with `ocr_engines = ["apple_vision"]` — it is
    Vision and ONNX Runtime in one process, not something this module can fix. So the default keeps
    the old, weak behaviour rather than trading a wrong crop for a dead process; set
    `orientation_probe = true` if you do not run rapidocr, and re-run `ocrmd reprocess`.
    """
    if probe:
        # Fast first, accurate second, and the order is not an optimisation. The *fast* recogniser
        # does not auto-orient, so on a page it can read the rotations separate cleanly — but it
        # reads nothing at all from a marginal webcam capture. The *accurate* one reads those, and
        # partly auto-orients, so on a clean page it returns the same words at every rotation and
        # decides nothing. Each is blind exactly where the other sees.
        small = _probe_scale(bgr)
        for level in ("fast", "accurate"):
            hits = {rot: _dictionary_hits(apply_rotation(small, rot), level=level) for rot in (0, 90, 180, 270)}
            ranked = sorted(hits, key=lambda r: -hits[r])
            best, runner = ranked[0], ranked[1]
            if hits[best] >= ORIENT_MARGIN and hits[best] >= hits[runner] + ORIENT_MARGIN:
                return best
    # The original cascade, which is also what `probe=False` gets: geometry narrows to two candidates,
    # a *fast* OCR pass decides between them, ink asymmetry breaks the tie. Every step of it is at or
    # near chance on real captures (the table above), but it is what shipped and it costs two Vision
    # calls rather than eight — so it stays the fallback rather than being replaced by something even
    # weaker. Skipping the OCR step here entirely was tried and is worse than the code it replaced.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    horizontal = line_direction_score(gray)
    vertical = line_direction_score(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE))
    candidates = (0, 180) if horizontal >= vertical else (90, 270)
    hits = {rot: _dictionary_hits(apply_rotation(bgr, rot), target=2000, level="fast")
            for rot in candidates}
    best, other = sorted(candidates, key=lambda r: -hits[r])
    if hits[best] >= 3 and hits[best] >= hits[other] + 2:
        return best
    asym = {rot: ink_asymmetry(cv2.cvtColor(apply_rotation(bgr, rot), cv2.COLOR_BGR2GRAY))
            for rot in candidates}
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


# The "flat" rendering: the page as a flatbed scanner would have delivered it. Constants measured on
# this store's 40 reference pages, scored the way `ocrmd ocr-bench` scores an engine (bag-of-words F1
# against the page's own LLM transcription) — see `legible`.
FLAT_STROKE = 15         # px, the morphological close that removes glyphs before the light is estimated
FLAT_SHARPEN = 0.8       # unsharp amount; 1.2 was measured worse (halos become ink at these scales)
FLAT_SIGMA = 1.6         # ...over this radius
FLAT_MIN_INK = 0.002     # ink share below which the crop is blank and the tone curve is left alone
FLAT_MIN_RANGE = 20      # ...as is a crop whose ink and paper are already this close together
FLAT_INK_PCT = 10        # where in the ink class black lands, and where in the paper class white does.
FLAT_PAPER_PCT = 40      # Not the medians: those clip half of each class flat, F1 0.693 against 0.741.


def background(gray: np.ndarray) -> np.ndarray:
    """The page's illumination: what is left when the glyphs are closed away and the rest is blurred."""
    k = max(31, (min(gray.shape) // 8) | 1)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FLAT_STROKE, FLAT_STROKE)))
    return cv2.GaussianBlur(closed, (k, k), 0)


def legible(bgr: np.ndarray) -> np.ndarray:
    """The page crop rendered to be *read* — by a person or by a model asked about one region of it.

    Three steps: divide out the illumination (page curl and lamp fall-off are what make a webcam page
    look nothing like a scan), an unsharp mask, and a tone curve that puts the ink percentile on black
    and the paper percentile on white. Deliberately no denoising — a bilateral filter here looks
    cleaner and measured 0.597 F1 against 0.734 without it. At webcam x-heights the "noise" it smooths
    away is the thin strokes.

    It is greyscale, not binary, and that is the difference from `<pid>.bin.png`. Adaptive thresholding
    has to decide, per pixel, whether a grey belongs to ink or paper; on webcam type at 8-10 px x-height
    that decision eats thin strokes and speckles the margins. Keeping the greys costs nothing to look at
    and measurably reads better: Apple Vision scores F1 0.741 on this against **0.246** on the `bin`
    view, and 0.744 on the untouched colour crop the engines are actually given. That last number is the
    point of the exercise — the flat rendering is for a person to read and for `pipeline.read_region` to
    enlarge, and it gets there without costing the words a rendering like `bin` throws away. It costs
    33 ms on a real crop (measured over all 194 in this store).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    flat = cv2.divide(gray, background(gray), scale=255)
    blur = cv2.GaussianBlur(flat, (0, 0), FLAT_SIGMA)
    return tone(cv2.addWeighted(flat, 1 + FLAT_SHARPEN, blur, -FLAT_SHARPEN, 0))


def tone(gray: np.ndarray) -> np.ndarray:
    """Put the ink on black and the paper on white, with the two classes found by Otsu rather than by a
    fixed percentile of the whole image.

    How much of a page is ink runs from about 1 % on a chapter opening to 15 % in a dense table, so a
    fixed percentile is sometimes the ink and sometimes the paper — and on the sparse page it lands on
    paper, which is the page that needed the help most.

    *Inside* each class the point is taken well off the middle. Mapping the class medians to black and
    white sounds right and measurably is not: it flattens half of every stroke and half of every
    highlight, and cost 0.048 F1 (0.693 against 0.741 for the percentiles below, 0.744 for the untouched
    colour crop).
    """
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    ink, paper = gray[gray < thr], gray[gray >= thr]
    if ink.size < FLAT_MIN_INK * gray.size or not paper.size:
        return gray
    lo, hi = float(np.percentile(ink, FLAT_INK_PCT)), float(np.percentile(paper, FLAT_PAPER_PCT))
    if hi - lo < FLAT_MIN_RANGE:
        return gray
    return np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def process(img: np.ndarray, min_page_width: int = MIN_PAGE_WIDTH, orientation: int | None = None,
            probe: bool = True) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """-> list of (colour crop for LLM, binarised for OCR, meta); 2 entries for a book spread (left, right).

    `orientation` skips the probe and rotates by the given angle instead. It exists for the focus
    bracket's variant frames: they are the same scene as the capture that has already been processed,
    the probe is the expensive step here (it runs an OCR pass), and — more importantly — a variant that
    probed to a different rotation than its own primary would produce a crop whose boxes cannot be
    compared with it at all. Leave it None for anything that decides a page's identity.
    """
    base: dict = {"pipeline": PIPELINE_VERSION, "page_detected": False}
    rect = find_page_rect(img)
    clipped: list[str] = []
    if rect is not None:
        clipped = clipped_edges(cv2.boxPoints(rect), img.shape, img=img)
        img = warp_rect(img, rect)
        base["page_detected"] = True
    base["clipped_edges"] = clipped
    rot = orientation_probe(img, probe) if orientation is None else orientation % 360
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
