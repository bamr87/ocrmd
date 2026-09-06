"""Put the page back: Markdown + LaTeX + SVG out, PDF/PNG back, and a number saying how close it is.

The pipeline reads a photograph into text. This reads the text back into a page, which is the only
way to see what the transcription *lost*: a heading that became a paragraph, a table that became
prose, a figure nobody mentioned. A side-by-side of the original crop and the reconstruction shows it
at a glance, and `fidelity()` puts a number on it.

Three outputs, because they are good at different things:

- **LaTeX** is the portable master. Absolutely positioned (`textpos`), so it is a facsimile rather
  than a reflow — the words land where they were photographed. It compiles anywhere a TeX
  distribution exists, and it is the format to hand to anyone who wants to edit the result.
- **SVG** is the same layout with no TeX needed, which matters because a TeX install is a large
  dependency and neither the development machine nor CI has one. It is also what makes the round trip
  testable: SVG renders to PNG through Chrome or Inkscape, and the result can be diffed against the
  original.
- **Markdown** is what the rest of ocrmd already writes, and stays the human-editable source.

Figures are vectorised *where that is honest*. A line drawing traces cleanly to SVG paths through
`cv2.findContours`, and the traced version is checked against the crop it came from — if it does not
reproduce it, the raster crop is embedded instead. A photograph is never traced. "Where possible"
has to mean "where it measurably worked", not "where it was attempted".

Nothing here invents content. The reconstruction is built only from the page's own words, its own
measured line boxes, and crops of its own image.
"""
from __future__ import annotations
import base64, html, logging, os, re, shutil, subprocess, tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .layout import Block as Figure, Layout, crop

log = logging.getLogger("ocrmd.reconstruct")

VECTOR_LEVELS = 24        # a region with more distinct grey levels than this is a photograph, not line art
VECTOR_FIDELITY = 0.80    # ...and a trace that reproduces its crop worse than this is not used
FONT_STACK = "'Times New Roman', 'Liberation Serif', Georgia, serif"


@dataclass
class Rendered:
    """One reconstruction and how close it came."""
    svg: str = ""
    latex: str = ""
    png: np.ndarray | None = None
    fidelity: float | None = None      # 0-1 against the original crop; None when nothing rendered it
    backend: str = ""                  # what rasterised it, or why nothing did
    figures: list[Figure] = field(default_factory=list)

    def describe(self) -> str:
        v = sum(1 for f in self.figures if f.kind == "vector")
        bits = [f"{len(self.figures)} figure(s) ({v} vector)"] if self.figures else []
        if self.fidelity is not None:
            bits.append(f"fidelity {self.fidelity:.2f} via {self.backend}")
        else:
            bits.append(self.backend or "not rasterised")
        return ", ".join(bits)


# --- figures ------------------------------------------------------------------------------------
def looks_like_line_art(region: np.ndarray) -> bool:
    """Few distinct tones and hard edges: a diagram. A photograph fails both."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    if gray.size < 64:
        return False
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    levels = int((hist > gray.size * 0.01).sum())
    return levels <= VECTOR_LEVELS // 8


def trace(region: np.ndarray, colour: str = "#111") -> str:
    """SVG paths for a line-art region, in its own 0-1 coordinate space."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    h, w = gray.shape[:2]
    thr = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for c in cnts:
        if cv2.contourArea(c) < 4:
            continue
        pts = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2)
        if len(pts) < 3:
            continue
        d = "M " + " L ".join(f"{x / w:.4f},{y / h:.4f}" for x, y in pts) + " Z"
        paths.append(f'<path d="{d}" fill="{colour}"/>')
    return "".join(paths)


def _rasterise_svg_fragment(paths: str, size: tuple[int, int]) -> np.ndarray | None:
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" height="{size[1]}" '
           f'viewBox="0 0 1 1" preserveAspectRatio="none">'
           f'<rect width="1" height="1" fill="#fff"/><g>{paths}</g></svg>')
    return rasterise(doc, size)


def vector_fidelity(region: np.ndarray, paths: str) -> float:
    """How well the trace reproduces the crop it came from. Unrenderable traces score 0."""
    h, w = region.shape[:2]
    shot = _rasterise_svg_fragment(paths, (w, h))
    if shot is None:
        return 0.0
    return image_similarity(region, shot)


def prepare_figures(img: np.ndarray, figs: list[Figure]) -> list[tuple[Figure, str]]:
    """Decide vector-or-raster per figure and return each with its SVG content."""
    out = []
    for f in figs:
        region = crop(img, f.bbox)
        content = ""
        if looks_like_line_art(region):
            paths = trace(region)
            if paths and vector_fidelity(region, paths) >= VECTOR_FIDELITY:
                f.kind, content = "vector", paths
        if not content:
            f.kind = "raster"
            ok, buf = cv2.imencode(".png", region)
            if ok:
                content = ('<image href="data:image/png;base64,'
                           + base64.b64encode(buf.tobytes()).decode()
                           + '" x="0" y="0" width="1" height="1" preserveAspectRatio="none"/>')
        out.append((f, content))
    return out


# --- SVG ----------------------------------------------------------------------------------------
def to_svg(lay: Layout, figures_svg: list[tuple[Figure, str]] | None = None) -> str:
    """The page as SVG, at its own pixel size, with every line where it was photographed."""
    w, h = lay.width, lay.height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}" font-family="{FONT_STACK}">',
             f'<rect width="{w}" height="{h}" fill="#fdfdfb"/>']
    for f, content in (figures_svg or []):
        x, y, fw, fh = f.bbox[0] * w, f.bbox[1] * h, f.bbox[2] * w, f.bbox[3] * h
        parts.append(f'<g transform="translate({x:.2f},{y:.2f}) scale({fw:.2f},{fh:.2f})">{content}</g>')
    body = lay.body_size * h
    for ln in lay.lines:
        x, y, lw, lh = ln.bbox[0] * w, ln.bbox[1] * h, ln.bbox[2] * w, ln.bbox[3] * h
        if not ln.text.strip() or lh <= 0:
            continue
        # Cap height is most of the box; the baseline sits near its bottom. textLength pins the line
        # to the width it actually occupied, which is what makes this a facsimile and not a re-set.
        # The size is clamped to a band around the page's median line, because the engines box a row
        # of text differently — one includes the descender space and one does not — and taking each
        # box at face value renders a single paragraph at three different sizes.
        lh = min(max(lh, body * 0.75), body * 1.6) if not ln.heading else max(lh, body)
        size = max(4.0, lh * 0.78)
        weight = ' font-weight="700"' if ln.heading else ""
        parts.append(f'<text x="{x:.2f}" y="{y + lh * 0.80:.2f}" font-size="{size:.2f}"{weight} '
                     f'textLength="{max(1.0, lw):.2f}" lengthAdjust="spacingAndGlyphs" '
                     f'fill="#111">{html.escape(ln.text)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# --- LaTeX --------------------------------------------------------------------------------------
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s: str) -> str:
    return "".join(_TEX.get(c, c) for c in s)


def to_latex(lay: Layout, title: str | None = None, figure_files: dict[int, str] | None = None) -> str:
    """A standalone LaTeX facsimile: the page at its true size, each line absolutely positioned.

    `textpos` in absolute units rather than a reflowed document, because the point is to reproduce the
    page. A reflowed LaTeX version of the Markdown is a different (also useful) artifact, and the
    Markdown itself already is one — this is the one that has to line up with the photograph.
    """
    w, h = lay.width, lay.height
    dpi = 150.0
    pw, ph = w / dpi, h / dpi          # inches
    L = [r"\documentclass{article}",
         rf"\usepackage[paperwidth={pw:.3f}in,paperheight={ph:.3f}in,margin=0in]{{geometry}}",
         r"\usepackage[absolute,overlay]{textpos}", r"\usepackage{graphicx}", r"\usepackage{xcolor}",
         r"\setlength{\TPHorizModule}{1in}", r"\setlength{\TPVertModule}{1in}",
         r"\pagestyle{empty}", r"\begin{document}"]
    if title:
        L.append(f"% {tex_escape(title)}")
    for i, f in enumerate(lay.figures):
        path = (figure_files or {}).get(i)
        if not path:
            continue
        L += [rf"\begin{{textblock}}{{{f.bbox[2] * pw:.4f}}}({f.bbox[0] * pw:.4f},{f.bbox[1] * ph:.4f})",
              rf"  \includegraphics[width={f.bbox[2] * pw:.4f}in,height={f.bbox[3] * ph:.4f}in]{{{path}}}",
              r"\end{textblock}"]
    for ln in lay.lines:
        if not ln.text.strip():
            continue
        pt = max(3.0, ln.bbox[3] * ph * 72 * 0.78)
        body = tex_escape(ln.text)
        if ln.heading:
            body = rf"\textbf{{{body}}}"
        L += [rf"\begin{{textblock}}{{{ln.bbox[2] * pw + 0.2:.4f}}}({ln.bbox[0] * pw:.4f},{ln.bbox[1] * ph:.4f})",
              rf"  \fontsize{{{pt:.1f}pt}}{{{pt * 1.05:.1f}pt}}\selectfont \noindent {body}",
              r"\end{textblock}"]
    L.append(r"\end{document}")
    return "\n".join(L)


# --- rendering ----------------------------------------------------------------------------------
def chrome() -> str | None:
    """A headless Chrome that can rasterise an SVG. Already a dependency of the repo's own driver."""
    for c in (os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome"), shutil.which("chromium")):
        if c and Path(c).exists():
            return c
    return None


def backends() -> list[str]:
    """Which rasterisers this machine has, best first."""
    out = []
    if shutil.which("inkscape"):
        out.append("inkscape")
    if chrome():
        out.append("chrome")
    if shutil.which("rsvg-convert"):
        out.append("rsvg-convert")
    return out


def rasterise(svg: str, size: tuple[int, int] | None = None, timeout: float = 60.0) -> np.ndarray | None:
    """SVG -> BGR image, through whichever converter is installed. None when none is."""
    avail = backends()
    if not avail:
        return None
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "page.svg", Path(td) / "page.png"
        src.write_text(svg, encoding="utf-8")
        for name in avail:
            try:
                if name == "inkscape":
                    cmd = ["inkscape", str(src), "--export-type=png", f"--export-filename={dst}"]
                    if size:
                        cmd += [f"--export-width={size[0]}", f"--export-height={size[1]}"]
                elif name == "rsvg-convert":
                    cmd = ["rsvg-convert", "-o", str(dst), str(src)]
                    if size:
                        cmd += ["-w", str(size[0]), "-h", str(size[1])]
                else:
                    cmd = [chrome(), "--headless", "--disable-gpu", "--no-sandbox",
                           f"--screenshot={dst}", "--hide-scrollbars", "--default-background-color=ffffff"]
                    if size:
                        cmd.append(f"--window-size={size[0]},{size[1]}")
                    cmd.append(src.as_uri())
                subprocess.run(cmd, check=True, timeout=timeout,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                img = cv2.imread(str(dst), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except (subprocess.SubprocessError, OSError) as e:
                log.debug("rasteriser %s failed: %s", name, e)
    return None


def compile_pdf(tex: str, out: Path, timeout: float = 120.0) -> Path | None:
    """LaTeX -> PDF, if a TeX engine is installed. None (with a log line) when none is."""
    engine = next((e for e in ("tectonic", "latexmk", "xelatex", "pdflatex") if shutil.which(e)), None)
    if engine is None:
        log.info("no TeX engine (tectonic/latexmk/xelatex/pdflatex); LaTeX written but not compiled")
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "page.tex"
        src.write_text(tex, encoding="utf-8")
        cmd = ([ "tectonic", "-o", td, str(src)] if engine == "tectonic" else
               ["latexmk", "-pdf", "-outdir=" + td, str(src)] if engine == "latexmk" else
               [engine, "-interaction=nonstopmode", "-output-directory", td, str(src)])
        try:
            subprocess.run(cmd, check=True, timeout=timeout, cwd=td,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.SubprocessError, OSError) as e:
            log.warning("%s failed to compile the page: %s", engine, e)
            return None
        pdf = Path(td) / "page.pdf"
        if not pdf.exists():
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(pdf.read_bytes())
        return out


# --- how close did it come? ----------------------------------------------------------------------
def image_similarity(a: np.ndarray, b: np.ndarray, width: int = 480) -> float:
    """Structural agreement of two renderings of one page, 0-1.

    Compared as *ink*: both sides are high-pass filtered and normalised before correlating, because
    the reconstruction is clean black on white and the original is a photograph under a desk lamp.
    Correlating raw pixels would score the lighting, not the layout.
    """
    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, (width, max(1, int(g.shape[0] * width / g.shape[1]))), interpolation=cv2.INTER_AREA)
        f = g.astype(np.float32)
        f = np.abs(f - cv2.GaussianBlur(f, (0, 0), 6))
        m = f.max()
        return f / m if m > 1e-6 else f
    x, y = prep(a), prep(b)
    h = min(x.shape[0], y.shape[0])
    x, y = x[:h], y[:h]
    xc, yc = x - x.mean(), y - y.mean()
    d = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    return max(0.0, float((xc * yc).sum() / d)) if d > 1e-9 else 0.0


def fidelity(original: np.ndarray, svg: str) -> tuple[float | None, str, np.ndarray | None]:
    """Render the SVG and score it against the original crop. (score, backend, image)."""
    avail = backends()
    if not avail:
        return None, "no SVG rasteriser (install inkscape or librsvg, or set CHROME)", None
    h, w = original.shape[:2]
    shot = rasterise(svg, (w, h))
    if shot is None:
        return None, f"rasteriser failed ({'+'.join(avail)})", None
    return image_similarity(original, shot), avail[0], shot


def rebuild(img: np.ndarray, lay: Layout, title: str | None = None) -> Rendered:
    """The whole round trip for one page: figures, SVG, LaTeX, render, score."""
    prepared = prepare_figures(img, lay.figures)
    svg = to_svg(lay, prepared)
    score, backend, shot = fidelity(img, svg)
    return Rendered(svg=svg, latex=to_latex(lay, title), png=shot, fidelity=score,
                    backend=backend, figures=[f for f, _ in prepared])


def strip_markdown_for_review(md: str) -> str:
    return re.sub(r"\[\^ocrmd-\d+\]:.*$", "", md, flags=re.M).strip()
