"""A PDF as a source: what a born-digital page already knows, before anything is guessed from pixels.

`ocrmd ingest file.pdf` used to rasterise each page and hand the picture to the webcam pipeline — page
detection, deskew, an OCR engine reading glyphs the file could have spelled out exactly. Everything a
PDF knows (its text layer with fonts and positions, where its images sit, its true page size) was
thrown away and then approximately recovered.

This module reads it instead. Each page yields its raster (the picture the model and the engines still
read, and the one the round trip is scored against), its text layer as OCR-shaped lines — one
reading with confidence 1.0, positioned exactly — and the boxes of its placed images, which become
figure blocks with no ink detection. A scanned-image PDF has no text layer and falls back to the
engines, so the cross-check is the same either way: the text layer is the *primary* reading and the
engines vote on it (`ocr.merge`), which is what makes a doctored or mis-encoded text layer show up as
disagreement instead of being trusted blind.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .ocr import Block

log = logging.getLogger("ocrmd.pdf")

ENGINE = "pdf_text"
DPI = 300                   # ~2480 px across a Letter page: above `ocr.MIN_SIDE`, so nothing is upscaled
MIN_TEXT_CHARS = 20         # fewer glyphs than this on a page is not a text layer, it is a watermark


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("pymupdf") is not None or importlib.util.find_spec("fitz") is not None


def _open(path: Path):
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf      # older releases
        except ImportError as e:
            raise RuntimeError(f"{path.name}: PDF support needs pymupdf — pip install 'ocrmd[pdf]'") from e
    return pymupdf.open(path)


def version() -> str:
    try:
        from importlib.metadata import version as v
        return f"{ENGINE}@{v('pymupdf')}"
    except Exception:
        return ENGINE


@dataclass
class PdfPage:
    """One page of a PDF, read rather than looked at."""
    name: str                                   # "<file>.pdf#p<n>"
    index: int                                  # 1-based
    image: np.ndarray                           # BGR raster at `DPI`
    width_pt: float
    height_pt: float
    blocks: list[Block] = field(default_factory=list)   # the text layer as normalised line boxes
    images: list[list[float]] = field(default_factory=list)   # placed images, normalised [x, y, w, h]
    drawings: int = 0                           # vector paths on the page

    @property
    def text_layer(self) -> bool:
        return sum(len(b.text) for b in self.blocks) >= MIN_TEXT_CHARS

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    def meta(self) -> dict:
        return {"width_pt": round(self.width_pt, 2), "height_pt": round(self.height_pt, 2),
                "text_layer": self.text_layer, "lines": len(self.blocks), "images": self.images,
                "drawings": self.drawings}


def _norm_box(rect, w: float, h: float) -> list[float]:
    x0, y0, x1, y1 = rect
    return [max(0.0, x0 / w), max(0.0, y0 / h), min(1.0, (x1 - x0) / w), min(1.0, (y1 - y0) / h)]


def read_page(page, name: str, index: int, dpi: int = DPI) -> PdfPage:
    rect = page.rect
    w, h = float(rect.width), float(rect.height)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = PdfPage(name=name, index=index, image=img, width_pt=w, height_pt=h)
    d = page.get_text("dict")
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            text = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
            if not text:
                continue
            out.blocks.append(Block(text=" ".join(text.split()), confidence=1.0, engine=ENGINE,
                                    bbox=_norm_box(ln["bbox"], w, h)))
    out.blocks.sort(key=lambda b: (round(b.bbox[1], 3), b.bbox[0]))
    try:
        for info in page.get_image_info():
            bb = _norm_box(info["bbox"], w, h)
            if bb[2] * bb[3] >= 0.002:                 # a bullet glyph as an image is not a figure
                out.images.append([round(v, 4) for v in bb])
    except Exception as e:                             # an odd PDF must not lose its text for a picture
        log.debug("%s: image info unavailable (%s)", name, e)
    try:
        out.drawings = len(page.get_drawings())
    except Exception:
        out.drawings = 0
    return out


def pages(path: Path, dpi: int = DPI) -> Iterator[PdfPage]:
    with _open(path) as pdf:
        for i, page in enumerate(pdf, 1):
            yield read_page(page, f"{path.name}#p{i}", i, dpi)


def page_count(path: Path) -> int:
    with _open(path) as pdf:
        return pdf.page_count
