"""Shared fixtures: synthetic pages, a temporary ocrmd root, and a fake LLM provider."""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocrmd.config import load_settings          # noqa: E402
from ocrmd.llm import LLMResult, PageContext, Transcription  # noqa: E402
from ocrmd.store import Store                  # noqa: E402

LONG = ("it is a truth universally acknowledged that a single man in possession of a good fortune "
        "must be in want of a wife however little known the feelings or views of such a man may be")


def text_page(lines: list[str] | None = None, size=(1400, 1800), upside_down: bool = False) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, (245, 242, 235))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 40)
    except OSError:  # non-mac CI: PIL's bitmap font is enough for geometry tests
        f = ImageFont.load_default()
    y = 200
    for line in (lines or LONG.split(" however ")[0].split(" that ") * 6):
        d.text((120, y), line, font=f, fill="black"); y += 70
    a = np.array(img)[:, :, ::-1].copy()
    return cv2.rotate(a, cv2.ROTATE_180) if upside_down else a


def on_desk(page: np.ndarray, margin: int = 90, desk=(60, 60, 60)) -> np.ndarray:
    """Put a page on a background with a margin, the way a real capture sees it."""
    h, w = page.shape[:2]
    desk_img = np.full((h + 2 * margin, w + 2 * margin, 3), desk, dtype=np.uint8)
    desk_img[margin:margin + h, margin:margin + w] = page
    return desk_img


def with_hand(frame: np.ndarray, cx: float = 0.55, cy: float = 0.6, size: float = 0.22) -> np.ndarray:
    """A skin-toned blob with a few 'fingers', over the page."""
    out = frame.copy()
    h, w = out.shape[:2]
    skin = (80, 120, 170)  # BGR: a hand under a desk lamp, darker than white paper
    cv2.ellipse(out, (int(w * cx), int(h * cy)), (int(w * size * 0.5), int(h * size * 0.6)), 0, 0, 360, skin, -1)
    for i in range(4):
        x = int(w * (cx - size * 0.35 + i * size * 0.25))
        cv2.rectangle(out, (x, int(h * (cy - size * 0.9))), (x + int(w * 0.03), int(h * cy)), skin, -1)
    return out


class FakeProvider:
    """Deterministic stand-in for the LLM: returns whatever transcription the test hands it."""
    name = "anthropic"
    model = "fake-model"

    def __init__(self, markdown: str | None = None, page_number: str | None = None, legible: bool = True,
                 chapter: str | None = None, raise_exc: Exception | None = None, model: str = "fake-model"):
        self.markdown, self.page_number, self.legible, self.chapter, self.raise_exc = markdown, page_number, legible, chapter, raise_exc
        self.model = model
        self.calls: list[tuple[str, object]] = []
        self.focus: list[object] = []          # the FocusViews (or None) each call received

    def transcribe(self, image_bgr, ocr_hint, hint=None, focus=None) -> LLMResult:
        self.calls.append((ocr_hint, hint))
        self.focus.append(focus)
        if self.raise_exc:
            raise self.raise_exc
        md = self.markdown if self.markdown is not None else ocr_hint
        tr = Transcription(kind="book_page", markdown=md, page_number=self.page_number, legible=self.legible,
                           illegible_reason=None if self.legible else "blurry", context=PageContext(chapter=self.chapter))
        return LLMResult(transcription=tr, provider=self.name, model=self.model, request_id="msg_test",
                         usage={"input_tokens": 10, "output_tokens": 5}, latency_ms=1)


@pytest.fixture
def root(tmp_path: Path):
    s = load_settings(root=tmp_path, env={})
    s.ensure_dirs()
    return s


@pytest.fixture
def store(root):
    return Store(root.db_path)
