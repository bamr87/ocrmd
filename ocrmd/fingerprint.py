"""Hashes: exact (sha256) and perceptual (dHash) for dedup + page-turn detection."""
from __future__ import annotations
import hashlib
import cv2, numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dhash(gray: np.ndarray, size: int = 8) -> str:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return "".join("1" if b else "0" for b in bits.flatten())


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b, strict=False))


def signature(bgr: np.ndarray, size: tuple[int, int] = (96, 72)) -> np.ndarray:
    """Normalised small grey image for `similarity`."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two frames (1.0 = identical up to brightness/contrast).

    A dHash alone cannot tell two pages of the same book apart when the layout matches — measured on
    real captures, consecutive *different* pages sit at 0.6–0.98 while a re-capture of the same page
    scores 0.99+, so the hash is only a prefilter and this is the decision."""
    return float((signature(a) * signature(b)).mean())


def detail_signature(bgr: np.ndarray, width: int = 320) -> np.ndarray:
    """High-pass grey image: edges and text strokes without the lighting/layout."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h = max(1, int(g.shape[0] * width / g.shape[1]))
    g = cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 6)
    return (g - g.mean()) / (g.std() + 1e-6)


def detail_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of the high-pass content. Sensitive to a few pixels of shift, so it is only the
    fallback for dedup when no OCR text is available: measured on real captures, re-captures of the
    same page score 0.75-0.81, consecutive different pages <= 0.47."""
    return float((detail_signature(a) * detail_signature(b)).mean())
