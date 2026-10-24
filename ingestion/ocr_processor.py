"""Render PDF pages to images and run Tesseract OCR on them.

Requires the `tesseract` binary on PATH; `TesseractNotFoundError` is surfaced
so callers can decide to flag the page for review instead of silently failing.
"""
from __future__ import annotations

import io
import logging
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300


class OcrQuality(str, Enum):
    OK = "ok"
    LOW = "low"
    FAILED = "failed"


@dataclass
class OcrResult:
    page_number: int
    text: str
    quality: OcrQuality
    mean_confidence: float | None  # 0-100 scale from Tesseract; None if unavailable


def tesseract_available() -> bool:
    """Return True if the tesseract CLI is installed and callable."""
    if shutil.which("tesseract") is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def render_page_to_image(pdf_path: Path | str, page_number: int, dpi: int = DEFAULT_DPI) -> Image.Image:
    """Render a 1-indexed page of the PDF as a PIL image at the given DPI."""
    with pymupdf.open(Path(pdf_path)) as doc:
        page = doc[page_number - 1]
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png")))


def ocr_image(image: Image.Image, lang: str = "eng") -> tuple[str, float | None]:
    """Run OCR on an image, returning (text, mean_confidence_or_None)."""
    text = pytesseract.image_to_string(image, lang=lang)
    mean_conf: float | None = None
    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data.get("conf", []) if c not in ("-1", -1, "")]
        if confs:
            mean_conf = sum(confs) / len(confs)
    except Exception as exc:  # pragma: no cover - best-effort stat
        logger.debug("tesseract confidence lookup failed: %s", exc)
    return text, mean_conf


def ocr_pdf_page(
    pdf_path: Path | str,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    low_confidence_threshold: float = 60.0,
    min_alpha_chars: int = 20,
) -> OcrResult:
    """Render then OCR a single page, classifying the result's quality."""
    if not tesseract_available():
        raise RuntimeError(
            "tesseract binary not found on PATH; install it (e.g. `brew install tesseract`) "
            "to run OCR."
        )

    try:
        image = render_page_to_image(pdf_path, page_number, dpi=dpi)
        text, mean_conf = ocr_image(image)
    except Exception as exc:
        logger.warning("OCR failed on page %s of %s: %s", page_number, pdf_path, exc)
        return OcrResult(page_number=page_number, text="", quality=OcrQuality.FAILED, mean_confidence=None)

    alpha = sum(1 for c in text if c.isalpha())
    if alpha < min_alpha_chars:
        quality = OcrQuality.LOW
    elif mean_conf is not None and mean_conf < low_confidence_threshold:
        quality = OcrQuality.LOW
    else:
        quality = OcrQuality.OK
    return OcrResult(page_number=page_number, text=text, quality=quality, mean_confidence=mean_conf)
