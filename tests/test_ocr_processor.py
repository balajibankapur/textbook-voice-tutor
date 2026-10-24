from pathlib import Path

import pytest

from ingestion.ocr_processor import (
    OcrQuality,
    ocr_pdf_page,
    render_page_to_image,
    tesseract_available,
)


def test_render_page_to_image_returns_pil_image(image_only_pdf: Path) -> None:
    image = render_page_to_image(image_only_pdf, 1, dpi=150)
    assert image.width > 0
    assert image.height > 0


def test_ocr_pdf_page_raises_without_tesseract(image_only_pdf: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ingestion.ocr_processor.tesseract_available", lambda: False
    )
    with pytest.raises(RuntimeError, match="tesseract"):
        ocr_pdf_page(image_only_pdf, 1)


@pytest.mark.skipif(not tesseract_available(), reason="tesseract binary not installed")
def test_ocr_pdf_page_recovers_text(image_only_pdf: Path) -> None:
    result = ocr_pdf_page(image_only_pdf, 1)
    assert result.page_number == 1
    assert result.quality in {OcrQuality.OK, OcrQuality.LOW}
    assert "image" in result.text.lower() or "page" in result.text.lower()
