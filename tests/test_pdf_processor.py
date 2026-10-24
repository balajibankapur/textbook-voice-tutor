from pathlib import Path

import pytest

from ingestion.pdf_processor import ExtractionSource, extract_pages, page_count


def test_extract_pages_marks_text_pages_as_embedded(text_pdf: Path) -> None:
    results = extract_pages(text_pdf)
    assert len(results) == 2
    assert all(r.source == ExtractionSource.EMBEDDED for r in results)
    assert any("Photosynthesis" in r.text for r in results)
    assert any("chloroplast" in r.text for r in results)


def test_extract_pages_preserves_page_numbers(text_pdf: Path) -> None:
    results = extract_pages(text_pdf)
    assert [r.page_number for r in results] == [1, 2]


def test_extract_pages_flags_image_only_as_needs_ocr(image_only_pdf: Path) -> None:
    results = extract_pages(image_only_pdf)
    assert len(results) == 1
    assert results[0].source == ExtractionSource.NEEDS_OCR
    assert results[0].text == ""


def test_extract_pages_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pages(tmp_path / "nope.pdf")


def test_page_count(text_pdf: Path) -> None:
    assert page_count(text_pdf) == 2
