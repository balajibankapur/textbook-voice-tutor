"""Open a PDF, extract embedded text per page, and decide which pages need OCR.

Produces a list of PageExtraction records that downstream OCR / normalization
steps consume. This module does NOT run OCR itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pymupdf

from ingestion.text_normalizer import is_text_usable


class ExtractionSource(str, Enum):
    EMBEDDED = "embedded"
    NEEDS_OCR = "needs_ocr"


@dataclass
class PageExtraction:
    page_number: int  # 1-indexed, matches the PDF
    source: ExtractionSource
    text: str  # empty when NEEDS_OCR
    char_count: int
    alpha_char_count: int


def extract_pages(pdf_path: Path | str, min_alpha_chars: int = 50) -> list[PageExtraction]:
    """Open a PDF and classify each page as having usable embedded text or needing OCR."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    results: list[PageExtraction] = []
    with pymupdf.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            raw_text = page.get_text() or ""
            alpha = sum(1 for c in raw_text if c.isalpha())
            if is_text_usable(raw_text, min_alpha_chars=min_alpha_chars):
                results.append(
                    PageExtraction(
                        page_number=idx,
                        source=ExtractionSource.EMBEDDED,
                        text=raw_text,
                        char_count=len(raw_text),
                        alpha_char_count=alpha,
                    )
                )
            else:
                results.append(
                    PageExtraction(
                        page_number=idx,
                        source=ExtractionSource.NEEDS_OCR,
                        text="",
                        char_count=len(raw_text),
                        alpha_char_count=alpha,
                    )
                )
    return results


def page_count(pdf_path: Path | str) -> int:
    with pymupdf.open(Path(pdf_path)) as doc:
        return doc.page_count
