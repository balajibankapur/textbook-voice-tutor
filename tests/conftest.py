"""Shared fixtures for ingestion tests."""
from __future__ import annotations

import os
from pathlib import Path

# Must run before sentence-transformers/faiss import. Disables tqdm's monitor
# thread, which otherwise races with faiss SWIG bindings and segfaults the
# test process on exit.
os.environ.setdefault("TQDM_DISABLE", "1")
try:
    import tqdm as _tqdm

    _tqdm.tqdm.monitor_interval = 0
except Exception:
    pass

import pymupdf
import pytest


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF with real embedded text."""
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Chapter 3\n\n"
        "Photosynthesis is the process by which green plants make food.\n"
        "They use sunlight, water, and carbon dioxide to produce glucose.",
        fontsize=12,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "The chloroplast contains chlorophyll which absorbs light.\n"
        "Oxygen is released as a by-product of photosynthesis.",
        fontsize=12,
    )
    out = tmp_path / "sample.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def image_only_pdf(tmp_path: Path) -> Path:
    """A 1-page PDF whose only content is a rasterized image of text.

    PyMuPDF returns no embedded text for this page, so the pipeline must
    classify it as NEEDS_OCR.
    """
    # Build the source page with text, render it to a PNG, then insert the
    # PNG into a fresh PDF so no text layer survives.
    source = pymupdf.open()
    src_page = source.new_page()
    src_page.insert_text((72, 72), "This page is an image.", fontsize=24)
    pix = src_page.get_pixmap(dpi=150)
    png_path = tmp_path / "page.png"
    pix.save(png_path)
    source.close()

    out_doc = pymupdf.open()
    page = out_doc.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(png_path))
    out = tmp_path / "image_only.pdf"
    out_doc.save(out)
    out_doc.close()
    return out
