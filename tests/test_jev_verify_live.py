"""Live Jev.verify_answer tests."""
from __future__ import annotations

import os

import pytest

from decision.jev_client import JevClient
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus

pytestmark = pytest.mark.live_jev


def _chunk(text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        chapter_id="ch",
        page_start=page,
        page_end=page,
        section_title=None,
        chunk_text=text,
        score=0.5,
        ocr_quality_status=OcrQualityStatus.OK,
    )


@pytest.fixture(scope="module")
def jev():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    client = JevClient(
        api_key=key,
        model=os.getenv("JEV_MODEL", "typesafe/jev-router"),
        max_retries=1,
    )
    try:
        yield client
    finally:
        client.close()


def test_live_supported_answer(jev) -> None:
    passage = _chunk(
        "Photosynthesis is the process by which plants use sunlight, water, "
        "and carbon dioxide to make glucose. Chlorophyll absorbs the sunlight."
    )
    answer = (
        "Plants make their food through photosynthesis. They use sunlight, water, "
        "and carbon dioxide, and chlorophyll absorbs the light."
    )
    v = jev.verify_answer(
        question="How do plants make food?", passages=[passage], answer=answer
    )
    assert v.supported is True


def test_live_unsupported_answer(jev) -> None:
    passage = _chunk("Plants have roots, stems, and leaves.")
    answer = (
        "Plants have roots, stems, and leaves. On Mars, plants grow underground "
        "and use geothermal energy instead of sunlight."
    )
    v = jev.verify_answer(
        question="Where do plants grow?", passages=[passage], answer=answer
    )
    assert v.supported is False
