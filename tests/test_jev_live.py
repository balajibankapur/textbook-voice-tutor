"""Live Jev smoke test — only runs with a real OPENROUTER_API_KEY.

Marked `live_jev` and excluded from the default suite.
"""
from __future__ import annotations

import os

import pytest

from decision.jev_client import JevClient
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus

pytestmark = pytest.mark.live_jev


def _chunk(text: str, page: int = 1, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=page,
        chapter_id="ch",
        page_start=page,
        page_end=page,
        section_title=None,
        chunk_text=text,
        score=score,
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


def test_relevant_passage_accepted(jev) -> None:
    passage = _chunk(
        "Photosynthesis is the process by which plants use sunlight, water, and "
        "carbon dioxide to make glucose and release oxygen. Chlorophyll absorbs "
        "the light energy."
    )
    d = jev.decide(question="How do plants make their food?", passages=[passage])
    assert d.question_clear is True
    assert d.evidence_relevant is True
    assert d.evidence_sufficient is True


def test_off_topic_passage_rejected(jev) -> None:
    passage = _chunk(
        "Trees are classified by size into herbs, shrubs, and trees. Plants grow "
        "all over the Earth."
    )
    d = jev.decide(question="Who was Isaac Newton?", passages=[passage])
    assert d.evidence_relevant is False or d.evidence_sufficient is False


def test_vague_question_marked_unclear(jev) -> None:
    passage = _chunk("Plants have roots, stems, and leaves.")
    d = jev.decide(question="tell me more", passages=[passage])
    assert d.question_clear is False
    assert d.decision_reason  # Jev should propose a clarification sentence.
