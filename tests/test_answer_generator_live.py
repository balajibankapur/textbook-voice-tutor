"""Live LLM smoke test — uses OPENROUTER_API_KEY to call the real model."""
from __future__ import annotations

import os

import pytest

from llm.answer_generator import AnswerGenerator
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus

pytestmark = pytest.mark.live_jev  # same gate: needs OPENROUTER_API_KEY


def _chunk(chunk_id: int, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chapter_id="ch",
        page_start=page,
        page_end=page,
        section_title=None,
        chunk_text=text,
        score=0.5,
        ocr_quality_status=OcrQualityStatus.OK,
    )


@pytest.fixture(scope="module")
def generator():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    gen = AnswerGenerator(
        api_key=key,
        model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
        max_retries=1,
    )
    try:
        yield gen
    finally:
        gen.close()


def test_live_answer_from_supportive_passage(generator) -> None:
    passages = [
        _chunk(
            1,
            "Photosynthesis is the process by which plants use sunlight, water, "
            "and carbon dioxide to make glucose. Chlorophyll absorbs the sunlight.",
            page=3,
        )
    ]
    out = generator.generate(
        question="How do plants make their food?", passages=passages
    )
    assert out.answerable
    assert out.answer
    assert 3 in out.supporting_page_numbers
    assert 1 in out.supporting_passage_ids


def test_live_refuses_when_evidence_mismatches(generator) -> None:
    passages = [
        _chunk(1, "Plants are classified into herbs, shrubs, and trees.", page=1)
    ]
    out = generator.generate(
        question="What were Newton's three laws of motion?", passages=passages
    )
    assert out.answerable is False
