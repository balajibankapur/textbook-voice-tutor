"""Fast tests for AnswerGenerator using httpx MockTransport."""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from llm.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
    GeneratedAnswer,
)
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus


def _chunk(chunk_id: int = 1, page: int = 1, text: str = "Plants have roots.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chapter_id="ch1",
        page_start=page,
        page_end=page,
        section_title=None,
        chunk_text=text,
        score=0.5,
        ocr_quality_status=OcrQualityStatus.OK,
    )


def _resp(content: dict | str, status: int = 200) -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    body = {
        "id": "gen-test",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }
    return httpx.Response(status, json=body) if status == 200 else httpx.Response(status, text=text)


def _generator(handler: Callable[[httpx.Request], httpx.Response], **kw) -> AnswerGenerator:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return AnswerGenerator(api_key="sk-test", http_client=http, max_retries=kw.pop("max_retries", 2), **kw)


def test_happy_path_builds_generated_answer() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": True,
                "answer": "Plants make food through photosynthesis.",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
            }
        )

    g = _generator(handler)
    out = g.generate(question="How do plants make food?", passages=[_chunk()])
    assert isinstance(out, GeneratedAnswer)
    assert out.answerable
    assert "photosynthesis" in out.answer
    assert out.supporting_page_numbers == [1]
    assert out.supporting_passage_ids == [1]


def test_drops_citations_outside_evidence_set() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": True,
                "answer": "Plants have roots.",
                "supporting_page_numbers": [1, 99],   # 99 not in evidence
                "supporting_passage_ids": [1, 777],   # 777 not in evidence
            }
        )

    g = _generator(handler)
    out = g.generate(question="roots?", passages=[_chunk(chunk_id=1, page=1)])
    assert out.supporting_page_numbers == [1]
    assert out.supporting_passage_ids == [1]


def test_fills_in_pages_when_llm_omits_them() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": True,
                "answer": "Roots absorb water.",
                "supporting_page_numbers": [],
                "supporting_passage_ids": [],
            }
        )

    g = _generator(handler)
    out = g.generate(
        question="roots?",
        passages=[_chunk(chunk_id=1, page=3), _chunk(chunk_id=2, page=4)],
    )
    assert out.supporting_page_numbers == [3, 4]


def test_rejects_answerable_true_with_empty_answer() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": True,
                "answer": "   ",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
            }
        )

    g = _generator(handler)
    with pytest.raises(AnswerGenerationError, match="empty"):
        g.generate(question="x?", passages=[_chunk()])


def test_passes_answerable_false_through() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": False,
                "answer": "The chapter does not cover this.",
                "supporting_page_numbers": [],
                "supporting_passage_ids": [],
            }
        )

    g = _generator(handler)
    out = g.generate(question="who was Newton?", passages=[_chunk()])
    assert not out.answerable
    assert "chapter" in out.answer.lower()


def test_sends_bearer_and_model() -> None:
    seen: dict = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content.decode())
        return _resp(
            {
                "answerable": True,
                "answer": "ok",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
            }
        )

    g = _generator(handler, model="anthropic/claude-haiku-5-5")
    g.generate(question="x?", passages=[_chunk()])
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "anthropic/claude-haiku-5-5"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_retries_on_502_then_succeeds() -> None:
    import llm.answer_generator as mod

    mod.time.sleep = lambda _s: None  # type: ignore[assignment]

    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(502, text="bad gateway")
        return _resp(
            {
                "answerable": True,
                "answer": "fine",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
            }
        )

    g = _generator(handler, max_retries=2)
    out = g.generate(question="x?", passages=[_chunk()])
    assert out.answer == "fine"
    assert calls["n"] == 2


def test_does_not_retry_on_400() -> None:
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    g = _generator(handler, max_retries=3)
    with pytest.raises(AnswerGenerationError, match="400"):
        g.generate(question="x?", passages=[_chunk()])
    assert calls["n"] == 1


def test_raises_on_invalid_json() -> None:
    def handler(req):
        return _resp("not json")

    g = _generator(handler)
    with pytest.raises(AnswerGenerationError, match="non-JSON"):
        g.generate(question="x?", passages=[_chunk()])


def test_raises_on_missing_required_field() -> None:
    def handler(req):
        return _resp({"answerable": True, "answer": "x"})

    g = _generator(handler)
    with pytest.raises(AnswerGenerationError, match="missing fields"):
        g.generate(question="x?", passages=[_chunk()])


def test_empty_passages_rejected() -> None:
    def handler(req):
        return _resp({"answerable": True, "answer": "x"})

    g = _generator(handler)
    with pytest.raises(AnswerGenerationError, match="no passages"):
        g.generate(question="x?", passages=[])


def test_empty_api_key_raises() -> None:
    with pytest.raises(AnswerGenerationError, match="API key"):
        AnswerGenerator(api_key="")


def test_captures_follow_up_question_when_present() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": True,
                "answer": "Climbers have weak stems and need support.",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
                "follow_up_question": "Would you like to know some examples of climbers?",
            }
        )

    g = _generator(handler)
    out = g.generate(question="what do climbers do?", passages=[_chunk()])
    assert out.follow_up_question == "Would you like to know some examples of climbers?"


def test_missing_follow_up_question_is_none() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": True,
                "answer": "ok",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
            }
        )

    g = _generator(handler)
    out = g.generate(question="x?", passages=[_chunk()])
    assert out.follow_up_question is None


def test_blank_follow_up_question_becomes_none() -> None:
    def handler(req):
        return _resp(
            {
                "answerable": True,
                "answer": "ok",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
                "follow_up_question": "   ",
            }
        )

    g = _generator(handler)
    out = g.generate(question="x?", passages=[_chunk()])
    assert out.follow_up_question is None


def test_system_prompt_tells_llm_not_to_add_inferences() -> None:
    seen = {}

    def handler(req):
        body = json.loads(req.content.decode())
        seen["system"] = body["messages"][0]["content"]
        return _resp(
            {
                "answerable": True,
                "answer": "ok",
                "supporting_page_numbers": [1],
                "supporting_passage_ids": [1],
            }
        )

    g = _generator(handler)
    g.generate(question="x?", passages=[_chunk()])
    system_lc = seen["system"].lower()
    # Core grounding + follow-up instructions must be present.
    assert "literally stated" in system_lc
    assert "follow_up_question" in seen["system"]
