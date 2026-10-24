"""Fast tests for JevClient.verify_answer using httpx MockTransport."""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from decision.jev_client import JevClient, JevError, JevVerification
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus


def _chunk(text: str = "Plants have roots.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        chapter_id="ch1",
        page_start=1,
        page_end=1,
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


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> JevClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return JevClient(api_key="sk-test", http_client=http, max_retries=0)


def test_verify_supported_true() -> None:
    def handler(req):
        return _resp({"supported": True, "confidence": 0.9, "reason": "matches passage"})

    c = _client(handler)
    v = c.verify_answer(question="roots?", passages=[_chunk()], answer="Plants have roots.")
    assert isinstance(v, JevVerification)
    assert v.supported
    assert v.confidence == pytest.approx(0.9)
    assert "match" in v.reason


def test_verify_supported_false_with_reason() -> None:
    def handler(req):
        return _resp(
            {
                "supported": False,
                "confidence": 0.8,
                "reason": "mentions Mars but Mars is not in the passages",
            }
        )

    c = _client(handler)
    v = c.verify_answer(
        question="roots?",
        passages=[_chunk()],
        answer="Plants on Mars have special roots.",
    )
    assert not v.supported
    assert "Mars" in v.reason


def test_verify_rejects_empty_answer() -> None:
    def handler(req):
        pytest.fail("should not be called")

    c = _client(handler)
    with pytest.raises(JevError, match="empty"):
        c.verify_answer(question="x?", passages=[_chunk()], answer="   ")


def test_verify_missing_supported_field_raises() -> None:
    def handler(req):
        return _resp({"confidence": 0.5})  # no 'supported'

    c = _client(handler)
    with pytest.raises(JevError, match="missing fields"):
        c.verify_answer(question="x?", passages=[_chunk()], answer="ok")


def test_verify_invalid_json_raises() -> None:
    def handler(req):
        return _resp("not json at all")

    c = _client(handler)
    with pytest.raises(JevError, match="non-JSON"):
        c.verify_answer(question="x?", passages=[_chunk()], answer="ok")


def test_verify_uses_verify_system_prompt_and_includes_answer() -> None:
    seen: dict = {}

    def handler(req):
        body = json.loads(req.content.decode())
        seen["system"] = body["messages"][0]["content"]
        seen["user"] = body["messages"][1]["content"]
        return _resp({"supported": True, "confidence": 0.7, "reason": "ok"})

    c = _client(handler)
    c.verify_answer(
        question="roots?",
        passages=[_chunk()],
        answer="Plants have roots that absorb water.",
    )
    assert "supported" in seen["system"].lower()
    assert "verif" in seen["system"].lower()  # verify / verified / verifier
    assert "Tutor's answer" in seen["user"]
    assert "roots that absorb water" in seen["user"]
