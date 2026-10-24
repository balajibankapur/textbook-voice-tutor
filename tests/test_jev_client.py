"""Fast tests for JevClient using httpx MockTransport.

Covers: happy path parsing, retries on 5xx and timeouts, fail-closed on
invalid JSON, missing fields, HTTP 4xx. No network calls.
"""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from decision.jev_client import (
    JevClient,
    JevDecision,
    JevError,
)
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus


def _chunk(text: str = "Plants make food.", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        chapter_id="ch1",
        page_start=page,
        page_end=page,
        section_title="Intro",
        chunk_text=text,
        score=0.5,
        ocr_quality_status=OcrQualityStatus.OK,
    )


def _resp(content: dict | str, status: int = 200) -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    body = {
        "id": "gen-test",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "model": "typesafe/jev-router",
    }
    return httpx.Response(status, json=body) if status == 200 else httpx.Response(status, text=text)


def _client(handler: Callable[[httpx.Request], httpx.Response], **kw) -> JevClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return JevClient(api_key="sk-test", http_client=http, max_retries=kw.pop("max_retries", 2), **kw)


def test_happy_path_parses_json() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _resp(
            {
                "question_clear": True,
                "evidence_relevant": True,
                "evidence_sufficient": True,
                "confidence": 0.9,
                "reason": "passages cover the question",
            }
        )

    c = _client(handler)
    d = c.decide(question="What are parts of a plant?", passages=[_chunk()])
    assert isinstance(d, JevDecision)
    assert d.accepted
    assert d.decision_confidence == pytest.approx(0.9)
    assert "cover" in d.decision_reason


def test_strips_markdown_code_fence() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        content = (
            "```json\n"
            '{"question_clear": true, "evidence_relevant": false, '
            '"evidence_sufficient": false, "confidence": 0.4, "reason": "off topic"}\n'
            "```"
        )
        return _resp(content)

    c = _client(handler)
    d = c.decide(question="Who was Newton?", passages=[_chunk()])
    assert not d.accepted
    assert d.decision_reason == "off topic"


def test_sends_bearer_auth_and_model() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content.decode())
        return _resp(
            {
                "question_clear": True,
                "evidence_relevant": True,
                "evidence_sufficient": True,
                "confidence": 0.8,
                "reason": "ok",
            }
        )

    c = _client(handler, model="typesafe/jev-1.14")
    c.decide(question="x?", passages=[_chunk()])
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "typesafe/jev-1.14"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["temperature"] == 0.0


def test_retries_on_503_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return _resp(
            {
                "question_clear": True,
                "evidence_relevant": True,
                "evidence_sufficient": True,
                "confidence": 0.7,
                "reason": "ok",
            }
        )

    c = _client(handler, max_retries=3)
    # Shortcut sleeps so the test stays fast.
    import decision.jev_client as mod

    mod.time.sleep = lambda _s: None  # type: ignore[assignment]
    d = c.decide(question="x?", passages=[_chunk()])
    assert d.accepted
    assert calls["n"] == 3


def test_does_not_retry_on_400() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    c = _client(handler, max_retries=3)
    with pytest.raises(JevError, match="400"):
        c.decide(question="x?", passages=[_chunk()])
    assert calls["n"] == 1


def test_raises_on_invalid_json() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _resp("not actually json")

    c = _client(handler)
    with pytest.raises(JevError, match="non-JSON"):
        c.decide(question="x?", passages=[_chunk()])


def test_raises_on_missing_required_field() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _resp({"question_clear": True})  # missing the others

    c = _client(handler)
    with pytest.raises(JevError, match="missing fields"):
        c.decide(question="x?", passages=[_chunk()])


def test_timeout_fails_closed_after_retries() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    import decision.jev_client as mod

    mod.time.sleep = lambda _s: None  # type: ignore[assignment]
    c = _client(handler, max_retries=1)
    with pytest.raises(JevError, match="network error"):
        c.decide(question="x?", passages=[_chunk()])


def test_empty_api_key_raises() -> None:
    with pytest.raises(JevError, match="API key"):
        JevClient(api_key="")
