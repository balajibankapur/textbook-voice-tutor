"""Fast tests for JevDecisionsClient using httpx MockTransport."""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from decision.jev_client import JevError
from decision.jev_decisions_client import ChoiceMatch, JevDecisionsClient
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus


def _chunk(text: str = "Plants have roots.", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1, chapter_id="c", page_start=page, page_end=page,
        section_title=None, chunk_text=text, score=0.5,
        ocr_quality_status=OcrQualityStatus.OK,
    )


def _client(handler: Callable[[httpx.Request], httpx.Response], **kw) -> JevDecisionsClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return JevDecisionsClient(api_key="sk-test", http_client=http, max_retries=kw.pop("max_retries", 0), **kw)


def _decisions_response(answers: dict) -> httpx.Response:
    return httpx.Response(200, json={"model": "typesafe/jev-1.14", "answers": answers})


# ---- decide (gate) --------------------------------------------------------


def test_decide_three_nouls_accepted_when_above_threshold() -> None:
    def handler(req):
        return _decisions_response(
            {
                "clear": {"type": "noul", "noul": 0.9},
                "relevant": {"type": "noul", "noul": 0.85},
                "sufficient": {"type": "noul", "noul": 0.95},
            }
        )

    c = _client(handler, noul_threshold=0.5)
    d = c.decide(question="x?", passages=[_chunk()])
    assert d.question_clear
    assert d.evidence_relevant
    assert d.evidence_sufficient
    assert d.accepted
    assert d.decision_confidence == pytest.approx((0.9 + 0.85 + 0.95) / 3)


def test_decide_rejects_when_any_noul_below_threshold() -> None:
    def handler(req):
        return _decisions_response(
            {
                "clear": {"type": "noul", "noul": 0.1},  # below threshold
                "relevant": {"type": "noul", "noul": 0.9},
                "sufficient": {"type": "noul", "noul": 0.9},
            }
        )

    c = _client(handler, noul_threshold=0.5)
    d = c.decide(question="?", passages=[_chunk()])
    assert d.question_clear is False
    assert d.evidence_relevant is True
    assert d.evidence_sufficient is True
    assert d.accepted is False


def test_decide_posts_three_questions_in_one_request() -> None:
    seen: dict = {}

    def handler(req):
        body = json.loads(req.content.decode())
        seen["url"] = str(req.url)
        seen["body"] = body
        return _decisions_response(
            {
                "clear": {"type": "noul", "noul": 0.9},
                "relevant": {"type": "noul", "noul": 0.9},
                "sufficient": {"type": "noul", "noul": 0.9},
            }
        )

    c = _client(handler)
    c.decide(question="x?", passages=[_chunk()])
    assert "/decisions" in seen["url"]
    assert set(seen["body"]["questions"].keys()) == {"clear", "relevant", "sufficient"}
    for q in seen["body"]["questions"].values():
        assert q["type"] == "noul"


def test_decide_raises_on_missing_answer() -> None:
    def handler(req):
        return _decisions_response({"clear": {"type": "noul", "noul": 0.9}})  # missing rel/suff

    c = _client(handler)
    with pytest.raises(JevError, match="malformed noul"):
        c.decide(question="?", passages=[_chunk()])


def test_decide_raises_on_wrong_answer_type() -> None:
    def handler(req):
        return _decisions_response(
            {
                "clear": {"type": "choice", "choice": "yes"},  # wrong type
                "relevant": {"type": "noul", "noul": 0.9},
                "sufficient": {"type": "noul", "noul": 0.9},
            }
        )

    c = _client(handler)
    with pytest.raises(JevError, match="not a noul"):
        c.decide(question="?", passages=[_chunk()])


def test_decide_http_error_raises() -> None:
    def handler(req):
        return httpx.Response(500, text="boom")

    c = _client(handler, max_retries=0)
    with pytest.raises(JevError, match="500"):
        c.decide(question="?", passages=[_chunk()])


# ---- verify ---------------------------------------------------------------


def test_verify_supported() -> None:
    def handler(req):
        return _decisions_response({"supported": {"type": "noul", "noul": 0.92}})

    c = _client(handler, noul_threshold=0.5)
    v = c.verify_answer(question="?", passages=[_chunk()], answer="Plants have roots.")
    assert v.supported is True
    assert v.confidence == pytest.approx(0.92)


def test_verify_unsupported_when_below_threshold() -> None:
    def handler(req):
        return _decisions_response({"supported": {"type": "noul", "noul": 0.2}})

    c = _client(handler, noul_threshold=0.5)
    v = c.verify_answer(question="?", passages=[_chunk()], answer="x")
    assert v.supported is False
    assert v.confidence == pytest.approx(0.2)


def test_verify_rejects_empty_answer() -> None:
    def handler(req):
        pytest.fail("should not be called")

    c = _client(handler)
    with pytest.raises(JevError, match="empty"):
        c.verify_answer(question="?", passages=[_chunk()], answer="   ")


# ---- match_choice --------------------------------------------------------


def test_match_choice_returns_chosen_key_with_probability() -> None:
    def handler(req):
        return _decisions_response(
            {
                "match": {
                    "type": "choice",
                    "choice": "c1",
                    "probabilities": {"c0": 0.1, "c1": 0.85, "none": 0.05},
                    "confidence": 0.9,
                }
            }
        )

    c = _client(handler)
    m = c.match_choice(
        state_text="student q",
        candidates={"c0": "A", "c1": "B"},
        instructions="pick",
    )
    assert isinstance(m, ChoiceMatch)
    assert m.chosen_key == "c1"
    assert m.probability == pytest.approx(0.85)
    assert m.confidence == pytest.approx(0.9)
    assert set(m.probabilities.keys()) >= {"c0", "c1", "none"}


def test_match_choice_includes_none_option_by_default() -> None:
    seen: dict = {}

    def handler(req):
        body = json.loads(req.content.decode())
        seen["criteria"] = body["questions"]["match"]["criteria"]
        return _decisions_response(
            {
                "match": {
                    "type": "choice", "choice": "none",
                    "probabilities": {"c0": 0.0, "none": 1.0}, "confidence": 1.0,
                }
            }
        )

    c = _client(handler)
    c.match_choice(state_text="s", candidates={"c0": "A"}, instructions="pick")
    assert "none" in seen["criteria"]


def test_match_choice_without_candidates_raises() -> None:
    def handler(req):
        pytest.fail("should not be called")

    c = _client(handler)
    with pytest.raises(JevError, match="at least one candidate"):
        c.match_choice(state_text="s", candidates={}, instructions="pick")


def test_empty_api_key_raises() -> None:
    with pytest.raises(JevError, match="API key"):
        JevDecisionsClient(api_key="")


def test_invalid_threshold_raises() -> None:
    with pytest.raises(JevError, match="threshold"):
        JevDecisionsClient(api_key="sk-test", noul_threshold=1.5)
