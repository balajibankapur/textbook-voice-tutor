"""Fast tests for the Q&A generator using httpx MockTransport."""
from __future__ import annotations

import json

import httpx
import pytest

from ingestion.qa_generator import (
    QAGenerationError,
    QAGenerator,
    _ChunkIn,
    _parse_pairs_json,
    to_qa_rows,
)


def _generator(handler, **kw) -> QAGenerator:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return QAGenerator(api_key="sk-test", http_client=http, **kw)


def _ok_response(pairs: list[dict]) -> httpx.Response:
    body = {
        "id": "x",
        "choices": [
            {"message": {"content": json.dumps({"pairs": pairs})}}
        ],
    }
    return httpx.Response(200, json=body)


def _chunk(chunk_id: int = 1, text: str = "Plants have roots and stems.") -> _ChunkIn:
    return _ChunkIn(
        chunk_id=chunk_id,
        page_start=1,
        page_end=2,
        section_title="Intro",
        text=text,
    )


def test_generate_for_one_chunk_returns_pairs() -> None:
    def handler(req):
        return _ok_response(
            [
                {"question": "What are the parts of a plant?", "answer": "Roots, stems, leaves."},
                {"question": "What do roots do?", "answer": "Hold the plant and absorb water."},
            ]
        )

    g = _generator(handler)
    pairs = g.generate_for_chunks([_chunk()])
    assert len(pairs) == 2
    assert pairs[0].question.startswith("What are the parts")
    assert pairs[0].page_numbers == [1, 2]
    assert pairs[0].source_chunk_ids == [1]


def test_sends_bearer_and_model() -> None:
    seen: dict = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content.decode())
        return _ok_response([{"question": "q", "answer": "a"}])

    g = _generator(handler, model="gpt-4o-mini", pairs_per_chunk=3)
    g.generate_for_chunks([_chunk()])
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_skips_empty_chunks() -> None:
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return _ok_response([{"question": "q", "answer": "a"}])

    g = _generator(handler)
    pairs = g.generate_for_chunks(
        [_chunk(text="   "), _chunk(chunk_id=2, text="real content here")]
    )
    assert calls["n"] == 1  # only the non-empty chunk hits the API
    assert len(pairs) == 1


def test_tolerates_chunk_level_failure_continues_rest() -> None:
    sequence = [500, 200]
    bodies = [httpx.Response(500, text="oops"), _ok_response([{"question": "q", "answer": "a"}])]

    def handler(req):
        _ = sequence.pop(0)
        return bodies.pop(0)

    g = _generator(handler)
    pairs = g.generate_for_chunks([_chunk(), _chunk(chunk_id=2)])
    assert len(pairs) == 1  # second chunk succeeded, first was logged + skipped


def test_rejects_empty_api_key() -> None:
    with pytest.raises(QAGenerationError, match="API key"):
        QAGenerator(api_key="")


def test_invalid_pairs_per_chunk() -> None:
    with pytest.raises(QAGenerationError, match="pairs_per_chunk"):
        QAGenerator(api_key="sk-test", pairs_per_chunk=0)


def test_parse_pairs_json_handles_fence() -> None:
    content = (
        '```json\n{"pairs": [{"question": "q?", "answer": "a."}]}\n```'
    )
    out = _parse_pairs_json(content)
    assert out == [{"question": "q?", "answer": "a."}]


def test_parse_pairs_json_rejects_wrong_shape() -> None:
    with pytest.raises(QAGenerationError, match="pairs"):
        _parse_pairs_json('{"not_pairs": 1}')


def test_parse_pairs_json_rejects_invalid_items() -> None:
    content = '{"pairs": [{"question": "ok"}, {"question": "q", "answer": "a"}]}'
    out = _parse_pairs_json(content)
    # First item lacks 'answer' and is dropped; second passes through.
    assert out == [{"question": "q", "answer": "a"}]


def test_to_qa_rows_dedupes_by_question() -> None:
    from ingestion.qa_generator import GeneratedPair

    pairs = [
        GeneratedPair(question="What is a plant?", answer="A living thing.",
                      page_numbers=[1], source_chunk_ids=[1]),
        GeneratedPair(question="what is a plant?", answer="Different answer.",
                      page_numbers=[2], source_chunk_ids=[2]),
    ]
    rows = to_qa_rows(pairs, chapter_id="ch1")
    assert len(rows) == 1
    assert rows[0].chapter_id == "ch1"
    assert rows[0].answer == "A living thing."
