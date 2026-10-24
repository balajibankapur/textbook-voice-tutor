"""Fast tests for the evaluator — uses an in-process fake controller."""
from __future__ import annotations

import json
from pathlib import Path

from conversation.controller import TurnResult, TurnStatus, Timings
from decision.jev_client import JevDecision
from evaluation.evaluate_pipeline import (
    EvalResult,
    LabelledQuestion,
    _keyword_hit,
    _percentile,
    evaluate_one,
    load_dataset,
    summarize,
    write_csv,
)
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus


def _chunk(page: int = 1, text: str = "lorem") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=page,
        chapter_id="ch",
        page_start=page,
        page_end=page,
        section_title=None,
        chunk_text=text,
        score=0.5,
        ocr_quality_status=OcrQualityStatus.OK,
    )


class FakeController:
    def __init__(self, results: list[TurnResult]) -> None:
        self.results = results
        self.i = 0

    def run_turn(self, *, chapter_id, question, context) -> TurnResult:
        r = self.results[self.i]
        self.i += 1
        return r


def _turn(
    status: TurnStatus,
    message: str = "",
    pages: list[int] | None = None,
    total_ms: float = 100.0,
) -> TurnResult:
    passages = [_chunk(page=p, text=f"page {p}") for p in (pages or [])]
    return TurnResult(
        status=status,
        message=message,
        passages=passages,
        decision=JevDecision(
            question_clear=True,
            evidence_relevant=True,
            evidence_sufficient=True,
            decision_confidence=0.9,
            decision_reason="",
            raw={},
        ),
        attempts=1,
        timings=Timings(
            retrieval_ms=10.0, gate_ms=50.0, llm_ms=40.0, verify_ms=0.0, total_ms=total_ms
        ),
    )


# ---- unit-level helpers --------------------------------------------------


def test_percentile_handles_small_sets() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([10.0], 0.95) == 10.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_keyword_hit_is_any_match() -> None:
    assert _keyword_hit("Plants have roots and leaves.", ["roots"]) is True
    assert _keyword_hit("Plants have stems.", ["roots", "stems"]) is True
    assert _keyword_hit("Nothing to see.", ["roots", "stems"]) is False
    assert _keyword_hit("whatever", []) is True  # no keywords → vacuously pass


# ---- evaluate_one branches -----------------------------------------------


def test_evaluate_one_marks_answered_matching_expected() -> None:
    q = LabelledQuestion(
        id="q1",
        question="what are the parts of a plant?",
        category="answerable",
        expected_outcome="answered",
        expected_pages=[1, 2],
        expected_keywords=["roots"],
    )
    turn = _turn(
        TurnStatus.ANSWERED,
        message="Plants have roots, stems, and leaves.",
        pages=[1, 3],
    )
    ctrl = FakeController(results=[turn])
    r = evaluate_one(controller=ctrl, chapter_id="ch", q=q)
    assert r.actual_outcome == "answered"
    assert r.gate_correct is True
    assert r.retrieval_recall is True  # page 1 is in expected
    assert r.keyword_hit is True
    assert r.error is None


def test_evaluate_one_marks_refuse_matching_expected() -> None:
    q = LabelledQuestion(
        id="q2", question="who was Newton?", category="absent", expected_outcome="refuse"
    )
    turn = _turn(TurnStatus.REFUSE, message="not in chapter")
    ctrl = FakeController(results=[turn])
    r = evaluate_one(controller=ctrl, chapter_id="ch", q=q)
    assert r.actual_outcome == "refuse"
    assert r.gate_correct is True
    assert r.retrieval_recall is None  # no expected_pages


def test_evaluate_one_marks_gate_incorrect_on_mismatch() -> None:
    q = LabelledQuestion(
        id="q3", question="?", category="ambiguous", expected_outcome="clarify"
    )
    turn = _turn(TurnStatus.ANSWERED, message="unexpectedly answered")
    ctrl = FakeController(results=[turn])
    r = evaluate_one(controller=ctrl, chapter_id="ch", q=q)
    assert r.gate_correct is False


def test_evaluate_one_captures_crash_as_error() -> None:
    class BadController:
        def run_turn(self, **kw):
            raise RuntimeError("boom")

    q = LabelledQuestion(
        id="q4", question="x?", category="answerable", expected_outcome="answered"
    )
    r = evaluate_one(controller=BadController(), chapter_id="ch", q=q)
    assert r.actual_outcome == "error"
    assert r.error and "RuntimeError" in r.error
    assert r.gate_correct is False


# ---- summarize ----------------------------------------------------------


def _mk_result(**overrides) -> EvalResult:
    defaults = dict(
        id="x", category="answerable", question="?",
        expected_outcome="answered", actual_outcome="answered",
        retrieval_recall=True, gate_correct=True, keyword_hit=True,
        retrieval_ms=5.0, gate_ms=50.0, llm_ms=80.0, verify_ms=0.0, total_ms=140.0,
        clarify_depth=0, attempts=1, error=None,
    )
    defaults.update(overrides)
    return EvalResult(**defaults)


def test_summary_counts_and_rates() -> None:
    results = [
        _mk_result(id="a", gate_correct=True),
        _mk_result(id="b", gate_correct=False, actual_outcome="refuse"),
        _mk_result(
            id="c", category="absent", expected_outcome="refuse",
            actual_outcome="refuse", retrieval_recall=None, keyword_hit=None,
        ),
    ]
    s = summarize(results)
    assert s["total_questions"] == 3
    assert 0.0 <= s["gate_accuracy"] <= 1.0
    assert s["retrieval_recall"]["measured_on"] == 2
    assert s["answer_keyword_hit"]["measured_on"] == 2
    assert "p50_total" in s["latency_ms"]
    assert "answerable" in s["by_category"]
    assert "absent" in s["by_category"]


def test_summary_handles_empty() -> None:
    s = summarize([])
    assert s == {"total": 0}


# ---- dataset round-trip -------------------------------------------------


def test_load_dataset(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    path.write_text(
        json.dumps(
            {
                "chapter_id": "ch",
                "questions": [
                    {
                        "id": "q1",
                        "question": "x?",
                        "category": "answerable",
                        "expected_outcome": "answered",
                        "expected_pages": [1, 2],
                        "expected_keywords": ["foo"],
                    }
                ],
            }
        )
    )
    ds = load_dataset(path)
    assert ds.chapter_id == "ch"
    assert ds.questions[0].id == "q1"
    assert ds.questions[0].expected_pages == [1, 2]


def test_write_csv_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "eval.csv"
    write_csv([_mk_result(id="row1")], out)
    text = out.read_text()
    assert "row1" in text
    assert "actual_outcome" in text  # header
