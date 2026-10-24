"""End-to-end evaluator for the Jev-gated tutoring pipeline.

Loads a labelled question set, runs each question through the real pipeline
(retriever + Jev gate + optional LLM + optional verifier), and reports:
  - retrieval recall@k (did any top-K passage overlap an expected page?)
  - gate false-accept / false-reject (vs. expected outcome)
  - grounded answer accuracy (keyword presence on answered)
  - clarification success rate (did clarify fire on ambiguous questions?)
  - latency percentiles (retrieval, gate, LLM, verify, total)
  - error rate (FAIL_CLOSED counts)

Usage (from project root):
    .venv/bin/python -m evaluation.evaluate_pipeline \\
        --dataset evaluation/datasets/plants.json \\
        --out data/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config.settings import settings
from conversation.context_manager import ConversationContext
from conversation.controller import (
    ConversationController,
    TurnResult,
    TurnStatus,
)
from decision.decision_policy import DecisionPolicy
from decision.jev_decisions_client import JevDecisionsClient
from retrieval.qa_matcher import QAMatcher
from llm.answer_generator import AnswerGenerator
from retrieval.embedding_service import EmbeddingService
from retrieval.retriever import Retriever
from storage.metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)


# ---- dataset model -------------------------------------------------------


@dataclass
class LabelledQuestion:
    id: str
    question: str
    category: str
    expected_outcome: str  # "answered" | "refuse" | "clarify"
    expected_pages: list[int] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    chapter_id: str
    questions: list[LabelledQuestion]
    notes: str = ""


def load_dataset(path: Path) -> Dataset:
    data = json.loads(path.read_text())
    questions = [
        LabelledQuestion(
            id=q["id"],
            question=q["question"],
            category=q["category"],
            expected_outcome=q["expected_outcome"],
            expected_pages=list(q.get("expected_pages", [])),
            expected_keywords=list(q.get("expected_keywords", [])),
        )
        for q in data["questions"]
    ]
    return Dataset(
        chapter_id=data["chapter_id"],
        questions=questions,
        notes=data.get("notes", ""),
    )


# ---- per-question result model -------------------------------------------


@dataclass
class EvalResult:
    id: str
    category: str
    question: str
    expected_outcome: str
    actual_outcome: str
    retrieval_recall: bool | None  # None for non-answerable questions
    gate_correct: bool
    keyword_hit: bool | None  # None if no keywords specified or not answered
    retrieval_ms: float
    gate_ms: float
    llm_ms: float
    verify_ms: float
    total_ms: float
    clarify_depth: int
    attempts: int
    error: str | None = None


# ---- evaluator -----------------------------------------------------------


def _expected_to_status(expected: str) -> TurnStatus:
    return {
        "answered": TurnStatus.ANSWERED,
        "refuse": TurnStatus.REFUSE,
        "clarify": TurnStatus.CLARIFY,
    }[expected]


def _top_pages(result: TurnResult) -> set[int]:
    pages: set[int] = set()
    for p in result.passages:
        pages.update(range(p.page_start, p.page_end + 1))
    return pages


def _keyword_hit(answer_text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lc = answer_text.lower()
    # ANY keyword present passes; evaluator can require ALL by tightening later.
    return any(k.lower() in lc for k in keywords)


def _gate_matches_expected(result: TurnResult, expected: TurnStatus) -> bool:
    if result.status == expected:
        return True
    # Treat ANSWERED-from-clarify-expected as mismatch; everything else literal.
    return False


def evaluate_one(
    *,
    controller: ConversationController,
    chapter_id: str,
    q: LabelledQuestion,
) -> EvalResult:
    context = ConversationContext()
    t0 = time.perf_counter()
    try:
        result = controller.run_turn(
            chapter_id=chapter_id, question=q.question, context=context
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000
        logger.exception("evaluate_one crashed on %s", q.id)
        return EvalResult(
            id=q.id,
            category=q.category,
            question=q.question,
            expected_outcome=q.expected_outcome,
            actual_outcome="error",
            retrieval_recall=None,
            gate_correct=False,
            keyword_hit=None,
            retrieval_ms=0.0,
            gate_ms=0.0,
            llm_ms=0.0,
            verify_ms=0.0,
            total_ms=total_ms,
            clarify_depth=0,
            attempts=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    expected_status = _expected_to_status(q.expected_outcome)
    gate_correct = _gate_matches_expected(result, expected_status)

    recall: bool | None = None
    if q.expected_pages:
        hits = _top_pages(result)
        recall = any(p in hits for p in q.expected_pages)

    keyword_hit: bool | None = None
    if result.status == TurnStatus.ANSWERED and q.expected_keywords:
        keyword_hit = _keyword_hit(result.message, q.expected_keywords)

    return EvalResult(
        id=q.id,
        category=q.category,
        question=q.question,
        expected_outcome=q.expected_outcome,
        actual_outcome=result.status.value,
        retrieval_recall=recall,
        gate_correct=gate_correct,
        keyword_hit=keyword_hit,
        retrieval_ms=result.timings.retrieval_ms,
        gate_ms=result.timings.gate_ms,
        llm_ms=result.timings.llm_ms,
        verify_ms=result.timings.verify_ms,
        total_ms=result.timings.total_ms,
        clarify_depth=0,  # single-turn eval; multi-turn clarify loop is future work
        attempts=result.attempts,
        error=None,
    )


# ---- aggregate reporting -------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    if not results:
        return {"total": 0}

    by_cat: dict[str, list[EvalResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    def _rate(items: Iterable[EvalResult], pred) -> float:
        items = list(items)
        if not items:
            return 0.0
        return sum(1 for i in items if pred(i)) / len(items)

    recall_items = [r for r in results if r.retrieval_recall is not None]
    kw_items = [r for r in results if r.keyword_hit is not None]
    answered = [r for r in results if r.actual_outcome == "answered"]
    errors = [r for r in results if r.error]
    totals = [r.total_ms for r in results]

    summary: dict[str, Any] = {
        "total_questions": len(results),
        "errors": len(errors),
        "gate_accuracy": _rate(results, lambda r: r.gate_correct),
        "retrieval_recall": {
            "measured_on": len(recall_items),
            "hit_rate": _rate(recall_items, lambda r: r.retrieval_recall is True),
        },
        "answer_keyword_hit": {
            "measured_on": len(kw_items),
            "hit_rate": _rate(kw_items, lambda r: r.keyword_hit is True),
        },
        "latency_ms": {
            "p50_total": round(_percentile(totals, 0.5), 1),
            "p95_total": round(_percentile(totals, 0.95), 1),
            "avg_total": round(statistics.fmean(totals), 1) if totals else 0.0,
            "avg_retrieval": round(statistics.fmean(r.retrieval_ms for r in results), 1),
            "avg_gate": round(statistics.fmean(r.gate_ms for r in results), 1),
            "avg_llm": round(statistics.fmean(r.llm_ms for r in results), 1),
            "avg_verify": round(statistics.fmean(r.verify_ms for r in results), 1),
        },
        "by_category": {},
    }
    for cat, items in by_cat.items():
        summary["by_category"][cat] = {
            "count": len(items),
            "gate_accuracy": round(_rate(items, lambda r: r.gate_correct), 3),
            "outcomes": {
                outcome: sum(1 for r in items if r.actual_outcome == outcome)
                for outcome in sorted({r.actual_outcome for r in items})
            },
        }
    return summary


# ---- IO helpers ----------------------------------------------------------


def write_csv(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id", "category", "question", "expected_outcome", "actual_outcome",
                "gate_correct", "retrieval_recall", "keyword_hit",
                "retrieval_ms", "gate_ms", "llm_ms", "verify_ms", "total_ms",
                "attempts", "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.id, r.category, r.question, r.expected_outcome, r.actual_outcome,
                    r.gate_correct, r.retrieval_recall, r.keyword_hit,
                    round(r.retrieval_ms, 1), round(r.gate_ms, 1),
                    round(r.llm_ms, 1), round(r.verify_ms, 1), round(r.total_ms, 1),
                    r.attempts, r.error or "",
                ]
            )


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


# ---- controller factory --------------------------------------------------


def build_controller(verify_answers: bool = False, use_qa_cache: bool = False) -> ConversationController:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set — required for the Jev gate")
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM api key not set — set OPENAI_API_KEY or OPENROUTER_API_KEY "
            "and LLM_PROVIDER accordingly"
        )

    repo = MetadataRepository(settings.metadata_db_path)
    embedder = EmbeddingService(settings.embedding_model)
    retriever = Retriever(
        repo=repo,
        embedding_service=embedder,
        vector_store_dir=settings.vector_store_path,
    )
    base = settings.openrouter_base_url.rstrip("/")
    decisions_base = (base[:-3] + "/alpha") if base.endswith("/v1") else (base.rsplit("/", 1)[0] + "/alpha")
    jev = JevDecisionsClient(
        api_key=settings.openrouter_api_key,
        base_url=decisions_base,
        model=settings.jev_model,
    )
    answer_gen = AnswerGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    qa_matcher = None
    if use_qa_cache:
        qa_matcher = QAMatcher(
            repo=repo,
            embedding_service=embedder,
            qa_index_dir=settings.vector_store_path,
            decisions_client=jev,
        )

    return ConversationController(
        retriever=retriever,
        jev_client=jev,
        policy=DecisionPolicy(max_retrieval_attempts=settings.max_retrieval_attempts),
        base_top_k=settings.top_k,
        answer_generator=answer_gen,
        verify_answers=verify_answers,
        qa_matcher=qa_matcher,
    )


# ---- CLI -----------------------------------------------------------------


def run(
    *,
    dataset_path: Path,
    out_dir: Path,
    verify: bool,
    use_qa_cache: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    questions = dataset.questions if limit is None else dataset.questions[:limit]
    controller = build_controller(verify_answers=verify, use_qa_cache=use_qa_cache)

    results: list[EvalResult] = []
    for i, q in enumerate(questions, start=1):
        logger.info("[%d/%d] %s %s", i, len(questions), q.id, q.question)
        results.append(
            evaluate_one(
                controller=controller,
                chapter_id=dataset.chapter_id,
                q=q,
            )
        )

    summary = summarize(results)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, out_dir / f"eval_{ts}.csv")
    write_summary_json(summary, out_dir / f"eval_{ts}_summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline eval set.")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to labelled JSON set."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/eval"), help="Output directory."
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run with the answer verifier (M6) on."
    )
    parser.add_argument(
        "--qa-cache", action="store_true", help="Enable the Q&A cache (match-first path)."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N questions."
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARN...)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    summary = run(
        dataset_path=args.dataset,
        out_dir=args.out,
        verify=args.verify,
        use_qa_cache=args.qa_cache,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
