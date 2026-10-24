"""Conversation controller: wires retriever ↔ Jev ↔ policy for one student turn.

Tracks per-stage latency so M8's evaluation can report real numbers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from conversation.context_manager import ConversationContext
from decision.decision_policy import Action, DecisionPolicy
from decision.jev_client import JevClient, JevDecision, JevError, JevVerification
from llm.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
    GeneratedAnswer,
)
from retrieval.qa_matcher import QAMatch, QAMatcher
from retrieval.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class TurnStatus(str, Enum):
    ANSWER_READY = "answer_ready"      # evidence accepted; no LLM wired yet
    ANSWERED = "answered"              # LLM produced a validated answer
    CLARIFY = "clarify"
    REFUSE = "refuse"
    FAIL_CLOSED = "fail_closed"


@dataclass
class Timings:
    """Per-stage latency in milliseconds. Cumulative within one controller turn."""
    qa_cache_ms: float = 0.0
    retrieval_ms: float = 0.0
    gate_ms: float = 0.0
    llm_ms: float = 0.0
    verify_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class TurnResult:
    status: TurnStatus
    message: str  # student-facing text (clarification question, refusal, etc.)
    passages: list[RetrievedChunk] = field(default_factory=list)
    decision: JevDecision | None = None
    attempts: int = 0  # retrieval attempts actually made
    generated_answer: GeneratedAnswer | None = None
    verification: JevVerification | None = None
    regenerated: bool = False
    follow_up_question: str | None = None  # when the LLM wants to narrow the question
    timings: Timings = field(default_factory=Timings)
    qa_cache_hit: QAMatch | None = None  # populated when served from the Q&A corpus


class ConversationController:
    def __init__(
        self,
        *,
        retriever: Retriever,
        jev_client: JevClient,
        policy: DecisionPolicy,
        base_top_k: int = 5,
        expand_step: int = 5,
        answer_generator: AnswerGenerator | None = None,
        verify_answers: bool = False,
        qa_matcher: QAMatcher | None = None,
    ) -> None:
        self.retriever = retriever
        self.jev_client = jev_client
        self.policy = policy
        self.base_top_k = base_top_k
        self.expand_step = expand_step
        self.answer_generator = answer_generator
        self.verify_answers = verify_answers
        self.qa_matcher = qa_matcher

    def run_turn(
        self,
        *,
        chapter_id: str,
        question: str,
        context: ConversationContext,
    ) -> TurnResult:
        turn_start = time.perf_counter()
        timings = Timings()

        # Cache-first: see if a pre-generated Q&A pair satisfies this question.
        if self.qa_matcher is not None:
            t = time.perf_counter()
            match = self.qa_matcher.find_cached_answer(chapter_id, question)
            timings.qa_cache_ms = (time.perf_counter() - t) * 1000
            if match is not None:
                timings.total_ms = (time.perf_counter() - turn_start) * 1000
                logger.info(
                    "qa cache hit: qa_id=%s confidence=%.2f",
                    match.qa_pair.qa_id, match.confidence,
                )
                return TurnResult(
                    status=TurnStatus.ANSWERED,
                    message=match.qa_pair.answer,
                    attempts=0,
                    timings=timings,
                    qa_cache_hit=match,
                )

        attempt = 0
        passages: list[RetrievedChunk] = []
        decision: JevDecision | None = None

        while attempt < self.policy.max_retrieval_attempts:
            attempt += 1
            top_k = self.base_top_k + self.expand_step * (attempt - 1)

            t_r = time.perf_counter()
            passages = self.retriever.search(chapter_id, question, top_k=top_k)
            timings.retrieval_ms += (time.perf_counter() - t_r) * 1000
            logger.info(
                "turn attempt=%s chapter=%s top_k=%s hits=%s",
                attempt, chapter_id, top_k, len(passages),
            )

            t_g = time.perf_counter()
            try:
                decision = self.jev_client.decide(
                    question=question,
                    passages=passages,
                    context_messages=context.recent(),
                )
            except JevError as exc:
                timings.gate_ms += (time.perf_counter() - t_g) * 1000
                timings.total_ms = (time.perf_counter() - turn_start) * 1000
                logger.warning("Jev error: %s", exc)
                outcome = self.policy.on_jev_error()
                return TurnResult(
                    status=TurnStatus.FAIL_CLOSED,
                    message=outcome.message,
                    passages=passages,
                    decision=None,
                    attempts=attempt,
                    timings=timings,
                )
            timings.gate_ms += (time.perf_counter() - t_g) * 1000

            outcome = self.policy.decide(decision, attempt)
            logger.info(
                "jev decision: clear=%s relevant=%s sufficient=%s conf=%.2f -> %s",
                decision.question_clear,
                decision.evidence_relevant,
                decision.evidence_sufficient,
                decision.decision_confidence,
                outcome.action.value,
            )

            if outcome.action == Action.CLARIFY:
                timings.total_ms = (time.perf_counter() - turn_start) * 1000
                return TurnResult(
                    status=TurnStatus.CLARIFY,
                    message=outcome.message,
                    passages=passages,
                    decision=decision,
                    attempts=attempt,
                    timings=timings,
                )
            if outcome.action == Action.ANSWER:
                result = self._generate_or_defer(
                    question=question,
                    passages=passages,
                    decision=decision,
                    attempt=attempt,
                    context=context,
                    timings=timings,
                )
                result.timings.total_ms = (time.perf_counter() - turn_start) * 1000
                return result
            if outcome.action == Action.REFUSE:
                timings.total_ms = (time.perf_counter() - turn_start) * 1000
                return TurnResult(
                    status=TurnStatus.REFUSE,
                    message=outcome.message,
                    passages=passages,
                    decision=decision,
                    attempts=attempt,
                    timings=timings,
                )
            if outcome.action == Action.FAIL_CLOSED:
                timings.total_ms = (time.perf_counter() - turn_start) * 1000
                return TurnResult(
                    status=TurnStatus.FAIL_CLOSED,
                    message=outcome.message,
                    passages=passages,
                    decision=decision,
                    attempts=attempt,
                    timings=timings,
                )
            # EXPAND_RETRIEVAL — loop again with wider top_k.

        # Fell out of the loop: last decision didn't accept and we're out of attempts.
        outcome = self.policy.decide(decision, attempt) if decision else self.policy.on_jev_error()
        status = {
            Action.REFUSE: TurnStatus.REFUSE,
            Action.CLARIFY: TurnStatus.CLARIFY,
            Action.FAIL_CLOSED: TurnStatus.FAIL_CLOSED,
        }.get(outcome.action, TurnStatus.REFUSE)
        timings.total_ms = (time.perf_counter() - turn_start) * 1000
        return TurnResult(
            status=status,
            message=outcome.message,
            passages=passages,
            decision=decision,
            attempts=attempt,
            timings=timings,
        )

    # ---- answer generation ----------------------------------------------

    def _generate_or_defer(
        self,
        *,
        question: str,
        passages: list[RetrievedChunk],
        decision: JevDecision,
        attempt: int,
        context: ConversationContext,
        timings: Timings,
    ) -> TurnResult:
        """When an answer generator is wired, call it; otherwise return the raw
        'evidence accepted' state so earlier milestones' UI still works."""
        if self.answer_generator is None:
            return TurnResult(
                status=TurnStatus.ANSWER_READY,
                message="",
                passages=passages,
                decision=decision,
                attempts=attempt,
                timings=timings,
            )

        t = time.perf_counter()
        try:
            generated = self._generate(question, passages, context)
        except AnswerGenerationError as exc:
            timings.llm_ms += (time.perf_counter() - t) * 1000
            logger.warning("answer generator failed: %s", exc)
            outcome = self.policy.on_jev_error()  # same fail-closed message
            return TurnResult(
                status=TurnStatus.FAIL_CLOSED,
                message=outcome.message,
                passages=passages,
                decision=decision,
                attempts=attempt,
                timings=timings,
            )
        timings.llm_ms += (time.perf_counter() - t) * 1000

        if not generated.answerable:
            # LLM couldn't ground an answer. If it offered a narrowing question,
            # treat that as a CLARIFY turn so the student can steer; otherwise refuse.
            if generated.follow_up_question:
                return TurnResult(
                    status=TurnStatus.CLARIFY,
                    message=generated.follow_up_question,
                    passages=passages,
                    decision=decision,
                    attempts=attempt,
                    generated_answer=generated,
                    follow_up_question=generated.follow_up_question,
                    timings=timings,
                )
            refusal = generated.answer or (
                "I couldn't find a complete answer in the chapter."
            )
            return TurnResult(
                status=TurnStatus.REFUSE,
                message=refusal,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=generated,
                timings=timings,
            )

        if not self.verify_answers:
            return TurnResult(
                status=TurnStatus.ANSWERED,
                message=generated.answer,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=generated,
                follow_up_question=generated.follow_up_question,
                timings=timings,
            )

        return self._verify_and_maybe_regenerate(
            question=question,
            passages=passages,
            decision=decision,
            attempt=attempt,
            context=context,
            generated=generated,
            timings=timings,
        )

    def _generate(
        self,
        question: str,
        passages: list[RetrievedChunk],
        context: ConversationContext,
        revision_feedback: str | None = None,
    ) -> GeneratedAnswer:
        assert self.answer_generator is not None
        return self.answer_generator.generate(
            question=question,
            passages=passages,
            context_messages=context.recent(),
            revision_feedback=revision_feedback,
        )

    def _verify_and_maybe_regenerate(
        self,
        *,
        question: str,
        passages: list[RetrievedChunk],
        decision: JevDecision,
        attempt: int,
        context: ConversationContext,
        generated: GeneratedAnswer,
        timings: Timings,
    ) -> TurnResult:
        """Spec §13: verify; if unsupported, regenerate once; if still
        unsupported, refuse rather than speaking an unverified answer."""
        t = time.perf_counter()
        try:
            verification = self.jev_client.verify_answer(
                question=question, passages=passages, answer=generated.answer
            )
        except JevError as exc:
            timings.verify_ms += (time.perf_counter() - t) * 1000
            logger.warning("verification failed: %s", exc)
            outcome = self.policy.on_jev_error()
            return TurnResult(
                status=TurnStatus.FAIL_CLOSED,
                message=outcome.message,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=generated,
                timings=timings,
            )
        timings.verify_ms += (time.perf_counter() - t) * 1000

        logger.info(
            "verification: supported=%s conf=%.2f reason=%r",
            verification.supported,
            verification.confidence,
            verification.reason,
        )

        if verification.supported:
            return TurnResult(
                status=TurnStatus.ANSWERED,
                message=generated.answer,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=generated,
                verification=verification,
                follow_up_question=generated.follow_up_question,
            )

        # Unsupported — try one regeneration with the verifier's reason as feedback.
        try:
            revised = self._generate(
                question=question,
                passages=passages,
                context=context,
                revision_feedback=verification.reason or "answer not supported by passages",
            )
        except AnswerGenerationError as exc:
            logger.warning("regeneration failed: %s", exc)
            outcome = self.policy.on_jev_error()
            return TurnResult(
                status=TurnStatus.FAIL_CLOSED,
                message=outcome.message,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=generated,
                verification=verification,
            )

        if not revised.answerable:
            if revised.follow_up_question:
                return TurnResult(
                    status=TurnStatus.CLARIFY,
                    message=revised.follow_up_question,
                    passages=passages,
                    decision=decision,
                    attempts=attempt,
                    generated_answer=revised,
                    verification=verification,
                    regenerated=True,
                    follow_up_question=revised.follow_up_question,
                    timings=timings,
                )
            return TurnResult(
                status=TurnStatus.REFUSE,
                message=revised.answer or "I couldn't verify a complete answer in the chapter.",
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=revised,
                verification=verification,
                regenerated=True,
                timings=timings,
            )

        # Verify the revision; if still unsupported, refuse — do NOT speak it.
        t = time.perf_counter()
        try:
            re_verification = self.jev_client.verify_answer(
                question=question, passages=passages, answer=revised.answer
            )
        except JevError as exc:
            timings.verify_ms += (time.perf_counter() - t) * 1000
            logger.warning("re-verification failed: %s", exc)
            outcome = self.policy.on_jev_error()
            return TurnResult(
                status=TurnStatus.FAIL_CLOSED,
                message=outcome.message,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=revised,
                verification=verification,
                regenerated=True,
                timings=timings,
            )
        timings.verify_ms += (time.perf_counter() - t) * 1000

        if re_verification.supported:
            return TurnResult(
                status=TurnStatus.ANSWERED,
                message=revised.answer,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=revised,
                verification=re_verification,
                regenerated=True,
                follow_up_question=revised.follow_up_question,
                timings=timings,
            )

        # Still unsupported. If the revision carried a follow-up, use that to
        # ask the student to narrow — better than refusing silently.
        if revised.follow_up_question:
            return TurnResult(
                status=TurnStatus.CLARIFY,
                message=revised.follow_up_question,
                passages=passages,
                decision=decision,
                attempts=attempt,
                generated_answer=revised,
                verification=re_verification,
                regenerated=True,
                follow_up_question=revised.follow_up_question,
                timings=timings,
            )
        return TurnResult(
            status=TurnStatus.REFUSE,
            message=(
                "I couldn't produce an answer that stays within what the "
                "textbook actually says. The passages may be too noisy or "
                "incomplete for this question."
            ),
            passages=passages,
            decision=decision,
            attempts=attempt,
            generated_answer=revised,
            verification=re_verification,
            regenerated=True,
            timings=timings,
        )
