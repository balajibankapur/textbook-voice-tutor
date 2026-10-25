"""Lesson mode controller — a chapter walkthrough driven by Jev's noul judge.

Two directions share one state machine:

TEACH mode
    - PRESENT: tutor speaks a grounded, age-appropriate explanation
    - ASK:     tutor asks a question that tests the concept
    - LISTEN:  wait for the student's spoken answer
    - JUDGE:   Jev noul decides correct / partial / wrong
    - PRAISE / RE-EXPLAIN / RECLARIFY as needed

REVERSE mode
    - PROMPT:  tutor asks the student to explain the concept in their words
    - LISTEN:  wait for the student's answer
    - JUDGE:   same noul check
    - PRAISE / CLARIFY

The LLM only runs for presentation text; Jev decides correctness, the controller
decides routing. Praise is drawn from a pool to avoid repetition fatigue.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from decision.jev_client import JevError
from decision.jev_decisions_client import JevDecisionsClient
from llm.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
)
from retrieval.retriever import RetrievedChunk
from storage.metadata_repository import MetadataRepository
from storage.models import ChunkRow, ConceptRow, OcrQualityStatus

logger = logging.getLogger(__name__)


# ---- enums + results ----------------------------------------------------


class LessonMode(str, Enum):
    TEACH = "teach"      # tutor presents, then asks
    REVERSE = "reverse"  # tutor prompts, student explains


class LessonStage(str, Enum):
    PRESENT = "present"              # tutor speaks the concept (TEACH only)
    ASK = "ask"                      # tutor asks a question
    LISTEN = "listen"                # waiting for student's reply
    PRAISE_AND_NEXT = "praise_next"  # terminal for one concept
    CLARIFY = "clarify"              # answer was partial / wrong; tutor adds context
    LESSON_COMPLETE = "complete"     # no more concepts
    FAILED = "failed"                # Jev / LLM error, fail-closed


class JudgementTier(str, Enum):
    CORRECT = "correct"      # noul >= correct_threshold
    PARTIAL = "partial"      # partial_threshold <= noul < correct_threshold
    WRONG = "wrong"          # noul < partial_threshold


@dataclass
class JudgementResult:
    tier: JudgementTier
    probability: float
    noul_threshold_correct: float
    noul_threshold_partial: float


@dataclass
class LessonStep:
    """One tutor-driven step the UI should render.

    Exactly one of `speech_text` / `prompt_text` / `done_message` is populated,
    depending on `stage`.
    """
    stage: LessonStage
    concept: ConceptRow | None
    concept_index: int        # 0-based position within the lesson
    concept_total: int
    attempts_on_concept: int  # how many answers Jev has judged for this concept so far
    speech_text: str = ""     # text to speak + show when the tutor talks
    prompt_text: str = ""     # the question (ASK/PROMPT stages carry this too)
    judgement: JudgementResult | None = None
    error: str | None = None


# ---- praise / clarification pools --------------------------------------


_PRAISE_POOL = [
    "That's exactly right! ",
    "Perfect — well done! ",
    "Yes, you got it! ",
    "Nice thinking! ",
    "Spot on! ",
    "Correct! Great job. ",
    "You nailed it. ",
]

_PARTIAL_PREFIX_POOL = [
    "Close — you got the main idea. ",
    "Good try — part of that is right. ",
    "Nice — you're on the right track. ",
    "Half there — let's build on what you said. ",
]

_WRONG_PREFIX_POOL = [
    "Not quite — let me try a different way. ",
    "Good effort — the chapter actually says it a little differently. ",
    "Almost — here's another way to think about it. ",
]


def _pick(pool: list[str]) -> str:
    return random.choice(pool)


def _speak_explain_and_ask(explanation: str, question: str) -> str:
    """Combine an explanation and a follow-up question into one spoken block.

    Keeps a short pause ('. ') and a friendly transition so TTS reads them
    naturally instead of running them together.
    """
    explanation = (explanation or "").strip().rstrip(".") + "."
    question = (question or "").strip()
    if not question:
        return explanation
    return f"{explanation} Now, here's a question for you: {question}"


# ---- controller ---------------------------------------------------------


class LessonController:
    def __init__(
        self,
        *,
        repo: MetadataRepository,
        answer_generator: AnswerGenerator,
        decisions_client: JevDecisionsClient,
        correct_threshold: float = 0.70,
        partial_threshold: float = 0.40,
        max_attempts_per_concept: int = 2,
    ) -> None:
        if not 0 <= partial_threshold < correct_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= partial < correct <= 1")
        self.repo = repo
        self.answer_generator = answer_generator
        self.decisions_client = decisions_client
        self.correct_threshold = correct_threshold
        self.partial_threshold = partial_threshold
        self.max_attempts_per_concept = max_attempts_per_concept

    # ---- entry points ---------------------------------------------------

    def start(
        self,
        *,
        chapter_id: str,
        mode: LessonMode,
    ) -> tuple["LessonState", LessonStep]:
        """Begin a lesson. Returns the initial state and the first step to show."""
        concepts = self.repo.list_concepts(chapter_id)
        if not concepts:
            raise ValueError(
                f"chapter {chapter_id!r} has no concept lesson plan — build it first"
            )
        state = LessonState(
            chapter_id=chapter_id,
            mode=mode,
            concepts=concepts,
            concept_index=0,
            attempts=0,
        )
        step = self._open_concept(state)
        return state, step

    def submit_answer(
        self,
        state: "LessonState",
        student_answer: str,
    ) -> LessonStep:
        """Judge the student's answer and return the next step."""
        if state.stage is LessonStage.LESSON_COMPLETE:
            return self._complete_step(state)

        concept = state.current_concept()
        if concept is None:
            return self._complete_step(state)

        state.attempts += 1
        try:
            probability = self.decisions_client.judge_answer(
                concept_title=concept.title,
                learning_objective=concept.learning_objective,
                key_points=concept.key_points,
                student_answer=student_answer,
            )
        except JevError as exc:
            logger.warning("judge_answer failed: %s", exc)
            return LessonStep(
                stage=LessonStage.FAILED,
                concept=concept,
                concept_index=state.concept_index,
                concept_total=len(state.concepts),
                attempts_on_concept=state.attempts,
                error=f"Couldn't check that answer right now ({exc}). Try again?",
            )

        tier = self._tier(probability)
        judgement = JudgementResult(
            tier=tier,
            probability=probability,
            noul_threshold_correct=self.correct_threshold,
            noul_threshold_partial=self.partial_threshold,
        )

        if tier is JudgementTier.CORRECT:
            feedback = _pick(_PRAISE_POOL)
            step = LessonStep(
                stage=LessonStage.PRAISE_AND_NEXT,
                concept=concept,
                concept_index=state.concept_index,
                concept_total=len(state.concepts),
                attempts_on_concept=state.attempts,
                speech_text=feedback,
                judgement=judgement,
            )
            state.stage = LessonStage.PRAISE_AND_NEXT
            return step

        if tier is JudgementTier.PARTIAL:
            # Accept partial on first try, add the missing piece, move on.
            addendum = self._compose_clarification(concept, state.mode)
            feedback = _pick(_PARTIAL_PREFIX_POOL) + addendum
            step = LessonStep(
                stage=LessonStage.PRAISE_AND_NEXT,
                concept=concept,
                concept_index=state.concept_index,
                concept_total=len(state.concepts),
                attempts_on_concept=state.attempts,
                speech_text=feedback,
                judgement=judgement,
            )
            state.stage = LessonStage.PRAISE_AND_NEXT
            return step

        # WRONG → re-explain if we still have attempts; else give the answer and move on.
        if state.attempts < self.max_attempts_per_concept:
            prefix = _pick(_WRONG_PREFIX_POOL)
            try:
                clarification = self._compose_clarification(concept, state.mode)
            except AnswerGenerationError as exc:
                return LessonStep(
                    stage=LessonStage.FAILED,
                    concept=concept,
                    concept_index=state.concept_index,
                    concept_total=len(state.concepts),
                    attempts_on_concept=state.attempts,
                    error=f"Explanation generator failed: {exc}",
                )
            # Repeat the same question (spoken aloud too) so the student hears
            # what to answer again.
            spoken = _speak_explain_and_ask(prefix + clarification, state.last_prompt)
            step = LessonStep(
                stage=LessonStage.CLARIFY,
                concept=concept,
                concept_index=state.concept_index,
                concept_total=len(state.concepts),
                attempts_on_concept=state.attempts,
                speech_text=spoken,
                prompt_text=state.last_prompt,
                judgement=judgement,
            )
            state.stage = LessonStage.CLARIFY
            return step

        # Attempts exhausted — tell them the right answer kindly, move on.
        gentle = (
            "Here's what the chapter says: "
            + self._compose_clarification(concept, state.mode)
        )
        step = LessonStep(
            stage=LessonStage.PRAISE_AND_NEXT,
            concept=concept,
            concept_index=state.concept_index,
            concept_total=len(state.concepts),
            attempts_on_concept=state.attempts,
            speech_text=gentle,
            judgement=judgement,
        )
        state.stage = LessonStage.PRAISE_AND_NEXT
        return step

    def reveal_answer(self, state: "LessonState") -> LessonStep:
        """Give the student the answer and move on. Used when they don't answer
        within the UI's silence timeout, or when they tap "Reveal answer."

        No Jev judgement is run — the controller simply speaks the key points
        gently and marks the attempt budget as exhausted so advance() moves on.
        """
        concept = state.current_concept()
        if concept is None:
            return self._complete_step(state)
        state.attempts = max(state.attempts, self.max_attempts_per_concept)
        state.stage = LessonStage.PRAISE_AND_NEXT
        clarification = self._compose_clarification(concept, state.mode)
        gentle = "No worries — here's what the chapter says: " + clarification
        return LessonStep(
            stage=LessonStage.PRAISE_AND_NEXT,
            concept=concept,
            concept_index=state.concept_index,
            concept_total=len(state.concepts),
            attempts_on_concept=state.attempts,
            speech_text=gentle,
        )

    def advance(self, state: "LessonState") -> LessonStep:
        """Caller invokes this after showing a PRAISE_AND_NEXT or CLARIFY step
        to move on to the next concept (or re-present the same one on CLARIFY).
        """
        if state.stage is LessonStage.CLARIFY:
            # Keep the same concept; re-arm for another LISTEN.
            state.stage = LessonStage.LISTEN
            return LessonStep(
                stage=LessonStage.LISTEN,
                concept=state.current_concept(),
                concept_index=state.concept_index,
                concept_total=len(state.concepts),
                attempts_on_concept=state.attempts,
                prompt_text=state.last_prompt,
            )

        # PRAISE_AND_NEXT → move the pointer forward.
        state.concept_index += 1
        state.attempts = 0
        if state.concept_index >= len(state.concepts):
            state.stage = LessonStage.LESSON_COMPLETE
            return self._complete_step(state)
        return self._open_concept(state)

    # ---- helpers --------------------------------------------------------

    def _tier(self, probability: float) -> JudgementTier:
        if probability >= self.correct_threshold:
            return JudgementTier.CORRECT
        if probability >= self.partial_threshold:
            return JudgementTier.PARTIAL
        return JudgementTier.WRONG

    def _open_concept(self, state: "LessonState") -> LessonStep:
        concept = state.current_concept()
        assert concept is not None
        passages = self._passages_for(concept)
        if state.mode is LessonMode.TEACH:
            try:
                explanation = self._compose_explanation(concept, passages)
                question = self._compose_question(concept, passages)
            except AnswerGenerationError as exc:
                return LessonStep(
                    stage=LessonStage.FAILED,
                    concept=concept,
                    concept_index=state.concept_index,
                    concept_total=len(state.concepts),
                    attempts_on_concept=0,
                    error=f"Explanation generator failed: {exc}",
                )
            state.last_prompt = question
            state.stage = LessonStage.PRESENT
            # Speak BOTH the explanation and the question in one audio block so
            # a child who can't read sees the question text AND hears it.
            spoken = _speak_explain_and_ask(explanation, question)
            return LessonStep(
                stage=LessonStage.PRESENT,
                concept=concept,
                concept_index=state.concept_index,
                concept_total=len(state.concepts),
                attempts_on_concept=0,
                speech_text=spoken,
                prompt_text=question,
            )
        # REVERSE: ask the student to explain
        prompt = (
            f"Tell me what you know about: {concept.title.lower()}. "
            "Use your own words."
        )
        state.last_prompt = prompt
        state.stage = LessonStage.ASK
        return LessonStep(
            stage=LessonStage.ASK,
            concept=concept,
            concept_index=state.concept_index,
            concept_total=len(state.concepts),
            attempts_on_concept=0,
            speech_text=prompt,
            prompt_text=prompt,
        )

    def _complete_step(self, state: "LessonState") -> LessonStep:
        return LessonStep(
            stage=LessonStage.LESSON_COMPLETE,
            concept=None,
            concept_index=len(state.concepts),
            concept_total=len(state.concepts),
            attempts_on_concept=0,
            speech_text="Great work! You've finished all the concepts in this chapter.",
        )

    def _passages_for(self, concept: ConceptRow) -> list[RetrievedChunk]:
        rows = [self.repo.get_chunk(cid) for cid in concept.source_chunk_ids]
        passages: list[RetrievedChunk] = []
        for row in rows:
            if row is None or row.chunk_id is None:
                continue
            passages.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    chapter_id=row.chapter_id,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    section_title=row.section_title,
                    chunk_text=row.chunk_text,
                    score=1.0,
                    ocr_quality_status=row.ocr_quality_status,
                )
            )
        return passages

    def _compose_explanation(
        self, concept: ConceptRow, passages: list[RetrievedChunk]
    ) -> str:
        """Produce the tutor's spoken explanation of this concept."""
        if not passages:
            return "Let's talk about " + concept.title + "."
        question = (
            f"Explain the concept '{concept.title}' to a Grade 2 student. "
            f"Learning objective: {concept.learning_objective}. "
            "Cover these grounded points, in simple words: "
            + " · ".join(concept.key_points)
        )
        generated = self.answer_generator.generate(
            question=question, passages=passages
        )
        return generated.answer if generated.answerable else (
            "This part of the chapter talks about " + concept.title + "."
        )

    def _compose_question(
        self, concept: ConceptRow, passages: list[RetrievedChunk]
    ) -> str:
        """Produce the question the tutor asks after presenting the concept."""
        prompt = (
            f"Write ONE short question for a Grade 2 student that tests whether "
            f"they can meet this learning objective: {concept.learning_objective}. "
            f"The question must be answerable from these grounded key points: "
            + " · ".join(concept.key_points)
            + ". Return just the question text — no quotes, no label."
        )
        return self.answer_generator.rewrite_question(
            original=concept.title,
            follow_up=concept.learning_objective,
            reply=prompt,
        )

    def _compose_clarification(
        self, concept: ConceptRow, mode: LessonMode
    ) -> str:
        """Short clarification spoken on partial/wrong answers.

        Uses the key_points directly — no new LLM call needed, keeps latency down.
        """
        if not concept.key_points:
            return concept.learning_objective
        joined = "; ".join(concept.key_points)
        if mode is LessonMode.REVERSE:
            return f"The chapter says: {joined}."
        return f"Remember: {joined}."


@dataclass
class LessonState:
    chapter_id: str
    mode: LessonMode
    concepts: list[ConceptRow]
    concept_index: int = 0
    attempts: int = 0
    stage: LessonStage = LessonStage.PRESENT
    last_prompt: str = ""

    def current_concept(self) -> ConceptRow | None:
        if 0 <= self.concept_index < len(self.concepts):
            return self.concepts[self.concept_index]
        return None
