"""Fast tests for LessonController using fake generator + Jev + an in-memory repo."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conversation.lesson_controller import (
    JudgementTier,
    LessonController,
    LessonMode,
    LessonStage,
)
from decision.jev_client import JevError
from storage.metadata_repository import MetadataRepository
from storage.models import (
    ChapterRow,
    ChunkRow,
    ConceptRow,
    IngestionStatus,
    OcrQualityStatus,
)


# ---- fakes ---------------------------------------------------------------


@dataclass
class _GeneratedAnswer:
    answerable: bool
    answer: str
    supporting_page_numbers: list[int] = field(default_factory=list)
    supporting_passage_ids: list[int] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    follow_up_question: str | None = None


@dataclass
class FakeGenerator:
    answers_queue: list = field(default_factory=list)
    rewrites_queue: list = field(default_factory=list)

    def generate(self, *, question, passages, context_messages=(), revision_feedback=None):
        text = self.answers_queue.pop(0) if self.answers_queue else "an explanation"
        return _GeneratedAnswer(answerable=True, answer=text)

    def rewrite_question(self, *, original, follow_up, reply):
        return self.rewrites_queue.pop(0) if self.rewrites_queue else "What are the parts of a plant?"


@dataclass
class FakeDecisions:
    noul_queue: list = field(default_factory=list)
    error_queue: list = field(default_factory=list)
    seen: list = field(default_factory=list)

    def judge_answer(self, *, concept_title, learning_objective, key_points, student_answer):
        self.seen.append(
            {"title": concept_title, "answer": student_answer, "n_key_points": len(key_points)}
        )
        if self.error_queue:
            exc = self.error_queue.pop(0)
            raise exc
        return self.noul_queue.pop(0)


# ---- fixtures ------------------------------------------------------------


@pytest.fixture
def seeded(tmp_path: Path):
    repo = MetadataRepository(tmp_path / "tutor.db")
    now = datetime.now(timezone.utc)
    repo.upsert_chapter(
        ChapterRow(
            chapter_id="ch1", title="T", subject="S", source_file="x",
            page_count=5, ingestion_status=IngestionStatus.READY,
            embedding_model="static", created_at=now, updated_at=now,
        )
    )
    # one chunk per concept so _passages_for returns something
    chunk_rows = [
        ChunkRow(
            chunk_id=None, chapter_id="ch1",
            page_start=i, page_end=i, section_title=None,
            chunk_text=f"text for concept {i}",
            faiss_id=None,
            ocr_quality_status=OcrQualityStatus.OK,
        )
        for i in range(1, 4)
    ]
    chunk_ids = repo.insert_chunks(chunk_rows)
    concepts = [
        ConceptRow(
            concept_id=None, chapter_id="ch1", order_index=i,
            title=f"Concept {i + 1}",
            learning_objective="The student can describe this.",
            key_points=[f"point {i}a", f"point {i}b"],
            page_numbers=[i + 1],
            source_chunk_ids=[chunk_ids[i]],
            created_at=now,
        )
        for i in range(3)
    ]
    repo.insert_concepts(concepts)
    yield repo, chunk_ids
    repo.close()


def _controller(
    repo,
    *,
    generator=None,
    decisions=None,
    max_attempts=2,
) -> LessonController:
    return LessonController(
        repo=repo,
        answer_generator=generator or FakeGenerator(),
        decisions_client=decisions or FakeDecisions(),
        max_attempts_per_concept=max_attempts,
    )


# ---- start() ------------------------------------------------------------


def test_start_teach_mode_presents_and_prepares_question(seeded) -> None:
    repo, _ = seeded
    gen = FakeGenerator(answers_queue=["Plants have roots."], rewrites_queue=["What are the parts?"])
    ctrl = _controller(repo, generator=gen)
    state, step = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    assert step.stage is LessonStage.PRESENT
    # Both the explanation AND the question must be in the spoken block so the
    # student (who may not read fluently) hears the question too.
    assert "Plants have roots" in step.speech_text
    assert "What are the parts?" in step.speech_text
    assert step.prompt_text == "What are the parts?"
    assert state.last_prompt == "What are the parts?"
    assert step.concept_index == 0
    assert step.concept_total == 3


def test_start_reverse_mode_asks_student_to_explain(seeded) -> None:
    repo, _ = seeded
    ctrl = _controller(repo, generator=FakeGenerator())
    state, step = ctrl.start(chapter_id="ch1", mode=LessonMode.REVERSE)
    assert step.stage is LessonStage.ASK
    assert "concept 1" in step.speech_text.lower()
    assert "your own words" in step.speech_text.lower()


def test_start_raises_when_no_concepts(tmp_path: Path) -> None:
    repo = MetadataRepository(tmp_path / "tutor.db")
    now = datetime.now(timezone.utc)
    repo.upsert_chapter(
        ChapterRow(
            chapter_id="empty", title="T", subject="S", source_file="x",
            page_count=1, ingestion_status=IngestionStatus.READY,
            embedding_model="static", created_at=now, updated_at=now,
        )
    )
    try:
        ctrl = _controller(repo)
        with pytest.raises(ValueError, match="no concept"):
            ctrl.start(chapter_id="empty", mode=LessonMode.TEACH)
    finally:
        repo.close()


# ---- submit_answer tiers -------------------------------------------------


def test_correct_answer_praises(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(noul_queue=[0.95])
    ctrl = _controller(repo, decisions=jev)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    step = ctrl.submit_answer(state, "roots stems and leaves")
    assert step.stage is LessonStage.PRAISE_AND_NEXT
    assert step.judgement is not None
    assert step.judgement.tier is JudgementTier.CORRECT
    assert len(jev.seen) == 1


def test_partial_answer_still_moves_on_with_clarification(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(noul_queue=[0.55])  # between 0.40 and 0.70
    ctrl = _controller(repo, decisions=jev)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    step = ctrl.submit_answer(state, "umm roots?")
    assert step.stage is LessonStage.PRAISE_AND_NEXT
    assert step.judgement.tier is JudgementTier.PARTIAL
    # The clarification uses the key_points.
    assert "point 0a" in step.speech_text or "point 0b" in step.speech_text


def test_wrong_answer_first_try_clarifies_and_keeps_concept(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(noul_queue=[0.1])
    ctrl = _controller(repo, decisions=jev, max_attempts=2)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    step = ctrl.submit_answer(state, "a potato")
    assert step.stage is LessonStage.CLARIFY
    assert step.judgement.tier is JudgementTier.WRONG
    assert step.prompt_text == state.last_prompt


def test_wrong_answer_at_budget_moves_on_gently(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(noul_queue=[0.1, 0.1])
    ctrl = _controller(repo, decisions=jev, max_attempts=2)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    ctrl.submit_answer(state, "wrong 1")
    step = ctrl.submit_answer(state, "wrong 2")
    assert step.stage is LessonStage.PRAISE_AND_NEXT
    assert "chapter says" in step.speech_text.lower()


def test_jev_error_fails_closed(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(error_queue=[JevError("boom")])
    ctrl = _controller(repo, decisions=jev)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    step = ctrl.submit_answer(state, "anything")
    assert step.stage is LessonStage.FAILED
    assert "boom" in (step.error or "")


# ---- advance ------------------------------------------------------------


def test_advance_after_praise_moves_to_next_concept(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(noul_queue=[0.95, 0.95, 0.95])
    gen = FakeGenerator(
        answers_queue=["intro 1", "intro 2", "intro 3"],
        rewrites_queue=["Q1?", "Q2?", "Q3?"],
    )
    ctrl = _controller(repo, generator=gen, decisions=jev)
    state, step0 = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    assert step0.concept_index == 0

    _ = ctrl.submit_answer(state, "a1")
    step1 = ctrl.advance(state)
    assert step1.concept_index == 1
    assert step1.stage is LessonStage.PRESENT
    assert "intro 2" in step1.speech_text
    assert "Q2?" in step1.speech_text


def test_advance_after_clarify_stays_on_same_concept(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(noul_queue=[0.1])
    gen = FakeGenerator(answers_queue=["intro"], rewrites_queue=["Q?"])
    ctrl = _controller(repo, generator=gen, decisions=jev, max_attempts=2)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)
    step = ctrl.submit_answer(state, "wrong")
    assert step.stage is LessonStage.CLARIFY
    nxt = ctrl.advance(state)
    assert nxt.stage is LessonStage.LISTEN
    assert nxt.concept_index == 0  # still the same concept


def test_lesson_completes_after_last_concept(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions(noul_queue=[0.95, 0.95, 0.95])
    gen = FakeGenerator(
        answers_queue=["intro 1", "intro 2", "intro 3"],
        rewrites_queue=["Q1?", "Q2?", "Q3?"],
    )
    ctrl = _controller(repo, generator=gen, decisions=jev)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)

    for _ in range(3):
        ctrl.submit_answer(state, "right")
        step = ctrl.advance(state)

    assert step.stage is LessonStage.LESSON_COMPLETE


def test_reveal_answer_moves_on_without_judging(seeded) -> None:
    repo, _ = seeded
    jev = FakeDecisions()  # judge_answer must NOT be called
    gen = FakeGenerator(answers_queue=["intro"], rewrites_queue=["Q?"])
    ctrl = _controller(repo, generator=gen, decisions=jev)
    state, _ = ctrl.start(chapter_id="ch1", mode=LessonMode.TEACH)

    step = ctrl.reveal_answer(state)
    assert step.stage is LessonStage.PRAISE_AND_NEXT
    assert "chapter says" in step.speech_text.lower()
    assert jev.seen == []  # no judgement call
    assert state.attempts >= 2  # marked exhausted so advance() moves on


def test_thresholds_validated() -> None:
    repo_stub = None  # not needed for this check
    with pytest.raises(ValueError, match="thresholds"):
        LessonController(
            repo=repo_stub,
            answer_generator=FakeGenerator(),
            decisions_client=FakeDecisions(),
            correct_threshold=0.4,
            partial_threshold=0.6,
        )
