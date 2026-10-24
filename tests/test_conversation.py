"""Controller + context manager tests using a fake retriever and fake Jev."""
from __future__ import annotations

from dataclasses import dataclass, field

from conversation.clarification import combine_question_with_clarification
from conversation.context_manager import ConversationContext
from conversation.controller import ConversationController, TurnStatus
from decision.decision_policy import DecisionPolicy
from decision.jev_client import JevDecision, JevError, JevVerification
from llm.answer_generator import AnswerGenerationError, GeneratedAnswer
from retrieval.retriever import RetrievedChunk
from storage.models import OcrQualityStatus


def _chunk(page: int = 1, text: str = "lorem", score: float = 0.4) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=page,
        chapter_id="ch1",
        page_start=page,
        page_end=page,
        section_title=None,
        chunk_text=text,
        score=score,
        ocr_quality_status=OcrQualityStatus.OK,
    )


@dataclass
class FakeRetriever:
    hits_by_topk: dict[int, list[RetrievedChunk]] = field(default_factory=dict)
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def search(self, chapter_id: str, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append((chapter_id, query, top_k))
        return self.hits_by_topk.get(top_k, next(iter(self.hits_by_topk.values()), []))


@dataclass
class FakeJev:
    decisions: list[JevDecision | Exception] = field(default_factory=list)
    verifications: list[JevVerification | Exception] = field(default_factory=list)
    seen: list[dict] = field(default_factory=list)
    seen_verifies: list[dict] = field(default_factory=list)

    def decide(self, *, question, passages, context_messages=()):
        self.seen.append(
            {"question": question, "n_passages": len(passages), "context": list(context_messages)}
        )
        item = self.decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def verify_answer(self, *, question, passages, answer):
        self.seen_verifies.append({"question": question, "answer": answer})
        item = self.verifications.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _decision(**kw) -> JevDecision:
    defaults = {
        "question_clear": True,
        "evidence_relevant": True,
        "evidence_sufficient": True,
        "decision_confidence": 0.8,
        "decision_reason": "",
        "raw": {},
    }
    defaults.update(kw)
    return JevDecision(**defaults)


@dataclass
class FakeGenerator:
    outputs: list[GeneratedAnswer | Exception] = field(default_factory=list)
    seen: list[dict] = field(default_factory=list)

    def generate(self, *, question, passages, context_messages=(), revision_feedback=None):
        self.seen.append(
            {
                "question": question,
                "n_passages": len(passages),
                "revision_feedback": revision_feedback,
            }
        )
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _controller(
    retriever,
    jev,
    max_attempts: int = 2,
    generator=None,
    verify_answers: bool = False,
    qa_matcher=None,
) -> ConversationController:
    return ConversationController(
        retriever=retriever,
        jev_client=jev,
        policy=DecisionPolicy(max_retrieval_attempts=max_attempts),
        base_top_k=5,
        expand_step=5,
        answer_generator=generator,
        verify_answers=verify_answers,
        qa_matcher=qa_matcher,
    )


class _FakeQAMatcher:
    def __init__(self, next_match):
        self._next = next_match
        self.calls = []

    def find_cached_answer(self, chapter_id, question):
        self.calls.append((chapter_id, question))
        return self._next


def _verification(
    supported: bool = True, confidence: float = 0.8, reason: str = ""
):
    from decision.jev_client import JevVerification

    return JevVerification(
        supported=supported, confidence=confidence, reason=reason, raw={}
    )


# ---- controller tests ---------------------------------------------------


def test_accepted_evidence_returns_answer_ready() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk()]})
    jev = FakeJev(decisions=[_decision()])
    ctrl = _controller(retriever, jev)

    result = ctrl.run_turn(
        chapter_id="ch1", question="what?", context=ConversationContext()
    )
    assert result.status == TurnStatus.ANSWER_READY
    assert result.attempts == 1
    assert len(result.passages) == 1
    assert result.decision and result.decision.accepted


def test_unclear_question_asks_clarification() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk()]})
    jev = FakeJev(
        decisions=[_decision(question_clear=False, decision_reason="Which plant do you mean?")]
    )
    ctrl = _controller(retriever, jev)

    result = ctrl.run_turn(
        chapter_id="ch1", question="tell me more", context=ConversationContext()
    )
    assert result.status == TurnStatus.CLARIFY
    assert "Which plant" in result.message
    assert result.attempts == 1


def test_insufficient_evidence_expands_then_accepts() -> None:
    retriever = FakeRetriever(
        hits_by_topk={5: [_chunk(1, "narrow")], 10: [_chunk(1), _chunk(2), _chunk(3)]}
    )
    jev = FakeJev(
        decisions=[
            _decision(evidence_sufficient=False, decision_reason="need more"),
            _decision(),
        ]
    )
    ctrl = _controller(retriever, jev)

    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())
    assert result.status == TurnStatus.ANSWER_READY
    assert result.attempts == 2
    assert retriever.calls == [("ch1", "q?", 5), ("ch1", "q?", 10)]


def test_insufficient_then_still_bad_refuses() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk()], 10: [_chunk()]})
    jev = FakeJev(
        decisions=[
            _decision(evidence_relevant=False, decision_reason="off topic"),
            _decision(evidence_relevant=False, decision_reason="still off topic"),
        ]
    )
    ctrl = _controller(retriever, jev)

    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())
    assert result.status == TurnStatus.REFUSE
    assert result.attempts == 2
    assert "chapter" in result.message.lower()


def test_jev_error_fails_closed_without_asking_again() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk()]})
    jev = FakeJev(decisions=[JevError("boom")])
    ctrl = _controller(retriever, jev)

    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())
    assert result.status == TurnStatus.FAIL_CLOSED
    assert "verify" in result.message.lower()
    assert result.attempts == 1


def test_passes_conversation_context_to_jev() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk()]})
    jev = FakeJev(decisions=[_decision()])
    ctx = ConversationContext()
    ctx.add_student("What is photosynthesis?")
    ctx.add_tutor("Plants making food.")

    ctrl = _controller(retriever, jev)
    ctrl.run_turn(chapter_id="ch1", question="tell me more", context=ctx)

    assert jev.seen[0]["context"] == [
        ("student", "What is photosynthesis?"),
        ("tutor", "Plants making food."),
    ]


# ---- answer-generator wiring --------------------------------------------


def _answer(answerable: bool = True, text: str = "Plants make food.") -> GeneratedAnswer:
    return GeneratedAnswer(
        answerable=answerable,
        answer=text,
        supporting_page_numbers=[1],
        supporting_passage_ids=[1],
        raw={},
    )


def test_accept_with_generator_returns_answered() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])
    gen = FakeGenerator(outputs=[_answer()])

    ctrl = _controller(retriever, jev, generator=gen)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.ANSWERED
    assert result.message == "Plants make food."
    assert result.generated_answer and result.generated_answer.answerable
    assert len(gen.seen) == 1


def test_generator_answerable_false_is_refuse() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])
    gen = FakeGenerator(
        outputs=[_answer(answerable=False, text="The chapter does not cover this.")]
    )

    ctrl = _controller(retriever, jev, generator=gen)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.REFUSE
    assert "does not cover" in result.message
    assert result.generated_answer is not None


def test_generator_error_fails_closed() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])
    gen = FakeGenerator(outputs=[AnswerGenerationError("boom")])

    ctrl = _controller(retriever, jev, generator=gen)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.FAIL_CLOSED
    assert "verify" in result.message.lower()


def test_without_generator_returns_answer_ready() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])

    ctrl = _controller(retriever, jev)  # no generator
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())
    assert result.status == TurnStatus.ANSWER_READY
    assert result.generated_answer is None


# ---- M6: answer verification --------------------------------------------


def test_verify_supported_answer_is_answered() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()], verifications=[_verification(supported=True)])
    gen = FakeGenerator(outputs=[_answer()])

    ctrl = _controller(retriever, jev, generator=gen, verify_answers=True)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.ANSWERED
    assert result.verification is not None and result.verification.supported
    assert result.regenerated is False
    assert len(jev.seen_verifies) == 1
    assert len(gen.seen) == 1  # no regeneration


def test_verify_unsupported_then_regen_supported_is_answered() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(
        decisions=[_decision()],
        verifications=[
            _verification(supported=False, reason="added facts not in passages"),
            _verification(supported=True, confidence=0.9),
        ],
    )
    gen = FakeGenerator(
        outputs=[
            _answer(text="Plants grow on Mars."),
            _answer(text="Plants grow and have roots."),
        ]
    )

    ctrl = _controller(retriever, jev, generator=gen, verify_answers=True)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.ANSWERED
    assert result.regenerated is True
    assert result.message == "Plants grow and have roots."
    # Second generation received the verifier's reason as feedback.
    assert gen.seen[1]["revision_feedback"] == "added facts not in passages"
    assert len(jev.seen_verifies) == 2


def test_verify_unsupported_twice_refuses() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(
        decisions=[_decision()],
        verifications=[
            _verification(supported=False, reason="first bad"),
            _verification(supported=False, reason="still bad"),
        ],
    )
    gen = FakeGenerator(outputs=[_answer(text="first"), _answer(text="second")])

    ctrl = _controller(retriever, jev, generator=gen, verify_answers=True)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.REFUSE
    # Student is not shown either unverified answer.
    assert "first" not in result.message
    assert "second" not in result.message
    assert "textbook" in result.message.lower()
    assert result.regenerated is True


def test_verify_error_fails_closed() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()], verifications=[JevError("boom")])
    gen = FakeGenerator(outputs=[_answer()])

    ctrl = _controller(retriever, jev, generator=gen, verify_answers=True)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.FAIL_CLOSED
    assert "verify" in result.message.lower()


def test_verify_off_skips_verifier_calls() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])  # no verifications queued
    gen = FakeGenerator(outputs=[_answer()])

    ctrl = _controller(retriever, jev, generator=gen, verify_answers=False)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.ANSWERED
    assert jev.seen_verifies == []
    assert result.verification is None


def test_regen_answerable_false_refuses_without_second_verify() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(
        decisions=[_decision()],
        verifications=[_verification(supported=False, reason="x")],
    )
    gen = FakeGenerator(
        outputs=[_answer(text="first"), _answer(answerable=False, text="I can't answer.")]
    )

    ctrl = _controller(retriever, jev, generator=gen, verify_answers=True)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.REFUSE
    assert result.regenerated is True
    assert len(jev.seen_verifies) == 1  # second verify skipped


# ---- follow-up loop (post-M6) --------------------------------------------


def _answer_with_followup(
    answerable: bool = True,
    text: str = "Plants make food.",
    follow_up: str | None = None,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answerable=answerable,
        answer=text,
        supporting_page_numbers=[1],
        supporting_passage_ids=[1],
        raw={},
        follow_up_question=follow_up,
    )


def test_answerable_true_with_followup_carries_through() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])
    gen = FakeGenerator(outputs=[_answer_with_followup(follow_up="Which part?")])

    ctrl = _controller(retriever, jev, generator=gen)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.ANSWERED
    assert result.follow_up_question == "Which part?"


def test_answerable_false_with_followup_becomes_clarify() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])
    gen = FakeGenerator(
        outputs=[
            _answer_with_followup(
                answerable=False,
                text="The chapter doesn't say.",
                follow_up="Do you mean the leaves or the roots?",
            )
        ]
    )

    ctrl = _controller(retriever, jev, generator=gen)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.CLARIFY
    assert result.message == "Do you mean the leaves or the roots?"
    assert result.follow_up_question == "Do you mean the leaves or the roots?"


def test_answerable_false_without_followup_still_refuses() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])
    gen = FakeGenerator(
        outputs=[_answer_with_followup(answerable=False, text="The chapter doesn't say.")]
    )

    ctrl = _controller(retriever, jev, generator=gen)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.REFUSE


def test_verifier_rejects_regen_offers_clarify_when_followup_present() -> None:
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(
        decisions=[_decision()],
        verifications=[
            _verification(supported=False, reason="added facts"),
            _verification(supported=False, reason="still added facts"),
        ],
    )
    gen = FakeGenerator(
        outputs=[
            _answer_with_followup(text="first"),
            _answer_with_followup(
                text="second with inference",
                follow_up="Should I just stick to what the chapter says?",
            ),
        ]
    )

    ctrl = _controller(retriever, jev, generator=gen, verify_answers=True)
    result = ctrl.run_turn(chapter_id="ch1", question="q?", context=ConversationContext())

    assert result.status == TurnStatus.CLARIFY
    assert "stick to what the chapter says" in result.message
    assert result.regenerated is True


# ---- context manager tests -----------------------------------------------


def test_context_bounded_by_max_turns() -> None:
    ctx = ConversationContext(max_turns=2)
    ctx.add_student("one")
    ctx.add_tutor("two")
    ctx.add_student("three")
    recent = ctx.recent()
    assert len(recent) == 2
    assert recent[-1] == ("student", "three")


# ---- clarification helper ------------------------------------------------


def test_combine_question_with_clarification() -> None:
    assert (
        combine_question_with_clarification("tell me about plants", "specifically the roots")
        == "tell me about plants (specifically the roots)"
    )
    assert combine_question_with_clarification("x", "") == "x"
    assert combine_question_with_clarification("", "y") == "y"


# ---- Q&A cache integration (cache-first path) ---------------------------


def test_cache_hit_returns_answered_without_calling_jev() -> None:
    from datetime import datetime, timezone
    from retrieval.qa_matcher import QAMatch
    from storage.models import QAPairRow

    pair = QAPairRow(
        qa_id=42, chapter_id="ch1",
        question="what are the parts of a plant?",
        answer="Roots, stems, leaves.",
        page_numbers=[1, 2], source_chunk_ids=[1],
        faiss_id=42, created_at=datetime.now(timezone.utc),
    )
    match = QAMatch(qa_pair=pair, confidence=0.9, reason="match",
                    candidates_considered=3, raw={})
    matcher = _FakeQAMatcher(next_match=match)
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[])  # must NOT be consulted on cache hit

    ctrl = _controller(retriever, jev, qa_matcher=matcher)
    result = ctrl.run_turn(
        chapter_id="ch1", question="what are the parts of a plant?",
        context=ConversationContext(),
    )
    assert result.status == TurnStatus.ANSWERED
    assert result.message == "Roots, stems, leaves."
    assert result.qa_cache_hit is match
    assert jev.seen == []  # Jev gate was skipped
    assert matcher.calls == [("ch1", "what are the parts of a plant?")]


def test_cache_miss_falls_through_to_full_pipeline() -> None:
    matcher = _FakeQAMatcher(next_match=None)
    retriever = FakeRetriever(hits_by_topk={5: [_chunk(1)]})
    jev = FakeJev(decisions=[_decision()])

    ctrl = _controller(retriever, jev, qa_matcher=matcher)
    result = ctrl.run_turn(
        chapter_id="ch1", question="q?", context=ConversationContext()
    )
    assert result.status == TurnStatus.ANSWER_READY  # no generator wired here
    assert result.qa_cache_hit is None
    assert len(jev.seen) == 1
