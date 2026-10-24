from decision.decision_policy import Action, DecisionPolicy
from decision.jev_client import JevDecision


def _decision(
    *,
    clear: bool = True,
    relevant: bool = True,
    sufficient: bool = True,
    confidence: float = 0.8,
    reason: str = "ok",
) -> JevDecision:
    return JevDecision(
        question_clear=clear,
        evidence_relevant=relevant,
        evidence_sufficient=sufficient,
        decision_confidence=confidence,
        decision_reason=reason,
        raw={},
    )


def test_accepts_when_all_three_true() -> None:
    p = DecisionPolicy()
    out = p.decide(_decision(), attempt=1)
    assert out.action == Action.ANSWER


def test_clarifies_when_question_unclear() -> None:
    p = DecisionPolicy()
    out = p.decide(_decision(clear=False, reason="Which planet do you mean?"), attempt=1)
    assert out.action == Action.CLARIFY
    assert "Which planet" in out.message


def test_expands_when_irrelevant_within_budget() -> None:
    p = DecisionPolicy(max_retrieval_attempts=2)
    out = p.decide(_decision(relevant=False), attempt=1)
    assert out.action == Action.EXPAND_RETRIEVAL


def test_expands_when_insufficient_within_budget() -> None:
    p = DecisionPolicy(max_retrieval_attempts=2)
    out = p.decide(_decision(sufficient=False), attempt=1)
    assert out.action == Action.EXPAND_RETRIEVAL


def test_refuses_when_budget_exhausted_and_still_bad() -> None:
    p = DecisionPolicy(max_retrieval_attempts=2)
    out = p.decide(_decision(sufficient=False), attempt=2)
    assert out.action == Action.REFUSE
    assert "chapter" in out.message.lower()


def test_clarify_takes_precedence_over_evidence() -> None:
    # Even if evidence looks fine, an unclear question clarifies first.
    p = DecisionPolicy()
    out = p.decide(_decision(clear=False, relevant=True, sufficient=True), attempt=2)
    assert out.action == Action.CLARIFY


def test_on_jev_error_fails_closed() -> None:
    p = DecisionPolicy()
    out = p.on_jev_error()
    assert out.action == Action.FAIL_CLOSED
    assert "verify" in out.message.lower()


def test_invalid_max_attempts_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        DecisionPolicy(max_retrieval_attempts=0)
