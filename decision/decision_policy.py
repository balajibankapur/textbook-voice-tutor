"""Deterministic policy that converts a JevDecision into a next action.

Spec §11:
- unclear question → ask ONE clarification question, wait for student
- irrelevant/insufficient evidence → expand retrieval up to N attempts
- accepted → send to LLM
- gate error → refuse (fail-closed, spec §18)
- out of attempts → tell the student the chapter does not cover it
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from decision.jev_client import JevDecision


class Action(str, Enum):
    ANSWER = "answer"              # evidence accepted; generate answer
    CLARIFY = "clarify"            # ask the student a question, wait
    EXPAND_RETRIEVAL = "expand"    # try retrieval again with more / wider
    REFUSE = "refuse"              # chapter cannot support this question
    FAIL_CLOSED = "fail_closed"    # Jev failed; cannot verify, do not answer


@dataclass
class PolicyOutcome:
    action: Action
    message: str  # student-facing when CLARIFY/REFUSE/FAIL_CLOSED; dev-facing otherwise


class DecisionPolicy:
    def __init__(self, max_retrieval_attempts: int = 2) -> None:
        if max_retrieval_attempts < 1:
            raise ValueError("max_retrieval_attempts must be >= 1")
        self.max_retrieval_attempts = max_retrieval_attempts

    def decide(self, decision: JevDecision, attempt: int) -> PolicyOutcome:
        """`attempt` is the 1-indexed retrieval attempt that produced the current evidence."""
        if not decision.question_clear:
            reason = decision.decision_reason or "Could you say a bit more about what you're asking?"
            return PolicyOutcome(action=Action.CLARIFY, message=reason)

        if decision.accepted:
            return PolicyOutcome(action=Action.ANSWER, message="evidence accepted")

        # Clear question, but evidence is irrelevant or insufficient.
        if attempt < self.max_retrieval_attempts:
            return PolicyOutcome(
                action=Action.EXPAND_RETRIEVAL,
                message=(
                    f"attempt {attempt}/{self.max_retrieval_attempts}: expand retrieval - "
                    f"{decision.decision_reason or 'evidence not sufficient'}"
                ),
            )

        # Out of retrieval attempts — tell the student the chapter does not cover this.
        return PolicyOutcome(
            action=Action.REFUSE,
            message=(
                "I couldn't find enough in this chapter to answer that. "
                "The textbook may not cover it, or it may be in a different chapter."
            ),
        )

    def on_jev_error(self) -> PolicyOutcome:
        return PolicyOutcome(
            action=Action.FAIL_CLOSED,
            message=(
                "I couldn't verify the answer against the textbook right now. "
                "Please try again in a moment."
            ),
        )
