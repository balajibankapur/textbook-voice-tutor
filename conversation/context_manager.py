"""Short-horizon conversation history for follow-up questions.

Spec §9: "If the question refers to an earlier answer, resolve the reference
using conversation context before performing retrieval."
Spec §12 warning: do NOT treat prior assistant answers as authoritative
evidence; only use context to disambiguate the student's intent.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    role: str  # "student" | "tutor"
    content: str


class ConversationContext:
    def __init__(self, max_turns: int = 6) -> None:
        if max_turns < 2:
            raise ValueError("max_turns must be >= 2")
        self.max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add_student(self, content: str) -> None:
        self._turns.append(Turn(role="student", content=content))

    def add_tutor(self, content: str) -> None:
        self._turns.append(Turn(role="tutor", content=content))

    def recent(self) -> list[tuple[str, str]]:
        """List of (role, content) pairs, oldest first, excluding the latest student turn."""
        return [(t.role, t.content) for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()
