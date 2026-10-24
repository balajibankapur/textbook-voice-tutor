"""Helpers for building a combined query from an original question + the
student's clarification response. Kept tiny on purpose — the LLM-driven
query rewrite is a later concern.
"""
from __future__ import annotations


def combine_question_with_clarification(original: str, clarification_answer: str) -> str:
    """Produce a single retrieval query that carries both the original intent
    and the clarification the student just provided.
    """
    original = original.strip()
    clarification_answer = clarification_answer.strip()
    if not clarification_answer:
        return original
    if not original:
        return clarification_answer
    return f"{original} ({clarification_answer})"
