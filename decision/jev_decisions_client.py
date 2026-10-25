"""Real Jev decisions API client (/api/alpha/decisions).

Jev models like `typesafe/jev-1.14` do NOT accept chat completions — they
require a decision-structured request with `state` + `questions` and return
calibrated probabilities (`noul` for truth/presence, `choice` for selection,
`score` for weighted criteria).

This client exposes the same surface as `JevClient` (`decide`, `verify_answer`)
plus a new `match_choice` for the Q&A matcher, so controllers can swap it in.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from decision.jev_client import JevDecision, JevError, JevVerification
from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
RETRY_STATUSES = {429, 500, 502, 503, 504, 524, 529}
DEFAULT_NOUL_THRESHOLD = 0.5


@dataclass
class ChoiceMatch:
    """Result of a `choice` decision — the chosen key with its probability."""
    chosen_key: str
    probability: float
    confidence: float
    probabilities: dict[str, float]
    raw: dict[str, Any]


def _format_passages(passages: Sequence[RetrievedChunk]) -> str:
    if not passages:
        return "(no passages retrieved)"
    lines: list[str] = []
    for i, p in enumerate(passages, start=1):
        pages = (
            f"pages {p.page_start}" + (f"-{p.page_end}" if p.page_end != p.page_start else "")
        )
        section = f" | section: {p.section_title}" if p.section_title else ""
        lines.append(f"[{i}] ({pages}{section})\n{p.chunk_text}")
    return "\n\n".join(lines)


def _format_context(context_messages: Sequence[tuple[str, str]]) -> str:
    if not context_messages:
        return ""
    return "\n".join(f"{role}: {content}" for role, content in context_messages)


class JevDecisionsClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/alpha",
        model: str = "typesafe/jev-1.14",
        noul_threshold: float = DEFAULT_NOUL_THRESHOLD,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise JevError("JevDecisionsClient requires a non-empty API key")
        if not 0.0 <= noul_threshold <= 1.0:
            raise JevError("noul_threshold must be in [0, 1]")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.noul_threshold = noul_threshold
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "JevDecisionsClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- gate -----------------------------------------------------------

    def decide(
        self,
        *,
        question: str,
        passages: Sequence[RetrievedChunk],
        context_messages: Sequence[tuple[str, str]] = (),
    ) -> JevDecision:
        """Three parallel `noul` decisions: clarity, relevance, sufficiency."""
        state = self._build_gate_state(question, context_messages, passages)
        questions = {
            "clear": {
                "type": "noul",
                "instructions": (
                    "The student's question is specific and understandable enough "
                    "to answer directly from a textbook. The question is NOT vague, "
                    "ambiguous, or missing a referent."
                ),
            },
            "relevant": {
                "type": "noul",
                "instructions": (
                    "The retrieved textbook passages address the topic the "
                    "student is asking about."
                ),
            },
            "sufficient": {
                "type": "noul",
                "instructions": (
                    "The retrieved textbook passages contain enough information to "
                    "answer the student's question completely, using the textbook alone."
                ),
            },
        }
        answers = self._post_decisions(state=state, questions=questions)

        clear_p = _get_noul(answers, "clear")
        rel_p = _get_noul(answers, "relevant")
        suff_p = _get_noul(answers, "sufficient")

        return JevDecision(
            question_clear=clear_p >= self.noul_threshold,
            evidence_relevant=rel_p >= self.noul_threshold,
            evidence_sufficient=suff_p >= self.noul_threshold,
            decision_confidence=(clear_p + rel_p + suff_p) / 3.0,
            decision_reason="",  # noul doesn't carry a textual reason
            raw={
                "nouls": {"clear": clear_p, "relevant": rel_p, "sufficient": suff_p},
                "threshold": self.noul_threshold,
            },
        )

    # ---- verifier -------------------------------------------------------

    def verify_answer(
        self,
        *,
        question: str,
        passages: Sequence[RetrievedChunk],
        answer: str,
    ) -> JevVerification:
        """One `noul` decision: is every factual claim supported by the passages?"""
        if not answer.strip():
            raise JevError("cannot verify an empty answer")

        state = (
            f"Student question:\n{question}\n\n"
            f"Textbook passages (only acceptable source of facts):\n"
            f"{_format_passages(passages)}\n\n"
            f"Tutor's answer to verify:\n{answer}"
        )
        questions = {
            "supported": {
                "type": "noul",
                "instructions": (
                    "Every factual claim in the tutor's answer is directly "
                    "supported by the textbook passages. The answer does NOT add "
                    "new facts (even true ones) that are not in the passages. "
                    "Reasonable paraphrases and simplifications count as supported."
                ),
            },
        }
        answers = self._post_decisions(state=state, questions=questions)
        supported_p = _get_noul(answers, "supported")
        return JevVerification(
            supported=supported_p >= self.noul_threshold,
            confidence=supported_p,
            reason="",  # noul doesn't carry a reason
            raw={"noul": supported_p, "threshold": self.noul_threshold},
        )

    # ---- lesson-mode judge ----------------------------------------------

    def judge_answer(
        self,
        *,
        concept_title: str,
        learning_objective: str,
        key_points: Sequence[str],
        student_answer: str,
    ) -> float:
        """Return a calibrated 0–1 probability that the student's answer
        correctly captures the concept's learning objective. Used by Lesson mode.
        """
        if not student_answer.strip():
            raise JevError("cannot judge an empty answer")
        key_points_block = "\n".join(f"- {kp}" for kp in key_points) or "(none)"
        state = (
            f"Concept: {concept_title}\n"
            f"Learning objective: {learning_objective}\n"
            f"Key points the student should capture (any phrasing is fine):\n"
            f"{key_points_block}\n\n"
            f"Student's answer:\n{student_answer}"
        )
        questions = {
            "correct": {
                "type": "noul",
                "instructions": (
                    "The student's answer correctly captures the concept's "
                    "learning objective. Paraphrases and simpler wording count "
                    "as correct. The answer does NOT need to use the exact key "
                    "points verbatim — it needs to capture the same substance."
                ),
            },
        }
        answers = self._post_decisions(state=state, questions=questions)
        return _get_noul(answers, "correct")

    # ---- Q&A matcher -----------------------------------------------------

    def match_choice(
        self,
        *,
        state_text: str,
        candidates: dict[str, str],
        instructions: str,
        include_none_option: bool = True,
    ) -> ChoiceMatch:
        """One `choice` decision — pick one of the named candidates (or 'none')."""
        if not candidates:
            raise JevError("match_choice requires at least one candidate")
        criteria = dict(candidates)
        if include_none_option:
            criteria["none"] = (
                "None of the candidates is a semantic match for the state question."
            )
        questions = {
            "match": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            },
        }
        answers = self._post_decisions(state=state_text, questions=questions)
        try:
            answer = answers["match"]
            chosen = str(answer["choice"])
            probabilities = {str(k): float(v) for k, v in answer.get("probabilities", {}).items()}
            confidence = float(answer.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise JevError(f"malformed choice response: {exc}") from exc
        return ChoiceMatch(
            chosen_key=chosen,
            probability=probabilities.get(chosen, 0.0),
            confidence=confidence,
            probabilities=probabilities,
            raw=answer,
        )

    # ---- internals ------------------------------------------------------

    def _build_gate_state(
        self,
        question: str,
        context_messages: Sequence[tuple[str, str]],
        passages: Sequence[RetrievedChunk],
    ) -> str:
        parts: list[str] = [f"Student question:\n{question}"]
        ctx = _format_context(context_messages)
        if ctx:
            parts.append(f"Prior conversation (most recent last):\n{ctx}")
        parts.append(f"Retrieved textbook passages:\n{_format_passages(passages)}")
        return "\n\n".join(parts)

    def _post_decisions(
        self, *, state: Any, questions: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /decisions and return the `answers` object (one entry per question)."""
        payload = {"model": self.model, "state": state, "questions": questions}
        data = self._post_with_retries(payload)
        try:
            answers = data["answers"]
        except (KeyError, TypeError) as exc:
            raise JevError(f"missing 'answers' in response: {exc}") from exc
        if not isinstance(answers, dict):
            raise JevError(f"'answers' is not an object: {type(answers).__name__}")
        return answers

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/decisions"
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("decisions request network error (attempt %s): %s", attempt + 1, last_error)
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                raise JevError(f"network error after {self.max_retries + 1} tries: {last_error}") from exc

            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRY_STATUSES and attempt < self.max_retries:
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                logger.warning("decisions retryable status (attempt %s): %s", attempt + 1, last_error)
                time.sleep(_backoff(attempt))
                continue
            raise JevError(f"HTTP {r.status_code}: {r.text[:400]}")

        raise JevError(f"exhausted retries: {last_error}")


def _get_noul(answers: dict[str, Any], key: str) -> float:
    try:
        entry = answers[key]
        if entry.get("type") != "noul":
            raise JevError(f"answer {key!r} is not a noul: {entry.get('type')}")
        return float(entry["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"malformed noul answer for {key!r}: {exc}") from exc


def _backoff(attempt: int) -> float:
    return min(1.0 * (2**attempt), 8.0)
