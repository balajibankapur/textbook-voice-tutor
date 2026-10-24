"""Jev decision client (OpenRouter chat completions, typesafe/jev-* models).

Jev is a standard chat model on OpenRouter — it does not expose native
decision/probability fields. We prompt it with the question + retrieved
passages, require strict JSON output, and parse it into `JevDecision`.

The decision has three concepts the spec asks about (§10):
- question_clear      : is the student's question understandable?
- evidence_relevant   : do retrieved passages address the question?
- evidence_sufficient : do passages contain enough to answer from the textbook?

Spec §18 fail-closed rule: on timeout, HTTP error, or invalid JSON, this module
raises `JevError`. Callers (decision_policy) refuse to proceed rather than
guessing — the student is told the answer could not be verified.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
RETRY_STATUSES = {429, 500, 502, 503, 504, 524, 529}


class JevError(RuntimeError):
    """Raised when Jev cannot be reached or returns an unusable response."""


@dataclass
class JevDecision:
    question_clear: bool
    evidence_relevant: bool
    evidence_sufficient: bool
    decision_confidence: float  # Jev's self-report, 0.0–1.0 — NOT calibrated.
    decision_reason: str
    raw: dict[str, Any]  # kept for logging/evaluation

    @property
    def accepted(self) -> bool:
        return self.question_clear and self.evidence_relevant and self.evidence_sufficient


@dataclass
class JevVerification:
    """Result of asking Jev whether a generated answer is supported by the evidence."""
    supported: bool
    confidence: float
    reason: str
    raw: dict[str, Any]


_SYSTEM_PROMPT = (
    "You evaluate whether a student's textbook question can be answered from "
    "retrieved passages. You never answer the question yourself. You respond "
    "with ONLY a JSON object matching this exact schema and nothing else:\n"
    "{\n"
    '  "question_clear": true | false,\n'
    '  "evidence_relevant": true | false,\n'
    '  "evidence_sufficient": true | false,\n'
    '  "confidence": <number from 0.0 to 1.0>,\n'
    '  "reason": "<one short sentence>"\n'
    "}\n"
    "\n"
    "Definitions:\n"
    "- question_clear: the student's intent is specific enough to answer. If the "
    "question is vague, ambiguous, or missing a referent, set this to false and "
    "put the clarification you would ask in `reason`.\n"
    "- evidence_relevant: the retrieved passages address the question's topic.\n"
    "- evidence_sufficient: the passages contain enough information to answer "
    "the question from the textbook alone. If critical facts are missing, set "
    "this to false.\n"
    "- confidence: your overall confidence in the three judgments.\n"
    "- reason: one short sentence; if clarifying, phrase it as the question to "
    "ask the student.\n"
    "\n"
    "Rules: output valid JSON only, no markdown, no code fences, no commentary "
    "outside the JSON object. Do not try to answer the student's question."
)


_VERIFY_SYSTEM_PROMPT = (
    "You check whether a tutor's answer is supported by textbook passages. "
    "You do not evaluate style, completeness, or kindness — only whether every "
    "factual claim in the answer can be verified from the passages. An answer "
    "that adds new facts not present in the passages is NOT supported, even if "
    "those facts happen to be true in the world. Reasonable paraphrases and "
    "simplifications of the passages ARE supported.\n"
    "\n"
    "Respond with ONLY a JSON object matching this exact schema:\n"
    "{\n"
    '  "supported": true | false,\n'
    '  "confidence": <number from 0.0 to 1.0>,\n'
    '  "reason": "<one short sentence; if unsupported, name the unsupported claim>"\n'
    "}\n"
    "No markdown, no code fences, no prose outside the JSON object."
)


def _format_context(context_messages: Sequence[tuple[str, str]]) -> str:
    if not context_messages:
        return "(no prior turns)"
    return "\n".join(f"{role}: {content}" for role, content in context_messages)


def _format_passages(passages: Sequence[RetrievedChunk]) -> str:
    if not passages:
        return "(no passages retrieved)"
    lines: list[str] = []
    for i, p in enumerate(passages, start=1):
        pages = (
            f"pages {p.page_start}" + (f"–{p.page_end}" if p.page_end != p.page_start else "")
        )
        section = f" · section: {p.section_title}" if p.section_title else ""
        lines.append(f"[{i}] {pages}{section} (score {p.score:.3f})\n{p.chunk_text}")
    return "\n\n".join(lines)


def _build_user_prompt(
    question: str,
    context_messages: Sequence[tuple[str, str]],
    passages: Sequence[RetrievedChunk],
) -> str:
    return (
        f"Student question:\n{question}\n\n"
        f"Prior conversation (most recent last):\n{_format_context(context_messages)}\n\n"
        f"Retrieved textbook passages:\n{_format_passages(passages)}\n\n"
        "Produce the JSON decision."
    )


class JevClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "typesafe/jev-router",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise JevError("Jev client requires a non-empty API key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def decide(
        self,
        *,
        question: str,
        passages: Sequence[RetrievedChunk],
        context_messages: Sequence[tuple[str, str]] = (),
    ) -> JevDecision:
        """Ask Jev whether the retrieved passages can answer the question."""
        user_prompt = _build_user_prompt(question, context_messages, passages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        parsed = self._chat_json(payload, required=_REQUIRED_FIELDS)
        return JevDecision(
            question_clear=bool(parsed["question_clear"]),
            evidence_relevant=bool(parsed["evidence_relevant"]),
            evidence_sufficient=bool(parsed["evidence_sufficient"]),
            decision_confidence=float(parsed.get("confidence", 0.0)),
            decision_reason=str(parsed.get("reason", "")).strip(),
            raw=parsed,
        )

    def verify_answer(
        self,
        *,
        question: str,
        passages: Sequence[RetrievedChunk],
        answer: str,
    ) -> JevVerification:
        """Ask Jev whether the generated answer is supported by the passages."""
        if not answer.strip():
            raise JevError("cannot verify an empty answer")

        user_prompt = (
            f"Student question:\n{question}\n\n"
            f"Textbook passages (the only acceptable source of facts):\n"
            f"{_format_passages(passages)}\n\n"
            f"Tutor's answer:\n{answer}\n\n"
            "Produce the JSON verification."
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        parsed = self._chat_json(payload, required=_VERIFY_REQUIRED_FIELDS)
        return JevVerification(
            supported=bool(parsed["supported"]),
            confidence=float(parsed.get("confidence", 0.0)),
            reason=str(parsed.get("reason", "")).strip(),
            raw=parsed,
        )

    def _chat_json(
        self, payload: dict[str, Any], required: tuple[str, ...]
    ) -> dict[str, Any]:
        """Shared send-and-parse-JSON path for decide / verify."""
        data = self._post_with_retries(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JevError(f"unexpected response shape: {exc}") from exc
        return _parse_decision_json(content, required=required)

    # ---- internals -------------------------------------------------------

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
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
                logger.warning("Jev request network error (attempt %s): %s", attempt + 1, last_error)
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                raise JevError(f"network error after {self.max_retries + 1} tries: {last_error}") from exc

            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRY_STATUSES and attempt < self.max_retries:
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                logger.warning("Jev retryable status (attempt %s): %s", attempt + 1, last_error)
                time.sleep(_backoff(attempt))
                continue
            raise JevError(f"HTTP {r.status_code}: {r.text[:400]}")

        # Unreachable, but keeps type checkers happy.
        raise JevError(f"exhausted retries: {last_error}")


def _backoff(attempt: int) -> float:
    return min(1.0 * (2**attempt), 8.0)


_REQUIRED_FIELDS: tuple[str, ...] = (
    "question_clear",
    "evidence_relevant",
    "evidence_sufficient",
)
_VERIFY_REQUIRED_FIELDS: tuple[str, ...] = ("supported",)


def _parse_decision_json(
    content: str | None, required: tuple[str, ...] = _REQUIRED_FIELDS
) -> dict[str, Any]:
    """Parse Jev's JSON content; strip accidental code fences if present."""
    if content is None or not str(content).strip():
        raise JevError(
            "Jev returned empty content — the model may have refused, truncated, "
            "or spent its token budget on reasoning. Try again or raise max_tokens."
        )
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # drop language tag like `json\n...`
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JevError(f"Jev returned non-JSON content: {content[:200]}") from exc
    if not isinstance(parsed, dict):
        raise JevError(f"Jev JSON is not an object: {type(parsed).__name__}")
    missing = [k for k in required if k not in parsed]
    if missing:
        raise JevError(f"Jev JSON missing fields: {missing}; got keys {list(parsed.keys())}")
    return parsed
