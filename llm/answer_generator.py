"""LLM-driven answer generation on accepted textbook evidence.

Called only after the Jev gate accepts (spec §12). The LLM receives the
student question, conversation context, and the accepted passages, and
returns a structured answer with page citations.

Response is validated before being handed on — invalid output raises
`AnswerGenerationError` so the controller can fail-closed per spec §18.
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

DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 2
RETRY_STATUSES = {429, 500, 502, 503, 504, 524, 529}


class AnswerGenerationError(RuntimeError):
    """Raised when the LLM is unreachable or returns an unusable response."""


@dataclass
class GeneratedAnswer:
    answerable: bool
    answer: str
    supporting_page_numbers: list[int]
    supporting_passage_ids: list[int]
    raw: dict[str, Any]
    follow_up_question: str | None = None


_SYSTEM_PROMPT = (
    "You are a friendly tutor helping a Grade 2 student (around 7 years old) "
    "understand her textbook. "
    "Use the supplied textbook passages as the ONLY source of facts.\n"
    "\n"
    "Grounding rules — these are strict:\n"
    "- Use only facts that are literally stated in the passages. Reasonable "
    "paraphrases, simpler wording, and shorter sentences are fine. Do not add "
    "explanatory details, examples, consequences, or benefits that are not in "
    "the passages, even if they are true in the world.\n"
    "- If you are tempted to add a 'because…' or 'so that…' clause that is not "
    "in the passages, leave it out and consider asking a follow-up question "
    "instead (see below).\n"
    "- Keep the spoken answer concise (one or two short sentences when possible).\n"
    "- Use conversation context only to understand what the student is asking; "
    "do not treat previous assistant answers as authoritative evidence.\n"
    "\n"
    "Follow-up behavior — use this when the passages only partially cover the "
    "student's question:\n"
    "- If the passages give you some of the answer but not all of it, put the "
    "grounded part in `answer` AND put ONE short follow-up question in "
    "`follow_up_question` that would help the student narrow their question or "
    "point you at a part of the chapter that covers the rest.\n"
    "- If the passages cover the question completely, leave `follow_up_question` "
    "as null.\n"
    "- If the passages do not support ANY grounded answer, set `answerable` to "
    "false, leave `answer` as a short honest message (e.g. 'The chapter doesn't "
    "say.'), and put a short clarifying question in `follow_up_question` that, "
    "if the student answers, might point you at a part of the chapter that does "
    "cover it. If no such clarifying question would help, set "
    "`follow_up_question` to null.\n"
    "\n"
    "Return ONLY a JSON object matching this exact schema, no markdown, no prose "
    "outside the JSON:\n"
    "{\n"
    '  "answerable": true | false,\n'
    '  "answer": "<text; short honest message if answerable is false>",\n'
    '  "supporting_page_numbers": [<int>, ...],\n'
    '  "supporting_passage_ids": [<int>, ...],\n'
    '  "follow_up_question": "<one short question to the student>" | null\n'
    "}"
)


def _format_passages(passages: Sequence[RetrievedChunk]) -> str:
    if not passages:
        return "(no passages)"
    out: list[str] = []
    for p in passages:
        pages = (
            f"page {p.page_start}"
            if p.page_start == p.page_end
            else f"pages {p.page_start}-{p.page_end}"
        )
        section = f" · {p.section_title}" if p.section_title else ""
        quality = (
            f" · OCR quality: {p.ocr_quality_status.value}"
            if p.ocr_quality_status.value not in {"ok", "not_applicable"}
            else ""
        )
        out.append(
            f"[id={p.chunk_id}] ({pages}{section}{quality})\n{p.chunk_text}"
        )
    return "\n\n".join(out)


def _format_context(context_messages: Sequence[tuple[str, str]]) -> str:
    if not context_messages:
        return "(no prior turns)"
    return "\n".join(f"{role}: {content}" for role, content in context_messages)


def _build_user_prompt(
    question: str,
    context_messages: Sequence[tuple[str, str]],
    passages: Sequence[RetrievedChunk],
) -> str:
    return (
        f"Student question:\n{question}\n\n"
        f"Prior conversation (most recent last):\n{_format_context(context_messages)}\n\n"
        f"Accepted textbook passages:\n{_format_passages(passages)}\n\n"
        "Produce the JSON answer."
    )


class AnswerGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise AnswerGenerationError("AnswerGenerator requires a non-empty API key")
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

    def __enter__(self) -> "AnswerGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def generate(
        self,
        *,
        question: str,
        passages: Sequence[RetrievedChunk],
        context_messages: Sequence[tuple[str, str]] = (),
        revision_feedback: str | None = None,
    ) -> GeneratedAnswer:
        """Produce an answer. `revision_feedback` is appended as a dev note when
        regenerating after a failed verification (M6)."""
        if not passages:
            raise AnswerGenerationError("refusing to generate: no passages supplied")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(question, context_messages, passages),
            },
        ]
        if revision_feedback:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Your previous answer was rejected by the verifier. "
                        f"Reason: {revision_feedback}. Rewrite the answer so that "
                        "every claim is directly supported by the supplied passages. "
                        "Do not add facts that are not in the passages."
                    ),
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }
        data = self._post_with_retries(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AnswerGenerationError(f"unexpected response shape: {exc}") from exc

        parsed = _parse_answer_json(content)
        return _validate_and_build(parsed, passages)

    def rewrite_question(
        self,
        *,
        original: str,
        follow_up: str,
        reply: str,
    ) -> str:
        """Fold a student's clarification reply into a single clean standalone question.

        Used by the conversation controller / UI when the student answers a Jev
        follow-up. Returning the raw concatenation would stack parentheticals;
        the LLM produces one well-formed question instead.
        """
        system = (
            "You rewrite a student's clarified question into a single, clear, "
            "standalone question. Given the original (possibly vague) question, "
            "the tutor's follow-up asking for clarification, and the student's "
            "reply, produce ONE clear, concise question combining them. Return "
            "ONLY the question text, with no quotes, prefix, label, or commentary."
        )
        user = (
            f"Original question: {original}\n"
            f"Tutor's follow-up: {follow_up}\n"
            f"Student's reply: {reply}\n\n"
            "Rewritten question:"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 120,
        }
        data = self._post_with_retries(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AnswerGenerationError(f"unexpected response shape: {exc}") from exc
        if content is None:
            raise AnswerGenerationError("rewriter returned null content")
        text = str(content).strip().strip('"').strip("'").strip()
        if not text:
            raise AnswerGenerationError("rewriter returned an empty string")
        return text

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
                logger.warning(
                    "LLM request network error (attempt %s): %s", attempt + 1, last_error
                )
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                raise AnswerGenerationError(
                    f"network error after {self.max_retries + 1} tries: {last_error}"
                ) from exc

            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRY_STATUSES and attempt < self.max_retries:
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                logger.warning(
                    "LLM retryable status (attempt %s): %s", attempt + 1, last_error
                )
                time.sleep(_backoff(attempt))
                continue
            raise AnswerGenerationError(f"HTTP {r.status_code}: {r.text[:400]}")

        raise AnswerGenerationError(f"exhausted retries: {last_error}")


def _backoff(attempt: int) -> float:
    return min(1.0 * (2**attempt), 8.0)


_REQUIRED_FIELDS = (
    "answerable",
    "answer",
    "supporting_page_numbers",
    "supporting_passage_ids",
)
# follow_up_question is optional — the LLM leaves it out when none is needed.


def _parse_answer_json(content: str | None) -> dict[str, Any]:
    if content is None or not str(content).strip():
        raise AnswerGenerationError(
            "LLM returned empty content — the model may have refused, truncated, "
            "or spent its token budget on reasoning. Try again or raise max_tokens."
        )
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AnswerGenerationError(
            f"LLM returned non-JSON content: {content[:200]}"
        ) from exc
    if not isinstance(parsed, dict):
        raise AnswerGenerationError(
            f"LLM JSON is not an object: {type(parsed).__name__}"
        )
    missing = [k for k in _REQUIRED_FIELDS if k not in parsed]
    if missing:
        raise AnswerGenerationError(
            f"LLM JSON missing fields: {missing}; got keys {list(parsed.keys())}"
        )
    return parsed


def _validate_and_build(
    parsed: dict[str, Any], passages: Sequence[RetrievedChunk]
) -> GeneratedAnswer:
    answerable = bool(parsed["answerable"])
    answer = str(parsed["answer"]).strip()
    page_numbers = _coerce_int_list(parsed.get("supporting_page_numbers"))
    passage_ids = _coerce_int_list(parsed.get("supporting_passage_ids"))

    if answerable and not answer:
        raise AnswerGenerationError("answerable=true but answer is empty")

    valid_pages = {p for passage in passages for p in range(passage.page_start, passage.page_end + 1)}
    valid_ids = {passage.chunk_id for passage in passages if passage.chunk_id is not None}

    # Drop citations outside the evidence set rather than failing — the LLM
    # sometimes includes a near-miss, but we don't want to let unsupported
    # citations through to the student.
    page_numbers = [p for p in page_numbers if p in valid_pages]
    passage_ids = [pid for pid in passage_ids if pid in valid_ids]

    if answerable and not page_numbers:
        # Fall back to the pages touched by the evidence set.
        page_numbers = sorted(valid_pages)

    follow_up_raw = parsed.get("follow_up_question")
    follow_up: str | None = None
    if isinstance(follow_up_raw, str) and follow_up_raw.strip():
        follow_up = follow_up_raw.strip()

    return GeneratedAnswer(
        answerable=answerable,
        answer=answer,
        supporting_page_numbers=page_numbers,
        supporting_passage_ids=passage_ids,
        raw=parsed,
        follow_up_question=follow_up,
    )


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out
