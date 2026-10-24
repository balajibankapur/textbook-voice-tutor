"""Generate a corpus of Q&A pairs from a chapter's chunks.

Spec note: this is the "cache the expensive thing" layer that lets us answer
common student questions without a live LLM call. Each pair carries source
pages so citations are preserved. Runs once per chapter at ingestion time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx

from storage.models import QAPairRow

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
# Higher = better coverage + more cache hits, lower = cheaper to generate.
# For a 300–500 token chunk, 8–12 produces solid coverage without the LLM
# repeating itself; beyond ~15 dedupe kicks in and extra pairs are wasted.
DEFAULT_PAIRS_PER_CHUNK = 10
# Needs headroom: 10 pairs × ~60 tokens each ≈ 600 tokens of output.
DEFAULT_MAX_TOKENS = 1500


class QAGenerationError(RuntimeError):
    """Raised when Q&A generation fails or produces an unusable response."""


@dataclass
class GeneratedPair:
    question: str
    answer: str
    page_numbers: list[int]
    source_chunk_ids: list[int]


@dataclass
class GenerationReport:
    chapter_id: str
    chunks_processed: int
    pairs_generated: int
    pairs_discarded: int
    duration_seconds: float


_SYSTEM_PROMPT = (
    "You generate study-ready question/answer pairs from a textbook passage, "
    "for a Grade 2 student (around 7 years old). Rules:\n"
    "- Every answer must be directly supported by the passage. Do not add facts "
    "not in the passage, even if they are true in the world.\n"
    "- Use simple, age-appropriate language.\n"
    "- Prefer varied question styles: definitional (what is...), process (how does..., "
    "what happens when...), example (give an example of..., which of these are...), "
    "comparison (how is X different from Y), function (what is the role of X).\n"
    "- Keep each answer to 1-3 short sentences.\n"
    "- Do not repeat the same question twice.\n"
    "\n"
    "Return ONLY a JSON object matching this exact schema:\n"
    "{\n"
    '  "pairs": [\n'
    '    {"question": "<text>", "answer": "<text>"},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "No markdown, no commentary outside the JSON object."
)


def _build_user_prompt(passage: str, section: str | None, n: int) -> str:
    section_hint = f"Section: {section}\n" if section else ""
    return (
        f"{section_hint}"
        f"Passage:\n{passage}\n\n"
        f"Generate exactly {n} distinct question/answer pairs grounded in the passage. "
        "Produce the JSON object."
    )


@dataclass
class _ChunkIn:
    chunk_id: int
    page_start: int
    page_end: int
    section_title: str | None
    text: str


class QAGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        pairs_per_chunk: int = DEFAULT_PAIRS_PER_CHUNK,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise QAGenerationError("QAGenerator requires a non-empty API key")
        if pairs_per_chunk < 1:
            raise QAGenerationError("pairs_per_chunk must be >= 1")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.pairs_per_chunk = pairs_per_chunk
        self.timeout_seconds = timeout_seconds
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "QAGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def generate_for_chunks(
        self, chunks: Sequence[_ChunkIn]
    ) -> list[GeneratedPair]:
        """Generate pairs for each chunk and return the merged list."""
        all_pairs: list[GeneratedPair] = []
        for ch in chunks:
            if not ch.text.strip():
                continue
            try:
                pairs = self._generate_one_chunk(ch)
            except QAGenerationError as exc:
                logger.warning("QA generation failed on chunk %s: %s", ch.chunk_id, exc)
                continue
            all_pairs.extend(pairs)
        return all_pairs

    def _generate_one_chunk(self, ch: _ChunkIn) -> list[GeneratedPair]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        ch.text, ch.section_title, self.pairs_per_chunk
                    ),
                },
            ],
            "temperature": 0.5,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        data = self._post(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QAGenerationError(f"unexpected response shape: {exc}") from exc
        if content is None:
            raise QAGenerationError("LLM returned null content for Q&A generation")

        parsed = _parse_pairs_json(content)
        pages = list(range(ch.page_start, ch.page_end + 1))
        pairs: list[GeneratedPair] = []
        for item in parsed:
            q = item["question"].strip()
            a = item["answer"].strip()
            if not q or not a:
                continue
            pairs.append(
                GeneratedPair(
                    question=q,
                    answer=a,
                    page_numbers=pages,
                    source_chunk_ids=[ch.chunk_id],
                )
            )
        return pairs

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
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
            raise QAGenerationError(f"network error: {exc}") from exc
        if r.status_code != 200:
            raise QAGenerationError(f"HTTP {r.status_code}: {r.text[:400]}")
        return r.json()


def _parse_pairs_json(content: str) -> list[dict[str, str]]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QAGenerationError(
            f"Q&A JSON did not parse: {content[:200]}"
        ) from exc
    if not isinstance(parsed, dict) or "pairs" not in parsed:
        raise QAGenerationError(
            f"missing 'pairs' key in response: {list(parsed) if isinstance(parsed, dict) else type(parsed).__name__}"
        )
    pairs = parsed["pairs"]
    if not isinstance(pairs, list):
        raise QAGenerationError("'pairs' must be a JSON array")
    cleaned: list[dict[str, str]] = []
    for i, item in enumerate(pairs):
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        a = item.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        cleaned.append({"question": q, "answer": a})
    if not cleaned:
        raise QAGenerationError("no usable pairs in response")
    return cleaned


# ---- orchestrator ---------------------------------------------------------


def to_qa_rows(
    pairs: list[GeneratedPair], chapter_id: str
) -> list[QAPairRow]:
    """Convert generator output into DB rows (faiss_id filled in later)."""
    now = datetime.now(timezone.utc)
    rows: list[QAPairRow] = []
    seen_questions: set[str] = set()
    for p in pairs:
        key = p.question.lower().strip()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        rows.append(
            QAPairRow(
                qa_id=None,
                chapter_id=chapter_id,
                question=p.question,
                answer=p.answer,
                page_numbers=p.page_numbers,
                source_chunk_ids=p.source_chunk_ids,
                faiss_id=None,
                created_at=now,
            )
        )
    return rows
