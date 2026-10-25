"""Extract a chapter-level lesson plan: 3–6 teachable concepts from the chunks.

Each concept has a title, a learning objective, 3–5 grounded key points, and
the source chunk IDs it was drawn from. Pairs with `build_concepts.py` which
persists them to SQLite.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx

from storage.models import ChunkRow, ConceptRow

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MIN_CONCEPTS = 3
DEFAULT_MAX_CONCEPTS = 6
DEFAULT_MAX_TOKENS = 2000


class ConceptExtractionError(RuntimeError):
    """Raised when concept extraction fails or returns an unusable response."""


@dataclass
class ExtractedConcept:
    title: str
    learning_objective: str
    key_points: list[str]
    page_numbers: list[int]
    source_chunk_ids: list[int]


_SYSTEM_PROMPT = (
    "You distill a textbook chapter into a short lesson plan for a Grade 2 "
    "student (around 7 years old). You will be given the chapter's chunks "
    "(numbered). Produce 3–6 ORDERED teachable concepts that together cover "
    "the chapter.\n"
    "\n"
    "Rules:\n"
    "- Every key_point must be directly supported by the given chunks. Do not "
    "add facts that are not in the chunks, even if they are true in the world.\n"
    "- Order the concepts so earlier ones set up later ones.\n"
    "- Each concept must be worth 2–5 minutes of back-and-forth with a child — "
    "not a single sentence, not a whole chapter.\n"
    "- learning_objective MUST start with 'The student can…' and name a "
    "concrete, checkable behavior.\n"
    "- Each concept must list the source_chunk_ids it is drawn from (the ints "
    "shown as [id=N] in the input).\n"
    "- key_points: 3–5 short grounded bullet strings per concept.\n"
    "\n"
    "Return ONLY a JSON object matching this exact schema:\n"
    "{\n"
    '  "concepts": [\n'
    '    {\n'
    '      "title": "<short noun phrase>",\n'
    '      "learning_objective": "The student can ...",\n'
    '      "key_points": ["...", "...", "..."],\n'
    '      "source_chunk_ids": [<int>, ...]\n'
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}\n"
    "No markdown, no commentary outside the JSON object."
)


def _format_chunks(chunks: Sequence[ChunkRow]) -> str:
    out: list[str] = []
    for c in chunks:
        pages = (
            f"page {c.page_start}"
            if c.page_start == c.page_end
            else f"pages {c.page_start}-{c.page_end}"
        )
        section = f" · {c.section_title}" if c.section_title else ""
        out.append(f"[id={c.chunk_id}] ({pages}{section})\n{c.chunk_text}")
    return "\n\n".join(out)


class ConceptExtractor:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        min_concepts: int = DEFAULT_MIN_CONCEPTS,
        max_concepts: int = DEFAULT_MAX_CONCEPTS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ConceptExtractionError("ConceptExtractor requires a non-empty API key")
        if min_concepts < 1 or max_concepts < min_concepts:
            raise ConceptExtractionError("invalid min/max concept bounds")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.min_concepts = min_concepts
        self.max_concepts = max_concepts
        self.timeout_seconds = timeout_seconds
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ConceptExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def extract(self, chunks: Sequence[ChunkRow]) -> list[ExtractedConcept]:
        chunks = [c for c in chunks if c.chunk_text and c.chunk_text.strip()]
        if not chunks:
            raise ConceptExtractionError("no usable chunks to extract concepts from")

        user_prompt = (
            f"Chapter chunks:\n{_format_chunks(chunks)}\n\n"
            f"Produce {self.min_concepts}-{self.max_concepts} ordered teachable "
            "concepts as JSON."
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        data = self._post(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ConceptExtractionError(f"unexpected response shape: {exc}") from exc
        if content is None:
            raise ConceptExtractionError("LLM returned null content")

        parsed = _parse_concepts_json(content)
        page_lookup = {
            c.chunk_id: list(range(c.page_start, c.page_end + 1))
            for c in chunks
            if c.chunk_id is not None
        }
        valid_chunk_ids = set(page_lookup)

        extracted: list[ExtractedConcept] = []
        for item in parsed:
            source_ids = [cid for cid in item["source_chunk_ids"] if cid in valid_chunk_ids]
            if not source_ids:
                # Fall back to the whole chapter if the LLM cited unknown IDs.
                source_ids = sorted(valid_chunk_ids)
            pages = sorted({p for cid in source_ids for p in page_lookup[cid]})
            extracted.append(
                ExtractedConcept(
                    title=item["title"].strip(),
                    learning_objective=item["learning_objective"].strip(),
                    key_points=[kp.strip() for kp in item["key_points"] if kp.strip()],
                    page_numbers=pages,
                    source_chunk_ids=source_ids,
                )
            )
        if not extracted:
            raise ConceptExtractionError("no usable concepts in response")
        return extracted

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
            raise ConceptExtractionError(f"network error: {exc}") from exc
        if r.status_code != 200:
            raise ConceptExtractionError(f"HTTP {r.status_code}: {r.text[:400]}")
        return r.json()


def _parse_concepts_json(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConceptExtractionError(f"concepts JSON did not parse: {content[:200]}") from exc
    if not isinstance(parsed, dict) or "concepts" not in parsed:
        raise ConceptExtractionError("response missing 'concepts' array")
    concepts = parsed["concepts"]
    if not isinstance(concepts, list) or not concepts:
        raise ConceptExtractionError("'concepts' must be a non-empty JSON array")
    cleaned: list[dict[str, Any]] = []
    for item in concepts:
        if not isinstance(item, dict):
            continue
        try:
            title = str(item["title"])
            objective = str(item["learning_objective"])
            key_points = item["key_points"]
            source_ids = item["source_chunk_ids"]
        except KeyError:
            continue
        if not title.strip() or not objective.strip():
            continue
        if not isinstance(key_points, list) or not key_points:
            continue
        if not isinstance(source_ids, list):
            continue
        try:
            source_ids = [int(x) for x in source_ids]
        except (TypeError, ValueError):
            continue
        cleaned.append(
            {
                "title": title,
                "learning_objective": objective,
                "key_points": [str(k) for k in key_points if isinstance(k, (str, int, float))],
                "source_chunk_ids": source_ids,
            }
        )
    if not cleaned:
        raise ConceptExtractionError("no valid concepts in response")
    return cleaned


def to_concept_rows(
    extracted: list[ExtractedConcept], chapter_id: str
) -> list[ConceptRow]:
    now = datetime.now(timezone.utc)
    return [
        ConceptRow(
            concept_id=None,
            chapter_id=chapter_id,
            order_index=i,
            title=e.title,
            learning_objective=e.learning_objective,
            key_points=e.key_points,
            page_numbers=e.page_numbers,
            source_chunk_ids=e.source_chunk_ids,
            created_at=now,
        )
        for i, e in enumerate(extracted)
    ]
