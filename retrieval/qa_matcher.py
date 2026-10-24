"""Cache-first retrieval: match student questions to pre-generated Q&A pairs.

Flow:
  1. Embed the student's question
  2. FAISS top-K search over this chapter's Q&A index
  3. Ask Jev (via the native `choice` decision) which candidate semantically
     matches the student's question — Jev returns a calibrated probability
     per candidate
  4. If the chosen candidate has probability ≥ threshold, return its answer

Falls through to `None` on empty index, 'none' chosen, low probability, or
network errors — the caller then runs the live retrieval + LLM path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision.jev_decisions_client import JevDecisionsClient
from retrieval.embedding_service import EmbeddingService
from retrieval.vector_store import FaissVectorStore
from storage.metadata_repository import MetadataRepository
from storage.models import QAPairRow

logger = logging.getLogger(__name__)


class QAMatcherError(RuntimeError):
    """Raised when the matcher's HTTP call fails or returns an unusable response."""


@dataclass
class QAMatch:
    qa_pair: QAPairRow
    confidence: float  # probability the chosen candidate matches the student query
    reason: str
    candidates_considered: int
    raw: dict[str, Any]


def qa_index_path(root: Path | str, chapter_id: str) -> Path:
    return Path(root) / f"{chapter_id}.qa.faiss"


_MATCH_INSTRUCTIONS = (
    "Pick the candidate that is semantically equivalent to the student's "
    "question — same information requested, even if phrased differently. "
    "Pick 'none' if no candidate asks the same thing."
)


class QAMatcher:
    def __init__(
        self,
        *,
        repo: MetadataRepository,
        embedding_service: EmbeddingService,
        qa_index_dir: Path | str,
        decisions_client: JevDecisionsClient,
        top_k: int = 5,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.repo = repo
        self.embedding_service = embedding_service
        self.qa_index_dir = Path(qa_index_dir)
        self.decisions_client = decisions_client
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._cache: dict[str, FaissVectorStore] = {}

    def invalidate(self, chapter_id: str | None = None) -> None:
        if chapter_id is None:
            self._cache.clear()
            return
        self._cache.pop(chapter_id, None)

    def find_cached_answer(
        self, chapter_id: str, question: str
    ) -> QAMatch | None:
        """Return a QAMatch when the student's question maps to a stored pair."""
        if not question.strip():
            return None
        store = self._load_index(chapter_id)
        if store is None or store.size == 0:
            return None

        query_vec = self.embedding_service.embed_one(question)
        hits = store.search(query_vec, top_k=self.top_k)
        if not hits:
            return None

        pairs = self.repo.get_qa_pairs_by_faiss_ids(
            chapter_id, [h.faiss_id for h in hits]
        )
        if not pairs:
            return None

        candidates = {f"c{i}": p.question for i, p in enumerate(pairs)}
        try:
            match = self.decisions_client.match_choice(
                state_text=question,
                candidates=candidates,
                instructions=_MATCH_INSTRUCTIONS,
                include_none_option=True,
            )
        except Exception as exc:
            logger.warning("Jev matcher error, falling through: %s", exc)
            return None

        if match.chosen_key == "none":
            return None
        if match.chosen_key not in candidates:
            logger.warning("Jev chose unknown key %r — falling through", match.chosen_key)
            return None
        if match.probability < self.confidence_threshold:
            return None

        idx = int(match.chosen_key.lstrip("c"))
        matched = pairs[idx]
        return QAMatch(
            qa_pair=matched,
            confidence=match.probability,
            reason=f"jev-choice probability {match.probability:.2f}",
            candidates_considered=len(pairs),
            raw=match.raw,
        )

    def _load_index(self, chapter_id: str) -> FaissVectorStore | None:
        if chapter_id in self._cache:
            return self._cache[chapter_id]
        path = qa_index_path(self.qa_index_dir, chapter_id)
        if not path.exists():
            return None
        store = FaissVectorStore.load(path, expected_dim=self.embedding_service.dim)
        self._cache[chapter_id] = store
        return store
