"""Query-side retrieval: embed a question, search a chapter's FAISS index,
hydrate hits with chunk metadata from SQLite.

Keeps a per-chapter cache of loaded FAISS indexes so each question doesn't
reload from disk. Call `invalidate(chapter_id)` after re-ingesting a chapter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from retrieval.embedding_service import EmbeddingService
from retrieval.vector_store import FaissVectorStore, index_path
from storage.metadata_repository import MetadataRepository
from storage.models import OcrQualityStatus

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    chapter_id: str
    page_start: int
    page_end: int
    section_title: str | None
    chunk_text: str
    score: float
    ocr_quality_status: OcrQualityStatus


class Retriever:
    def __init__(
        self,
        *,
        repo: MetadataRepository,
        embedding_service: EmbeddingService,
        vector_store_dir: Path | str,
    ) -> None:
        self.repo = repo
        self.embedding_service = embedding_service
        self.vector_store_dir = Path(vector_store_dir)
        self._cache: dict[str, FaissVectorStore] = {}

    def search(
        self,
        chapter_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        store = self._load_index(chapter_id)
        if store is None or store.size == 0:
            return []

        query_vec = self.embedding_service.embed_one(query)
        hits = store.search(query_vec, top_k=top_k)
        if not hits:
            return []

        faiss_ids = [h.faiss_id for h in hits]
        score_by_id = {h.faiss_id: h.score for h in hits}
        rows = self.repo.get_chunks_by_faiss_ids(chapter_id, faiss_ids)

        results: list[RetrievedChunk] = []
        for row in rows:
            assert row.faiss_id is not None and row.chunk_id is not None
            results.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    chapter_id=row.chapter_id,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    section_title=row.section_title,
                    chunk_text=row.chunk_text,
                    score=score_by_id[row.faiss_id],
                    ocr_quality_status=row.ocr_quality_status,
                )
            )
        return results

    def invalidate(self, chapter_id: str | None = None) -> None:
        """Drop cached index(es) so the next query reloads from disk."""
        if chapter_id is None:
            self._cache.clear()
            return
        self._cache.pop(chapter_id, None)

    def _load_index(self, chapter_id: str) -> FaissVectorStore | None:
        if chapter_id in self._cache:
            return self._cache[chapter_id]
        path = index_path(self.vector_store_dir, chapter_id)
        if not path.exists():
            logger.warning("no FAISS index found for chapter %s at %s", chapter_id, path)
            return None
        store = FaissVectorStore.load(path, expected_dim=self.embedding_service.dim)
        self._cache[chapter_id] = store
        return store
