"""Orchestrate Q&A corpus generation + FAISS index build for a chapter.

End-to-end:
  1. Read the chapter's chunks from SQLite
  2. Ask the LLM to produce question/answer pairs per chunk
  3. Deduplicate questions
  4. Embed each question and build a per-chapter Q&A FAISS index
  5. Store pairs + faiss_id mapping in SQLite
  6. Return a report with counts and duration

Idempotent per chapter: a rebuild wipes the chapter's old pairs and index first.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ingestion.qa_generator import (
    GenerationReport,
    QAGenerationError,
    QAGenerator,
    _ChunkIn,
    to_qa_rows,
)
from retrieval.embedding_service import EmbeddingService
from retrieval.qa_matcher import qa_index_path
from retrieval.vector_store import FaissVectorStore
from storage.metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)


@dataclass
class QACacheReport:
    chapter_id: str
    chunks_processed: int
    pairs_generated: int
    pairs_persisted: int
    duration_seconds: float


def build_qa_cache(
    *,
    chapter_id: str,
    repo: MetadataRepository,
    embedding_service: EmbeddingService,
    qa_generator: QAGenerator,
    qa_index_dir: Path | str,
) -> QACacheReport:
    start = time.perf_counter()
    qa_index_dir = Path(qa_index_dir)

    chunks = list(repo.iter_chunks(chapter_id))
    if not chunks:
        logger.warning("no chunks found for chapter %s; nothing to cache", chapter_id)
        return QACacheReport(
            chapter_id=chapter_id,
            chunks_processed=0,
            pairs_generated=0,
            pairs_persisted=0,
            duration_seconds=time.perf_counter() - start,
        )

    chunk_inputs = [
        _ChunkIn(
            chunk_id=c.chunk_id or 0,
            page_start=c.page_start,
            page_end=c.page_end,
            section_title=c.section_title,
            text=c.chunk_text,
        )
        for c in chunks
        if c.chunk_text and c.chunk_text.strip()
    ]
    pairs = qa_generator.generate_for_chunks(chunk_inputs)
    rows = to_qa_rows(pairs, chapter_id=chapter_id)
    if not rows:
        logger.warning("chapter %s: generation yielded no pairs", chapter_id)
        return QACacheReport(
            chapter_id=chapter_id,
            chunks_processed=len(chunk_inputs),
            pairs_generated=len(pairs),
            pairs_persisted=0,
            duration_seconds=time.perf_counter() - start,
        )

    # Replace prior corpus for this chapter.
    repo.delete_qa_pairs_for_chapter(chapter_id)
    _delete_existing_qa_index(qa_index_dir, chapter_id)

    qa_ids = repo.insert_qa_pairs(rows)

    questions = [r.question for r in rows]
    embeddings = embedding_service.embed(questions)

    store = FaissVectorStore(dim=embedding_service.dim)
    store.add(embeddings, ids=qa_ids)
    store.save(qa_index_path(qa_index_dir, chapter_id))
    repo.set_qa_faiss_ids([(qid, qid) for qid in qa_ids])

    return QACacheReport(
        chapter_id=chapter_id,
        chunks_processed=len(chunk_inputs),
        pairs_generated=len(pairs),
        pairs_persisted=len(qa_ids),
        duration_seconds=time.perf_counter() - start,
    )


def _delete_existing_qa_index(qa_index_dir: Path, chapter_id: str) -> None:
    path = qa_index_path(qa_index_dir, chapter_id)
    if path.exists():
        path.unlink()
