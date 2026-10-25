"""Orchestrate concept extraction for a chapter.

Reads the chapter's chunks, calls the extractor, persists concepts. Idempotent:
rebuilding wipes the chapter's existing concept rows first.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ingestion.concept_extractor import ConceptExtractor, to_concept_rows
from storage.metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)


@dataclass
class ConceptBuildReport:
    chapter_id: str
    chunks_processed: int
    concepts_persisted: int
    duration_seconds: float


def build_concepts(
    *,
    chapter_id: str,
    repo: MetadataRepository,
    extractor: ConceptExtractor,
) -> ConceptBuildReport:
    start = time.perf_counter()
    chunks = list(repo.iter_chunks(chapter_id))
    if not chunks:
        return ConceptBuildReport(
            chapter_id=chapter_id,
            chunks_processed=0,
            concepts_persisted=0,
            duration_seconds=time.perf_counter() - start,
        )

    extracted = extractor.extract(chunks)
    rows = to_concept_rows(extracted, chapter_id=chapter_id)

    repo.delete_concepts_for_chapter(chapter_id)
    repo.insert_concepts(rows)

    return ConceptBuildReport(
        chapter_id=chapter_id,
        chunks_processed=len(chunks),
        concepts_persisted=len(rows),
        duration_seconds=time.perf_counter() - start,
    )
