"""End-to-end chapter ingestion: PDF → chunks → embeddings → FAISS + SQLite.

Idempotent per chapter_id: a second call replaces the chapter's chunks and
rebuilds its vector index rather than leaving duplicates behind.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ingestion.chunker import PageText, chunk_pages
from ingestion.ocr_processor import OcrQuality, ocr_pdf_page, tesseract_available
from ingestion.pdf_processor import ExtractionSource, extract_pages
from ingestion.text_normalizer import normalize
from retrieval.embedding_service import EmbeddingService
from retrieval.vector_store import FaissVectorStore, index_path
from storage.metadata_repository import MetadataRepository
from storage.models import (
    ChapterRow,
    ChunkRow,
    IngestionStatus,
    OcrQualityStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    chapter_id: str
    page_count: int
    embedded_pages: int
    ocr_pages: int
    ocr_failed_pages: list[int]
    chunk_count: int
    duration_seconds: float


_OCR_QUALITY_MAP = {
    OcrQuality.OK: OcrQualityStatus.OK,
    OcrQuality.LOW: OcrQualityStatus.LOW,
    OcrQuality.FAILED: OcrQualityStatus.FAILED,
}


def ingest_chapter(
    *,
    pdf_path: Path | str,
    chapter_id: str,
    title: str,
    subject: str,
    repo: MetadataRepository,
    embedding_service: EmbeddingService,
    vector_store_dir: Path | str,
    run_ocr: bool = True,
) -> IngestionReport:
    """Run the full ingestion workflow for one chapter."""
    pdf_path = Path(pdf_path)
    vector_store_dir = Path(vector_store_dir)
    start = time.perf_counter()

    chapter = ChapterRow(
        chapter_id=chapter_id,
        title=title,
        subject=subject,
        source_file=str(pdf_path),
        page_count=0,
        ingestion_status=IngestionStatus.EXTRACTING,
        embedding_model=embedding_service.model_name,
        created_at=_placeholder_dt(),
        updated_at=_placeholder_dt(),
    )
    repo.upsert_chapter(chapter)

    try:
        page_texts, ocr_quality_by_page, ocr_failed = _extract_all_pages(
            pdf_path=pdf_path, run_ocr=run_ocr
        )

        repo.set_chapter_status(chapter_id, IngestionStatus.EMBEDDING)

        chunks = chunk_pages(page_texts)
        logger.info("chapter %s: %d chunks from %d pages", chapter_id, len(chunks), len(page_texts))

        # Replace prior ingestion for this chapter.
        repo.delete_chunks_for_chapter(chapter_id)
        _delete_existing_index(vector_store_dir, chapter_id)

        if not chunks:
            repo.upsert_chapter(_updated_chapter(chapter, len(page_texts), IngestionStatus.READY))
            return _report(
                chapter_id=chapter_id,
                page_texts=page_texts,
                ocr_quality_by_page=ocr_quality_by_page,
                ocr_failed=ocr_failed,
                chunk_count=0,
                start=start,
            )

        chunk_rows = [
            ChunkRow(
                chunk_id=None,
                chapter_id=chapter_id,
                page_start=ch.page_start,
                page_end=ch.page_end,
                section_title=ch.section_title,
                chunk_text=ch.text,
                faiss_id=None,
                ocr_quality_status=_chunk_quality(ch.source_page_numbers, ocr_quality_by_page),
            )
            for ch in chunks
        ]
        chunk_ids = repo.insert_chunks(chunk_rows)

        embeddings = embedding_service.embed([ch.text for ch in chunks])

        store = FaissVectorStore(dim=embedding_service.dim)
        # faiss_id == chunk_id keeps the mapping trivial.
        store.add(embeddings, ids=chunk_ids)
        store.save(index_path(vector_store_dir, chapter_id))
        repo.set_faiss_ids([(cid, cid) for cid in chunk_ids])

        repo.upsert_chapter(_updated_chapter(chapter, len(page_texts), IngestionStatus.READY))

        return _report(
            chapter_id=chapter_id,
            page_texts=page_texts,
            ocr_quality_by_page=ocr_quality_by_page,
            ocr_failed=ocr_failed,
            chunk_count=len(chunk_rows),
            start=start,
        )

    except Exception:
        repo.set_chapter_status(chapter_id, IngestionStatus.FAILED)
        raise


# ---- helpers --------------------------------------------------------------


def _extract_all_pages(
    *, pdf_path: Path, run_ocr: bool
) -> tuple[list[PageText], dict[int, OcrQualityStatus], list[int]]:
    extractions = extract_pages(pdf_path)
    page_texts: list[PageText] = []
    ocr_quality: dict[int, OcrQualityStatus] = {}
    ocr_failed_pages: list[int] = []

    for ext in extractions:
        if ext.source == ExtractionSource.EMBEDDED:
            page_texts.append(PageText(page_number=ext.page_number, text=normalize(ext.text)))
            ocr_quality[ext.page_number] = OcrQualityStatus.NOT_APPLICABLE
            continue

        if not run_ocr:
            logger.warning("page %s needs OCR but run_ocr=False; skipping", ext.page_number)
            ocr_quality[ext.page_number] = OcrQualityStatus.FAILED
            ocr_failed_pages.append(ext.page_number)
            page_texts.append(PageText(page_number=ext.page_number, text=""))
            continue

        if not tesseract_available():
            logger.warning(
                "page %s needs OCR but tesseract is not installed; flagging failed",
                ext.page_number,
            )
            ocr_quality[ext.page_number] = OcrQualityStatus.FAILED
            ocr_failed_pages.append(ext.page_number)
            page_texts.append(PageText(page_number=ext.page_number, text=""))
            continue

        result = ocr_pdf_page(pdf_path, ext.page_number)
        quality = _OCR_QUALITY_MAP[result.quality]
        ocr_quality[ext.page_number] = quality
        if result.quality == OcrQuality.FAILED:
            ocr_failed_pages.append(ext.page_number)
        page_texts.append(
            PageText(page_number=ext.page_number, text=normalize(result.text))
        )

    return page_texts, ocr_quality, ocr_failed_pages


def _chunk_quality(
    pages: list[int], per_page: dict[int, OcrQualityStatus]
) -> OcrQualityStatus:
    """Worst status across a chunk's source pages wins."""
    severities = {
        OcrQualityStatus.FAILED: 3,
        OcrQualityStatus.LOW: 2,
        OcrQualityStatus.OK: 1,
        OcrQualityStatus.NOT_APPLICABLE: 0,
    }
    worst = OcrQualityStatus.NOT_APPLICABLE
    for p in pages:
        status = per_page.get(p, OcrQualityStatus.NOT_APPLICABLE)
        if severities[status] > severities[worst]:
            worst = status
    return worst


def _delete_existing_index(vector_store_dir: Path, chapter_id: str) -> None:
    path = index_path(vector_store_dir, chapter_id)
    if path.exists():
        path.unlink()


def _updated_chapter(
    chapter: ChapterRow, page_count: int, status: IngestionStatus
) -> ChapterRow:
    return ChapterRow(
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        subject=chapter.subject,
        source_file=chapter.source_file,
        page_count=page_count,
        ingestion_status=status,
        embedding_model=chapter.embedding_model,
        created_at=chapter.created_at,
        updated_at=chapter.updated_at,
    )


def _report(
    *,
    chapter_id: str,
    page_texts: list[PageText],
    ocr_quality_by_page: dict[int, OcrQualityStatus],
    ocr_failed: list[int],
    chunk_count: int,
    start: float,
) -> IngestionReport:
    embedded = sum(
        1 for q in ocr_quality_by_page.values() if q == OcrQualityStatus.NOT_APPLICABLE
    )
    ocr_pages = len(page_texts) - embedded
    return IngestionReport(
        chapter_id=chapter_id,
        page_count=len(page_texts),
        embedded_pages=embedded,
        ocr_pages=ocr_pages,
        ocr_failed_pages=ocr_failed,
        chunk_count=chunk_count,
        duration_seconds=time.perf_counter() - start,
    )


def _placeholder_dt():
    """A placeholder used before upsert_chapter overwrites created_at/updated_at."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
