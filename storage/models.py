"""Row dataclasses mirroring the SQLite schema.

Keeping these as plain dataclasses (not SQLAlchemy) matches the spec's "no
unnecessary infrastructure" rule and keeps the repository layer thin.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class IngestionStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class OcrQualityStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"  # text came from embedded PDF layer
    OK = "ok"
    LOW = "low"
    FAILED = "failed"


@dataclass
class ChapterRow:
    chapter_id: str
    title: str
    subject: str
    source_file: str
    page_count: int
    ingestion_status: IngestionStatus
    embedding_model: str
    created_at: datetime
    updated_at: datetime


@dataclass
class ChunkRow:
    chunk_id: int | None  # None before insertion
    chapter_id: str
    page_start: int
    page_end: int
    section_title: str | None
    chunk_text: str
    faiss_id: int | None  # assigned when added to FAISS
    ocr_quality_status: OcrQualityStatus


@dataclass
class QAPairRow:
    """A pre-generated Q&A pair that can be served without a live LLM call."""
    qa_id: int | None  # None before insertion
    chapter_id: str
    question: str
    answer: str
    page_numbers: list[int]
    source_chunk_ids: list[int]
    faiss_id: int | None  # assigned when added to FAISS
    created_at: datetime
