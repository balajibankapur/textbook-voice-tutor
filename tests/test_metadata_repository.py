from datetime import datetime, timezone
from pathlib import Path

import pytest

from storage.metadata_repository import MetadataRepository
from storage.models import (
    ChapterRow,
    ChunkRow,
    IngestionStatus,
    OcrQualityStatus,
)


@pytest.fixture
def repo(tmp_path: Path) -> MetadataRepository:
    r = MetadataRepository(tmp_path / "tutor.db")
    yield r
    r.close()


def _sample_chapter(chapter_id: str = "ch3") -> ChapterRow:
    now = datetime.now(timezone.utc)
    return ChapterRow(
        chapter_id=chapter_id,
        title="Photosynthesis",
        subject="Biology",
        source_file="/tmp/ch3.pdf",
        page_count=5,
        ingestion_status=IngestionStatus.PENDING,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        created_at=now,
        updated_at=now,
    )


def _sample_chunk(chapter_id: str, text: str = "lorem ipsum") -> ChunkRow:
    return ChunkRow(
        chunk_id=None,
        chapter_id=chapter_id,
        page_start=1,
        page_end=1,
        section_title="Intro",
        chunk_text=text,
        faiss_id=None,
        ocr_quality_status=OcrQualityStatus.NOT_APPLICABLE,
    )


def test_upsert_and_get_chapter(repo: MetadataRepository) -> None:
    ch = _sample_chapter()
    repo.upsert_chapter(ch)
    got = repo.get_chapter("ch3")
    assert got is not None
    assert got.title == "Photosynthesis"
    assert got.subject == "Biology"


def test_upsert_chapter_updates_in_place(repo: MetadataRepository) -> None:
    ch = _sample_chapter()
    repo.upsert_chapter(ch)
    ch2 = _sample_chapter()
    ch2.title = "Photosynthesis — Revised"
    repo.upsert_chapter(ch2)
    assert repo.get_chapter("ch3").title == "Photosynthesis — Revised"
    assert len(repo.list_chapters()) == 1


def test_set_chapter_status(repo: MetadataRepository) -> None:
    repo.upsert_chapter(_sample_chapter())
    repo.set_chapter_status("ch3", IngestionStatus.READY)
    assert repo.get_chapter("ch3").ingestion_status == IngestionStatus.READY


def test_insert_chunks_returns_ids(repo: MetadataRepository) -> None:
    repo.upsert_chapter(_sample_chapter())
    ids = repo.insert_chunks(
        [_sample_chunk("ch3", "a"), _sample_chunk("ch3", "b")]
    )
    assert len(ids) == 2
    assert ids[0] != ids[1]
    assert repo.chunk_count("ch3") == 2


def test_set_and_lookup_faiss_ids_preserves_order(repo: MetadataRepository) -> None:
    repo.upsert_chapter(_sample_chapter())
    chunk_ids = repo.insert_chunks(
        [_sample_chunk("ch3", f"text-{i}") for i in range(3)]
    )
    # Use chunk_id == faiss_id (the orchestrator's convention).
    repo.set_faiss_ids([(cid, cid) for cid in chunk_ids])
    requested = [chunk_ids[2], chunk_ids[0]]
    chunks = repo.get_chunks_by_faiss_ids("ch3", requested)
    assert [c.faiss_id for c in chunks] == requested
    assert chunks[0].chunk_text == "text-2"
    assert chunks[1].chunk_text == "text-0"


def test_delete_chapter_cascades_to_chunks(repo: MetadataRepository) -> None:
    repo.upsert_chapter(_sample_chapter())
    repo.insert_chunks([_sample_chunk("ch3")])
    assert repo.chunk_count("ch3") == 1
    repo.delete_chapter("ch3")
    assert repo.get_chapter("ch3") is None
    assert repo.chunk_count("ch3") == 0


def test_delete_chunks_for_chapter_only(repo: MetadataRepository) -> None:
    repo.upsert_chapter(_sample_chapter())
    repo.insert_chunks([_sample_chunk("ch3"), _sample_chunk("ch3")])
    repo.delete_chunks_for_chapter("ch3")
    assert repo.chunk_count("ch3") == 0
    # Chapter row remains.
    assert repo.get_chapter("ch3") is not None
