"""End-to-end ingestion test using a fake embedder (duck-typed) to keep the
test fast and offline. The real EmbeddingService is exercised in
tests/test_embedding_service.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ingestion.ingest_chapter import ingest_chapter
from retrieval.vector_store import FaissVectorStore, index_path
from storage.metadata_repository import MetadataRepository
from storage.models import IngestionStatus


class FakeEmbeddingService:
    """Deterministic hash-based embeddings; same text → same vector."""

    model_name = "fake/hash-16"
    dim = 16

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            arr = np.frombuffer(digest[: self.dim * 2], dtype=np.int16).astype(np.float32)
            norm = np.linalg.norm(arr) or 1.0
            out[i] = arr / norm
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


def test_ingest_chapter_persists_chunks_and_index(text_pdf: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "tutor.db"
    index_dir = tmp_path / "indexes"
    repo = MetadataRepository(db_path)
    service = FakeEmbeddingService()

    report = ingest_chapter(
        pdf_path=text_pdf,
        chapter_id="ch3",
        title="Photosynthesis",
        subject="Biology",
        repo=repo,
        embedding_service=service,
        vector_store_dir=index_dir,
    )

    assert report.chapter_id == "ch3"
    assert report.page_count == 2
    assert report.embedded_pages == 2
    assert report.ocr_pages == 0
    assert report.chunk_count >= 1

    chapter = repo.get_chapter("ch3")
    assert chapter is not None
    assert chapter.ingestion_status == IngestionStatus.READY
    assert chapter.embedding_model == "fake/hash-16"
    assert chapter.page_count == 2

    assert repo.chunk_count("ch3") == report.chunk_count

    idx_path = index_path(index_dir, "ch3")
    assert idx_path.exists()
    store = FaissVectorStore.load(idx_path, expected_dim=service.dim)
    assert store.size == report.chunk_count

    # faiss_id round-trips to the stored chunk.
    chunks = list(repo.iter_chunks("ch3"))
    first = chunks[0]
    assert first.faiss_id == first.chunk_id

    repo.close()


def test_ingest_chapter_is_idempotent(text_pdf: Path, tmp_path: Path) -> None:
    repo = MetadataRepository(tmp_path / "tutor.db")
    index_dir = tmp_path / "indexes"
    service = FakeEmbeddingService()

    def run():
        return ingest_chapter(
            pdf_path=text_pdf,
            chapter_id="ch3",
            title="Photosynthesis",
            subject="Biology",
            repo=repo,
            embedding_service=service,
            vector_store_dir=index_dir,
        )

    r1 = run()
    r2 = run()

    # Chunk count stable; no duplicates left behind.
    assert repo.chunk_count("ch3") == r2.chunk_count == r1.chunk_count
    store = FaissVectorStore.load(index_path(index_dir, "ch3"), expected_dim=service.dim)
    assert store.size == r2.chunk_count

    repo.close()


def test_ingest_chapter_handles_image_only_without_tesseract(
    image_only_pdf: Path, tmp_path: Path, monkeypatch
) -> None:
    """When OCR is unavailable, the pipeline flags pages but doesn't crash."""
    monkeypatch.setattr(
        "ingestion.ingest_chapter.tesseract_available", lambda: False
    )
    repo = MetadataRepository(tmp_path / "tutor.db")
    service = FakeEmbeddingService()

    report = ingest_chapter(
        pdf_path=image_only_pdf,
        chapter_id="ch_img",
        title="Image-only",
        subject="Test",
        repo=repo,
        embedding_service=service,
        vector_store_dir=tmp_path / "indexes",
    )

    assert report.ocr_pages == 1
    assert report.ocr_failed_pages == [1]
    # No usable text → no chunks.
    assert report.chunk_count == 0

    chapter = repo.get_chapter("ch_img")
    assert chapter is not None
    assert chapter.ingestion_status == IngestionStatus.READY

    repo.close()
