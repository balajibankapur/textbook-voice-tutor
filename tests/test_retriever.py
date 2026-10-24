"""Fast retriever tests using hand-crafted vectors (no model download)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retrieval.retriever import Retriever
from retrieval.vector_store import FaissVectorStore, index_path
from storage.metadata_repository import MetadataRepository
from storage.models import (
    ChapterRow,
    ChunkRow,
    IngestionStatus,
    OcrQualityStatus,
)


class StaticEmbedder:
    """Looks up a fixed vector per query string. 4-dim so we can hand-craft hits."""

    model_name = "static/test"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = {q: _normalize(v) for q, v in mapping.items()}

    def embed(self, texts):
        return np.stack([self._mapping[t] for t in texts]).astype(np.float32)

    def embed_one(self, text):
        return self._mapping[text]


def _normalize(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    arr /= np.linalg.norm(arr) or 1.0
    return arr


def _make_chapter(repo: MetadataRepository, chapter_id: str = "ch1") -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    repo.upsert_chapter(
        ChapterRow(
            chapter_id=chapter_id,
            title="Test",
            subject="Test",
            source_file="/dev/null",
            page_count=3,
            ingestion_status=IngestionStatus.READY,
            embedding_model="static/test",
            created_at=now,
            updated_at=now,
        )
    )


def _seed_chunks(
    repo: MetadataRepository,
    index_dir: Path,
    chapter_id: str,
    docs: list[tuple[str, int, np.ndarray]],
) -> list[int]:
    """Insert chunks and populate the FAISS index with chunk_id == faiss_id."""
    rows = [
        ChunkRow(
            chunk_id=None,
            chapter_id=chapter_id,
            page_start=page,
            page_end=page,
            section_title=None,
            chunk_text=text,
            faiss_id=None,
            ocr_quality_status=OcrQualityStatus.NOT_APPLICABLE,
        )
        for text, page, _ in docs
    ]
    chunk_ids = repo.insert_chunks(rows)
    repo.set_faiss_ids([(cid, cid) for cid in chunk_ids])

    store = FaissVectorStore(dim=docs[0][2].shape[0])
    embeddings = np.stack([v for _, _, v in docs]).astype(np.float32)
    store.add(embeddings, ids=chunk_ids)
    store.save(index_path(index_dir, chapter_id))
    return chunk_ids


@pytest.fixture
def seeded(tmp_path: Path):
    repo = MetadataRepository(tmp_path / "tutor.db")
    _make_chapter(repo)
    docs = [
        ("photosynthesis uses light", 1, _normalize([1.0, 0.0, 0.0, 0.0])),
        ("newton laws of motion", 2, _normalize([0.0, 1.0, 0.0, 0.0])),
        ("chlorophyll absorbs light", 3, _normalize([0.9, 0.1, 0.0, 0.0])),
    ]
    chunk_ids = _seed_chunks(repo, tmp_path / "indexes", "ch1", docs)
    embedder = StaticEmbedder(
        {
            "how do plants make food?": _normalize([0.95, 0.05, 0.0, 0.0]),
            "what are the laws of motion?": _normalize([0.0, 1.0, 0.0, 0.0]),
            "": _normalize([0.0, 0.0, 0.0, 1.0]),
        }
    )
    retriever = Retriever(
        repo=repo,
        embedding_service=embedder,
        vector_store_dir=tmp_path / "indexes",
    )
    yield retriever, chunk_ids
    repo.close()


def test_search_returns_closest_first(seeded) -> None:
    retriever, chunk_ids = seeded
    hits = retriever.search("ch1", "how do plants make food?", top_k=3)
    assert len(hits) == 3
    assert hits[0].chunk_text == "photosynthesis uses light"
    assert hits[1].chunk_text == "chlorophyll absorbs light"
    assert hits[2].chunk_text == "newton laws of motion"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_populates_metadata(seeded) -> None:
    retriever, _ = seeded
    hit = retriever.search("ch1", "how do plants make food?", top_k=1)[0]
    assert hit.chapter_id == "ch1"
    assert hit.page_start == hit.page_end == 1
    assert hit.ocr_quality_status == OcrQualityStatus.NOT_APPLICABLE
    assert hit.chunk_id is not None


def test_empty_query_returns_empty(seeded) -> None:
    retriever, _ = seeded
    assert retriever.search("ch1", "", top_k=3) == []
    assert retriever.search("ch1", "   ", top_k=3) == []


def test_missing_chapter_index_returns_empty(seeded) -> None:
    retriever, _ = seeded
    assert retriever.search("does-not-exist", "how do plants make food?", top_k=3) == []


def test_top_k_bounds_result_count(seeded) -> None:
    retriever, _ = seeded
    assert len(retriever.search("ch1", "how do plants make food?", top_k=1)) == 1
    assert len(retriever.search("ch1", "how do plants make food?", top_k=2)) == 2


def test_cache_reuses_loaded_index(seeded, monkeypatch) -> None:
    retriever, _ = seeded
    retriever.search("ch1", "how do plants make food?", top_k=1)
    # After the first call the index is cached; FaissVectorStore.load should not
    # be called again.
    sentinel = {"loads": 0}
    original = FaissVectorStore.load

    def spy(*args, **kwargs):
        sentinel["loads"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(FaissVectorStore, "load", spy)
    retriever.search("ch1", "how do plants make food?", top_k=1)
    retriever.search("ch1", "what are the laws of motion?", top_k=1)
    assert sentinel["loads"] == 0


def test_invalidate_forces_reload(seeded, monkeypatch) -> None:
    retriever, _ = seeded
    retriever.search("ch1", "how do plants make food?", top_k=1)
    retriever.invalidate("ch1")

    sentinel = {"loads": 0}
    original = FaissVectorStore.load

    def spy(*args, **kwargs):
        sentinel["loads"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(FaissVectorStore, "load", spy)
    retriever.search("ch1", "how do plants make food?", top_k=1)
    assert sentinel["loads"] == 1
