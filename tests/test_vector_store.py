from pathlib import Path

import numpy as np
import pytest

from retrieval.vector_store import FaissVectorStore, index_path


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype(np.float32)


def test_add_and_search_returns_closest() -> None:
    store = FaissVectorStore(dim=4)
    vectors = _normalize(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    store.add(vectors, ids=[10, 20, 30])
    assert store.size == 3

    query = _normalize(np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32))
    hits = store.search(query, top_k=2)
    assert hits[0].faiss_id == 10
    assert hits[0].score > hits[1].score


def test_search_empty_index_returns_empty() -> None:
    store = FaissVectorStore(dim=4)
    hits = store.search(np.ones((1, 4), dtype=np.float32), top_k=5)
    assert hits == []


def test_remove_ids() -> None:
    store = FaissVectorStore(dim=2)
    vectors = _normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    store.add(vectors, ids=[1, 2])
    removed = store.remove([1])
    assert removed == 1
    assert store.size == 1
    hits = store.search(_normalize(np.array([[1.0, 0.0]], dtype=np.float32)), top_k=5)
    assert all(h.faiss_id != 1 for h in hits)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = FaissVectorStore(dim=3)
    vectors = _normalize(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    )
    store.add(vectors, ids=[7, 8, 9])
    path = index_path(tmp_path, "ch3")
    store.save(path)
    assert path.exists()

    reloaded = FaissVectorStore.load(path, expected_dim=3)
    assert reloaded.size == 3
    hits = reloaded.search(_normalize(np.array([[0, 1, 0]], dtype=np.float32)), top_k=1)
    assert hits[0].faiss_id == 8


def test_load_dim_mismatch_raises(tmp_path: Path) -> None:
    store = FaissVectorStore(dim=3)
    store.add(_normalize(np.array([[1, 0, 0]], dtype=np.float32)), ids=[1])
    path = index_path(tmp_path, "ch3")
    store.save(path)
    with pytest.raises(ValueError):
        FaissVectorStore.load(path, expected_dim=4)


def test_add_rejects_wrong_shape() -> None:
    store = FaissVectorStore(dim=4)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 3), dtype=np.float32), ids=[1, 2])


def test_add_rejects_id_count_mismatch() -> None:
    store = FaissVectorStore(dim=4)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 4), dtype=np.float32), ids=[1])
