"""FAISS vector store (one index per chapter), persisted to disk.

Uses IndexFlatIP wrapped in IndexIDMap2 so vector IDs are stable and map
directly to SQLite chunk_ids. Embeddings must already be L2-normalized —
inner product on normalized vectors is cosine similarity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    faiss_id: int
    score: float


class FaissVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @classmethod
    def load(cls, path: Path | str, expected_dim: int | None = None) -> "FaissVectorStore":
        path = Path(path)
        index = faiss.read_index(str(path))
        store = cls.__new__(cls)
        store._index = index
        store.dim = index.d
        if expected_dim is not None and expected_dim != store.dim:
            raise ValueError(
                f"index at {path} has dim={store.dim}, expected {expected_dim}"
            )
        return store

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"expected (n, {self.dim}) embeddings, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(ids):
            raise ValueError("embeddings and ids length mismatch")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)
        id_arr = np.asarray(ids, dtype=np.int64)
        self._index.add_with_ids(embeddings, id_arr)

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        selector = faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
        return int(self._index.remove_ids(selector))

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(
                f"query dim {query.shape[1]} != index dim {self.dim}"
            )
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)
        if self.size == 0:
            return []
        scores, ids = self._index.search(query, min(top_k, self.size))
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            hits.append(SearchHit(faiss_id=int(idx), score=float(score)))
        return hits


def index_path(root: Path | str, chapter_id: str) -> Path:
    return Path(root) / f"{chapter_id}.faiss"
