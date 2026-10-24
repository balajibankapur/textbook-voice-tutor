"""Sentence-Transformer embeddings for chunks and queries.

The same instance (and model name) must be used for ingestion and retrieval —
mixing embedding models in the same FAISS index silently breaks relevance.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384  # all-MiniLM-L6-v2


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)
        # cache the dimension; cheaper than probing the model each call.
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = int(self._model.get_embedding_dimension())
        else:  # older sentence-transformers
            self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized float32 embeddings with shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32, copy=False)

    def embed_one(self, text: str) -> np.ndarray:
        """Return a single (dim,) float32 embedding."""
        return self.embed([text])[0]


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Lazy import so tests that don't need the model avoid heavy imports."""
    from sentence_transformers import SentenceTransformer

    logger.info("loading embedding model %s", model_name)
    return SentenceTransformer(model_name)
