"""Live tests against sentence-transformers.

Marked `live_model` and excluded from the default pytest run because torch's
BLAS/OpenMP state conflicts with faiss in the same process. Run explicitly:

    pytest -m live_model

Also skipped when sentence-transformers isn't installed or the model can't
be downloaded (e.g. no internet on first run).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.live_model


@pytest.fixture(scope="module")
def service():
    # Import inside the fixture: at module/collection time sentence-transformers
    # would import torch, whose BLAS/OpenMP state conflicts with faiss in the
    # same process even when these tests are deselected.
    pytest.importorskip("sentence_transformers")
    from retrieval.embedding_service import EmbeddingService

    try:
        return EmbeddingService()
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not load embedding model: {exc}")


def test_embedding_dim_matches_default(service) -> None:
    from retrieval.embedding_service import DEFAULT_DIM

    assert service.dim == DEFAULT_DIM


def test_embeddings_are_normalized(service) -> None:
    vectors = service.embed(["photosynthesis is a process", "the sun is bright"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert vectors.shape == (2, service.dim)
    assert vectors.dtype == np.float32


def test_embed_one_returns_1d(service) -> None:
    v = service.embed_one("hello world")
    assert v.shape == (service.dim,)


def test_semantic_similarity_sanity(service) -> None:
    plants, physics, plants2 = service.embed(
        [
            "Plants use sunlight to make food in a process called photosynthesis.",
            "Newton formulated the laws of motion.",
            "Chlorophyll absorbs light for photosynthesis.",
        ]
    )
    assert float(plants @ plants2) > float(plants @ physics)


def test_embed_empty_returns_empty(service) -> None:
    out = service.embed([])
    assert out.shape == (0, service.dim)
