"""Fast tests for QAMatcher — uses a fake JevDecisionsClient + hand-crafted FAISS."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from decision.jev_decisions_client import ChoiceMatch
from retrieval.qa_matcher import QAMatch, QAMatcher, qa_index_path
from retrieval.vector_store import FaissVectorStore
from storage.metadata_repository import MetadataRepository
from storage.models import ChapterRow, IngestionStatus, QAPairRow


class StaticEmbedder:
    """Lookup embedder: text → fixed vector."""

    model_name = "static/test"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = {k: _normalize(v) for k, v in mapping.items()}

    def embed(self, texts):
        return np.stack([self._mapping[t] for t in texts]).astype(np.float32)

    def embed_one(self, text):
        return self._mapping[text]


def _normalize(vec):
    arr = np.asarray(vec, dtype=np.float32)
    arr /= np.linalg.norm(arr) or 1.0
    return arr


@dataclass
class FakeDecisionsClient:
    """Returns pre-queued ChoiceMatch results (or raises on error)."""
    outputs: list = field(default_factory=list)
    seen: list[dict] = field(default_factory=list)

    def match_choice(self, *, state_text, candidates, instructions, include_none_option=True):
        self.seen.append(
            {
                "state_text": state_text,
                "candidate_keys": list(candidates.keys()),
                "include_none": include_none_option,
            }
        )
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _choice(key: str, prob: float, probs: dict | None = None) -> ChoiceMatch:
    probs = probs or {k: 1.0 if k == key else 0.0 for k in [key, "none"]}
    return ChoiceMatch(
        chosen_key=key,
        probability=probs[key],
        confidence=prob,
        probabilities=probs,
        raw={"choice": key, "probabilities": probs},
    )


@pytest.fixture
def seeded(tmp_path: Path):
    repo = MetadataRepository(tmp_path / "tutor.db")
    now = datetime.now(timezone.utc)
    repo.upsert_chapter(
        ChapterRow(
            chapter_id="ch1", title="T", subject="S", source_file="x",
            page_count=1, ingestion_status=IngestionStatus.READY,
            embedding_model="static/test", created_at=now, updated_at=now,
        )
    )
    pairs = [
        QAPairRow(qa_id=None, chapter_id="ch1",
                  question="What are the parts of a plant?",
                  answer="Roots, stems, leaves.",
                  page_numbers=[1, 2], source_chunk_ids=[1],
                  faiss_id=None, created_at=now),
        QAPairRow(qa_id=None, chapter_id="ch1",
                  question="What is photosynthesis?",
                  answer="How plants make food from sunlight.",
                  page_numbers=[3], source_chunk_ids=[2],
                  faiss_id=None, created_at=now),
    ]
    ids = repo.insert_qa_pairs(pairs)
    repo.set_qa_faiss_ids([(qid, qid) for qid in ids])

    store = FaissVectorStore(dim=4)
    store.add(
        np.stack(
            [
                _normalize([1, 0, 0, 0]),
                _normalize([0, 1, 0, 0]),
            ]
        ).astype(np.float32),
        ids=ids,
    )
    index_dir = tmp_path / "indexes"
    store.save(qa_index_path(index_dir, "ch1"))

    embedder = StaticEmbedder(
        {
            "what are the parts of a plant?": _normalize([0.95, 0.05, 0, 0]),
            "unrelated": _normalize([0, 0, 0, 1]),
        }
    )
    yield repo, embedder, index_dir, ids
    repo.close()


def test_cache_hit_returns_stored_answer(seeded) -> None:
    repo, embedder, index_dir, _ = seeded
    client = FakeDecisionsClient(outputs=[
        _choice("c0", 0.95, probs={"c0": 0.9, "c1": 0.05, "none": 0.05}),
    ])
    m = QAMatcher(
        repo=repo, embedding_service=embedder, qa_index_dir=index_dir,
        decisions_client=client,
    )
    match = m.find_cached_answer("ch1", "what are the parts of a plant?")
    assert isinstance(match, QAMatch)
    assert match.qa_pair.answer == "Roots, stems, leaves."
    assert match.confidence == pytest.approx(0.9)


def test_low_probability_falls_through(seeded) -> None:
    repo, embedder, index_dir, _ = seeded
    client = FakeDecisionsClient(outputs=[
        _choice("c0", 0.5, probs={"c0": 0.4, "c1": 0.3, "none": 0.3}),
    ])
    m = QAMatcher(
        repo=repo, embedding_service=embedder, qa_index_dir=index_dir,
        decisions_client=client, confidence_threshold=0.6,
    )
    assert m.find_cached_answer("ch1", "what are the parts of a plant?") is None


def test_none_choice_falls_through(seeded) -> None:
    repo, embedder, index_dir, _ = seeded
    client = FakeDecisionsClient(outputs=[
        _choice("none", 0.95, probs={"c0": 0.1, "c1": 0.0, "none": 0.9}),
    ])
    m = QAMatcher(
        repo=repo, embedding_service=embedder, qa_index_dir=index_dir,
        decisions_client=client,
    )
    assert m.find_cached_answer("ch1", "what are the parts of a plant?") is None


def test_missing_index_returns_none(tmp_path: Path, seeded) -> None:
    repo, embedder, _, _ = seeded
    client = FakeDecisionsClient(outputs=[])  # Should never be called.
    m = QAMatcher(
        repo=repo, embedding_service=embedder, qa_index_dir=tmp_path / "no_such_dir",
        decisions_client=client,
    )
    assert m.find_cached_answer("ch1", "any question") is None
    assert client.seen == []


def test_jev_error_falls_through(seeded) -> None:
    repo, embedder, index_dir, _ = seeded
    client = FakeDecisionsClient(outputs=[RuntimeError("boom")])
    m = QAMatcher(
        repo=repo, embedding_service=embedder, qa_index_dir=index_dir,
        decisions_client=client,
    )
    # Must not raise; returns None so caller falls through.
    assert m.find_cached_answer("ch1", "what are the parts of a plant?") is None


def test_empty_question_returns_none(seeded) -> None:
    repo, embedder, index_dir, _ = seeded
    client = FakeDecisionsClient(outputs=[])
    m = QAMatcher(
        repo=repo, embedding_service=embedder, qa_index_dir=index_dir,
        decisions_client=client,
    )
    assert m.find_cached_answer("ch1", "   ") is None
    assert client.seen == []


def test_candidates_passed_as_choice_criteria(seeded) -> None:
    """Verify the matcher sends candidate questions as the 'criteria' map."""
    repo, embedder, index_dir, _ = seeded
    client = FakeDecisionsClient(outputs=[
        _choice("c0", 0.9, probs={"c0": 0.9, "c1": 0.05, "none": 0.05}),
    ])
    m = QAMatcher(
        repo=repo, embedding_service=embedder, qa_index_dir=index_dir,
        decisions_client=client,
    )
    m.find_cached_answer("ch1", "what are the parts of a plant?")
    seen = client.seen[0]
    assert seen["state_text"] == "what are the parts of a plant?"
    assert seen["include_none"] is True
    assert "c0" in seen["candidate_keys"] and "c1" in seen["candidate_keys"]
