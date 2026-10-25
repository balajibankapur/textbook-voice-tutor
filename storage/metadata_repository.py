"""SQLite-backed repository for chapters and chunks.

The schema and the FAISS index must stay in sync — this module owns the
`faiss_id` column, so vector_store assigns IDs and we write them here in the
same transaction as the chunk insert.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from storage.models import (
    ChapterRow,
    ChunkRow,
    ConceptRow,
    IngestionStatus,
    OcrQualityStatus,
    QAPairRow,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id       TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    subject          TEXT NOT NULL,
    source_file      TEXT NOT NULL,
    page_count       INTEGER NOT NULL,
    ingestion_status TEXT NOT NULL,
    embedding_model  TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id          TEXT NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    page_start          INTEGER NOT NULL,
    page_end            INTEGER NOT NULL,
    section_title       TEXT,
    chunk_text          TEXT NOT NULL,
    faiss_id            INTEGER,
    ocr_quality_status  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(chapter_id, faiss_id);

CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id        TEXT NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    question          TEXT NOT NULL,
    answer            TEXT NOT NULL,
    page_numbers_json TEXT NOT NULL,  -- JSON-encoded list[int]
    source_chunk_ids_json TEXT NOT NULL,  -- JSON-encoded list[int]
    faiss_id          INTEGER,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qa_chapter ON qa_pairs(chapter_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_qa_faiss ON qa_pairs(chapter_id, faiss_id);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id            TEXT NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    order_index           INTEGER NOT NULL,
    title                 TEXT NOT NULL,
    learning_objective    TEXT NOT NULL,
    key_points_json       TEXT NOT NULL,  -- JSON-encoded list[str]
    page_numbers_json     TEXT NOT NULL,  -- JSON-encoded list[int]
    source_chunk_ids_json TEXT NOT NULL,  -- JSON-encoded list[int]
    created_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_concepts_order ON concepts(chapter_id, order_index);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class MetadataRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit's @st.cache_resource stores this
        # on one thread but calls it from another when the script reruns.
        # Safe here because callers don't share this connection across concurrent
        # writes (Streamlit serializes per-session script runs).
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MetadataRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- chapters -------------------------------------------------------

    def upsert_chapter(self, chapter: ChapterRow) -> None:
        now = _now_iso()
        existing = self.get_chapter(chapter.chapter_id)
        created_at = existing.created_at.isoformat() if existing else now
        self._conn.execute(
            """
            INSERT INTO chapters(chapter_id, title, subject, source_file, page_count,
                                 ingestion_status, embedding_model, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                title=excluded.title,
                subject=excluded.subject,
                source_file=excluded.source_file,
                page_count=excluded.page_count,
                ingestion_status=excluded.ingestion_status,
                embedding_model=excluded.embedding_model,
                updated_at=excluded.updated_at
            """,
            (
                chapter.chapter_id,
                chapter.title,
                chapter.subject,
                chapter.source_file,
                chapter.page_count,
                chapter.ingestion_status.value,
                chapter.embedding_model,
                created_at,
                now,
            ),
        )
        self._conn.commit()

    def set_chapter_status(self, chapter_id: str, status: IngestionStatus) -> None:
        self._conn.execute(
            "UPDATE chapters SET ingestion_status=?, updated_at=? WHERE chapter_id=?",
            (status.value, _now_iso(), chapter_id),
        )
        self._conn.commit()

    def get_chapter(self, chapter_id: str) -> ChapterRow | None:
        row = self._conn.execute(
            "SELECT * FROM chapters WHERE chapter_id=?", (chapter_id,)
        ).fetchone()
        return self._row_to_chapter(row) if row else None

    def list_chapters(self) -> list[ChapterRow]:
        rows = self._conn.execute(
            "SELECT * FROM chapters ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_chapter(r) for r in rows]

    def delete_chapter(self, chapter_id: str) -> None:
        # Foreign-key cascade removes chunks.
        self._conn.execute("DELETE FROM chapters WHERE chapter_id=?", (chapter_id,))
        self._conn.commit()

    # ---- chunks ---------------------------------------------------------

    def delete_chunks_for_chapter(self, chapter_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE chapter_id=?", (chapter_id,))
        self._conn.commit()

    def insert_chunks(self, chunks: list[ChunkRow]) -> list[int]:
        """Insert chunks, returning the generated chunk_ids in input order."""
        ids: list[int] = []
        with self._conn:
            for chunk in chunks:
                cur = self._conn.execute(
                    """
                    INSERT INTO chunks(chapter_id, page_start, page_end, section_title,
                                       chunk_text, faiss_id, ocr_quality_status)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chapter_id,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section_title,
                        chunk.chunk_text,
                        chunk.faiss_id,
                        chunk.ocr_quality_status.value,
                    ),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def set_faiss_ids(self, pairs: list[tuple[int, int]]) -> None:
        """pairs: [(chunk_id, faiss_id), ...]"""
        with self._conn:
            self._conn.executemany(
                "UPDATE chunks SET faiss_id=? WHERE chunk_id=?",
                [(faiss_id, chunk_id) for chunk_id, faiss_id in pairs],
            )

    def get_chunk(self, chunk_id: int) -> ChunkRow | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_faiss_ids(self, chapter_id: str, faiss_ids: list[int]) -> list[ChunkRow]:
        if not faiss_ids:
            return []
        placeholders = ",".join("?" * len(faiss_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chapter_id=? AND faiss_id IN ({placeholders})",
            [chapter_id, *faiss_ids],
        ).fetchall()
        # Preserve the caller-requested faiss_id order.
        by_faiss = {int(r["faiss_id"]): self._row_to_chunk(r) for r in rows}
        return [by_faiss[f] for f in faiss_ids if f in by_faiss]

    def iter_chunks(self, chapter_id: str) -> Iterator[ChunkRow]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE chapter_id=? ORDER BY chunk_id",
            (chapter_id,),
        ).fetchall()
        for r in rows:
            yield self._row_to_chunk(r)

    def chunk_count(self, chapter_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chapter_id=?", (chapter_id,)
        ).fetchone()
        return int(row[0])

    # ---- mapping --------------------------------------------------------

    @staticmethod
    def _row_to_chapter(row: sqlite3.Row) -> ChapterRow:
        return ChapterRow(
            chapter_id=row["chapter_id"],
            title=row["title"],
            subject=row["subject"],
            source_file=row["source_file"],
            page_count=int(row["page_count"]),
            ingestion_status=IngestionStatus(row["ingestion_status"]),
            embedding_model=row["embedding_model"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ChunkRow:
        return ChunkRow(
            chunk_id=int(row["chunk_id"]),
            chapter_id=row["chapter_id"],
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            section_title=row["section_title"],
            chunk_text=row["chunk_text"],
            faiss_id=int(row["faiss_id"]) if row["faiss_id"] is not None else None,
            ocr_quality_status=OcrQualityStatus(row["ocr_quality_status"]),
        )

    # ---- Q&A pairs ----------------------------------------------------------

    def delete_qa_pairs_for_chapter(self, chapter_id: str) -> None:
        self._conn.execute("DELETE FROM qa_pairs WHERE chapter_id=?", (chapter_id,))
        self._conn.commit()

    def insert_qa_pairs(self, pairs: list[QAPairRow]) -> list[int]:
        """Insert Q&A pairs, returning generated qa_ids in input order."""
        ids: list[int] = []
        with self._conn:
            for p in pairs:
                cur = self._conn.execute(
                    """
                    INSERT INTO qa_pairs(chapter_id, question, answer,
                        page_numbers_json, source_chunk_ids_json, faiss_id, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p.chapter_id,
                        p.question,
                        p.answer,
                        json.dumps(p.page_numbers),
                        json.dumps(p.source_chunk_ids),
                        p.faiss_id,
                        p.created_at.isoformat(),
                    ),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def set_qa_faiss_ids(self, pairs: list[tuple[int, int]]) -> None:
        """pairs: [(qa_id, faiss_id), ...]"""
        with self._conn:
            self._conn.executemany(
                "UPDATE qa_pairs SET faiss_id=? WHERE qa_id=?",
                [(faiss_id, qa_id) for qa_id, faiss_id in pairs],
            )

    def qa_pair_count(self, chapter_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM qa_pairs WHERE chapter_id=?", (chapter_id,)
        ).fetchone()
        return int(row[0])

    def get_qa_pairs_by_faiss_ids(
        self, chapter_id: str, faiss_ids: list[int]
    ) -> list[QAPairRow]:
        if not faiss_ids:
            return []
        placeholders = ",".join("?" * len(faiss_ids))
        rows = self._conn.execute(
            f"SELECT * FROM qa_pairs WHERE chapter_id=? AND faiss_id IN ({placeholders})",
            [chapter_id, *faiss_ids],
        ).fetchall()
        by_faiss = {int(r["faiss_id"]): self._row_to_qa(r) for r in rows}
        return [by_faiss[f] for f in faiss_ids if f in by_faiss]

    def iter_qa_pairs(self, chapter_id: str) -> Iterator[QAPairRow]:
        rows = self._conn.execute(
            "SELECT * FROM qa_pairs WHERE chapter_id=? ORDER BY qa_id",
            (chapter_id,),
        ).fetchall()
        for r in rows:
            yield self._row_to_qa(r)

    @staticmethod
    def _row_to_qa(row: sqlite3.Row) -> QAPairRow:
        return QAPairRow(
            qa_id=int(row["qa_id"]),
            chapter_id=row["chapter_id"],
            question=row["question"],
            answer=row["answer"],
            page_numbers=json.loads(row["page_numbers_json"]),
            source_chunk_ids=json.loads(row["source_chunk_ids_json"]),
            faiss_id=int(row["faiss_id"]) if row["faiss_id"] is not None else None,
            created_at=_parse_dt(row["created_at"]),
        )

    # ---- concepts -----------------------------------------------------------

    def delete_concepts_for_chapter(self, chapter_id: str) -> None:
        self._conn.execute("DELETE FROM concepts WHERE chapter_id=?", (chapter_id,))
        self._conn.commit()

    def insert_concepts(self, rows: list[ConceptRow]) -> list[int]:
        ids: list[int] = []
        with self._conn:
            for r in rows:
                cur = self._conn.execute(
                    """
                    INSERT INTO concepts(chapter_id, order_index, title, learning_objective,
                        key_points_json, page_numbers_json, source_chunk_ids_json, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r.chapter_id,
                        r.order_index,
                        r.title,
                        r.learning_objective,
                        json.dumps(r.key_points),
                        json.dumps(r.page_numbers),
                        json.dumps(r.source_chunk_ids),
                        r.created_at.isoformat(),
                    ),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def concept_count(self, chapter_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE chapter_id=?", (chapter_id,)
        ).fetchone()
        return int(row[0])

    def list_concepts(self, chapter_id: str) -> list[ConceptRow]:
        rows = self._conn.execute(
            "SELECT * FROM concepts WHERE chapter_id=? ORDER BY order_index",
            (chapter_id,),
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def get_concept(self, concept_id: int) -> ConceptRow | None:
        row = self._conn.execute(
            "SELECT * FROM concepts WHERE concept_id=?", (concept_id,)
        ).fetchone()
        return self._row_to_concept(row) if row else None

    @staticmethod
    def _row_to_concept(row: sqlite3.Row) -> ConceptRow:
        return ConceptRow(
            concept_id=int(row["concept_id"]),
            chapter_id=row["chapter_id"],
            order_index=int(row["order_index"]),
            title=row["title"],
            learning_objective=row["learning_objective"],
            key_points=json.loads(row["key_points_json"]),
            page_numbers=json.loads(row["page_numbers_json"]),
            source_chunk_ids=json.loads(row["source_chunk_ids_json"]),
            created_at=_parse_dt(row["created_at"]),
        )
