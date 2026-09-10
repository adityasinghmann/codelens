"""
Local vector store
------------------
A SQLite-backed vector store with vectorised cosine similarity, chosen so the
tool needs zero external services and works fully offline: the index is a
single file on disk that can be copied, backed up or deleted.

Search is EXACT, not approximate. Every row for the repository is read and
scored; there is no approximate-nearest-neighbour index, and the code should
not be described as ANN. That is a deliberate trade: exhaustive scoring is
simple and exactly correct, and it is fast enough at the scale a single
repository reaches. It is also the first thing that would need to change to
scale much further.

The class is kept behind a narrow interface so a different storage backend
could be substituted later without touching the indexer or the query path.
"""

import uuid
import os
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional

from backend.config import Settings


class LocalVectorStore:
    """
    Stores chunk metadata and embedding vectors in a local SQLite file.

    Scoring is a single numpy matrix-vector product over every stored
    embedding, which is exact cosine similarity rather than an approximate
    index. The embedding width is whatever the configured model produces; it is
    not assumed to be any particular size.
    """

    def __init__(self):
        self.path = Settings.INDEX_PATH
        os.makedirs(self.path, exist_ok=True)
        self.db_file = os.path.join(self.path, "codelens.db")
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")   # Safe concurrent writes
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
        return conn

    def _init_schema(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id           TEXT PRIMARY KEY,
                    symbol_name  TEXT,
                    chunk_text   TEXT,
                    file_path    TEXT,
                    start_line   INTEGER,
                    end_line     INTEGER,
                    language     TEXT,
                    content_hash TEXT UNIQUE,
                    embedding    BLOB NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash     ON chunks(content_hash)")
            conn.commit()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def batch_upsert(self, metadata_list: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Insert or replace chunks + their embeddings atomically."""
        rows = []
        for meta, emb in zip(metadata_list, embeddings):
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            rows.append((
                str(uuid.uuid4()),
                meta.get("symbol_name", ""),
                meta.get("chunk_text", ""),
                meta.get("file_path", ""),
                meta.get("start_line", 0),
                meta.get("end_line", 0),
                meta.get("language", ""),
                meta.get("content_hash", str(uuid.uuid4())),
                emb_blob,
            ))
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                    (id, symbol_name, chunk_text, file_path,
                     start_line, end_line, language, content_hash, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

    def get_existing_hashes(self, hashes: List[str]) -> List[str]:
        """Return the subset of hashes already stored (for incremental indexing)."""
        if not hashes:
            return []
        placeholders = ",".join("?" * len(hashes))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT content_hash FROM chunks WHERE content_hash IN ({placeholders})",
                hashes,
            ).fetchall()
        return [r[0] for r in rows]

    def delete_by_filepath(self, file_path: str):
        """Remove all chunks for a file (used before re-indexing a modified file)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            conn.commit()

    # ------------------------------------------------------------------
    # Read path - exhaustive cosine similarity
    # ------------------------------------------------------------------

    def search(self, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Exact cosine-similarity search over every stored chunk.

        Reads all embeddings into one numpy matrix and scores them with a
        single matrix-vector product. This is exhaustive, not approximate:
        cost grows linearly with the number of indexed chunks.
        """
        query_vec = np.array(embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        query_vec /= q_norm

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol_name, chunk_text, file_path, "
                "start_line, end_line, language, embedding FROM chunks"
            ).fetchall()

        if not rows:
            return []

        # Stack all embedding blobs into one matrix
        emb_matrix = np.frombuffer(
            b"".join(r[6] for r in rows), dtype=np.float32
        ).reshape(len(rows), -1)

        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        scores = (emb_matrix / norms) @ query_vec   # (N,) cosine scores

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "symbol_name": rows[i][0],
                "chunk_text":  rows[i][1],
                "file_path":   rows[i][2],
                "start_line":  rows[i][3],
                "end_line":    rows[i][4],
                "language":    rows[i][5],
                "score":       float(scores[i]),
            }
            for i in top_indices
        ]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# Singleton - one store per process.
_db_instance: Optional[LocalVectorStore] = None


def get_db() -> LocalVectorStore:
    global _db_instance
    if _db_instance is None:
        _db_instance = LocalVectorStore()
    return _db_instance
