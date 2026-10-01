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

LocalVectorStore implements the VectorStore Protocol in backend/vector_store.py.
That interface is where the contract lives; this module is the only
implementation of it, and no second backend exists.

Schema
------
    repositories   one row per indexed repo, keyed by its resolved root path
    files          one row per indexed file, scoped to a repo
    chunks         one row per semantic code unit, scoped to a file and repo
    index_metadata per-repo embedding model / dimension / status

Identity is separated from change detection, which the previous single-table
schema conflated:

    chunk_id     = hash(repo_id, file_path, symbol_type, qualified_name[, n])
    content_hash = hash(chunk_text)

chunk_id deliberately excludes line numbers. Adding an import at the top of a
file shifts every symbol below it; line-based identity would change every
chunk_id and force a full re-embed of the file, defeating incremental
indexing. Lines are mutable metadata, updated in place on the existing row.
"""

import os
import hashlib
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Iterable, Sequence

import numpy as np

from backend.config import Settings
from backend.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Bump when the physical schema changes. On mismatch the index is dropped and
# rebuilt: it is a derived cache, never a source of truth.
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def _sha1(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")  # unambiguous separator
    return h.hexdigest()


def display_root(path: str) -> str:
    """
    Resolved repository root with the user's original casing preserved.

    This is what gets stored and shown back. Identity uses normalize_root();
    displaying that would print a lowercased path on Windows, which looks wrong
    even though it is the correct key.
    """
    return os.path.realpath(os.path.abspath(path))


def normalize_root(path: str) -> str:
    """
    Canonical form of a repository root, used for identity only.

    Resolves symlinks and relative segments, then applies normcase so that on
    Windows C:\\Repo and c:\\repo are the same repository. This is what makes
    repo_id stable across however the user happened to type the path.
    """
    return os.path.normcase(display_root(path))


def repo_id_for(root_path: str) -> str:
    """Stable id for a repository, derived from its normalised root path."""
    return _sha1("repo", normalize_root(root_path))


def normalize_rel_path(rel_path: str) -> str:
    """
    Repository-relative path in a canonical, portable form.

    Always forward slashes: os.path.relpath yields backslashes on Windows, and
    storing those would make the same repository index differently depending on
    the host OS, and break jump-to-file across platforms.

    Only a literal "./" prefix and leading slashes are removed. str.lstrip("./")
    would strip every leading "." too, turning ".github/ci.yml" into
    "github/ci.yml" - a path that does not exist.
    """
    path = rel_path.replace(os.sep, "/").replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def file_id_for(repo_id: str, rel_path: str) -> str:
    return _sha1("file", repo_id, normalize_rel_path(rel_path))


def chunk_id_for(
    repo_id: str,
    rel_path: str,
    symbol_type: str,
    qualified_name: str,
    ordinal: int = 0,
) -> str:
    """
    Stable identity for one semantic unit.

    Note what is absent: start_line and end_line. See the module docstring.

    `ordinal` disambiguates the rare case where qualified_name is not unique
    within a file (an overload, or two same-named symbols the parser could not
    tell apart). It is assigned by assign_chunk_ids in source order.
    """
    return _sha1("chunk", repo_id, normalize_rel_path(rel_path),
                 symbol_type, qualified_name, str(ordinal))


def assign_chunk_ids(repo_id: str, rel_path: str, chunks: List[Dict[str, Any]]) -> None:
    """
    Give every chunk in one file a stable chunk_id, in place.

    Chunks are keyed on (symbol_type, qualified_name). Where that collides
    within a file, an occurrence ordinal is appended in source order so the
    ids stay distinct and stable as long as the symbols keep their relative
    order.
    """
    seen: Dict[tuple, int] = {}
    for chunk in sorted(chunks, key=lambda c: (c.get("start_line", 0), c.get("end_line", 0))):
        qualified = chunk.get("qualified_name") or chunk.get("symbol_name", "")
        symbol_type = chunk.get("symbol_type", "unknown")
        key = (symbol_type, qualified)
        ordinal = seen.get(key, 0)
        seen[key] = ordinal + 1
        chunk["qualified_name"] = qualified
        chunk["symbol_type"] = symbol_type
        chunk["chunk_id"] = chunk_id_for(repo_id, rel_path, symbol_type, qualified, ordinal)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class LocalVectorStore:
    """
    Stores chunk metadata and embedding vectors in a local SQLite file.

    Scoring is a single numpy matrix-vector product over the embeddings of one
    repository, which is exact cosine similarity rather than an approximate
    index. The embedding width is whatever the configured model produces; it is
    not assumed to be any particular size.

    Implements the VectorStore Protocol structurally - there is no inheritance,
    so this module stays unaware of the interface module.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or Settings.INDEX_PATH
        os.makedirs(self.path, exist_ok=True)
        self.db_file = os.path.join(self.path, "codelens.db")
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- schema ------------------------------------------------------------

    def _init_schema(self):
        """
        Create the schema, rebuilding it if it was written by an older version.

        CREATE TABLE IF NOT EXISTS alone would silently leave an incompatible
        database in place, which is what the previous implementation did. The
        index is a rebuildable cache, so a version mismatch drops and recreates
        it and tells the user to re-index.
        """
        with self._conn() as conn:
            found = conn.execute("PRAGMA user_version").fetchone()[0]

            has_tables = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
                "('chunks','files','repositories','index_metadata')"
            ).fetchone()[0]

            if has_tables and found != SCHEMA_VERSION:
                logger.warning(
                    "Index schema version %s does not match %s; dropping and rebuilding "
                    "the index. Re-index your repositories to restore search.",
                    found or "0 (pre-versioning)", SCHEMA_VERSION,
                )
                for table in ("chunks", "files", "index_metadata", "repositories"):
                    conn.execute(f"DROP TABLE IF EXISTS {table}")

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS repositories (
                    repo_id    TEXT PRIMARY KEY,
                    root_path  TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS files (
                    file_id      TEXT PRIMARY KEY,
                    repo_id      TEXT NOT NULL,
                    path         TEXT NOT NULL,
                    content_hash TEXT,
                    language     TEXT,
                    indexed_at   TEXT,
                    FOREIGN KEY (repo_id) REFERENCES repositories(repo_id) ON DELETE CASCADE,
                    UNIQUE (repo_id, path)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id       TEXT PRIMARY KEY,
                    file_id        TEXT NOT NULL,
                    repo_id        TEXT NOT NULL,
                    symbol_name    TEXT,
                    qualified_name TEXT,
                    symbol_type    TEXT,
                    parent_symbol  TEXT,
                    file_path      TEXT NOT NULL,
                    start_line     INTEGER,
                    end_line       INTEGER,
                    language       TEXT,
                    chunk_text     TEXT,
                    content_hash   TEXT NOT NULL,
                    embedding      BLOB NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE,
                    FOREIGN KEY (repo_id) REFERENCES repositories(repo_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_metadata (
                    repo_id             TEXT PRIMARY KEY,
                    embedding_model     TEXT,
                    embedding_dimension INTEGER,
                    schema_version      INTEGER,
                    status              TEXT,
                    updated_at          TEXT,
                    FOREIGN KEY (repo_id) REFERENCES repositories(repo_id) ON DELETE CASCADE
                );

                -- Indexes on the columns actually queried.
                CREATE INDEX IF NOT EXISTS idx_chunks_repo    ON chunks(repo_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_file    ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_repo_fp ON chunks(repo_id, file_path);
                CREATE INDEX IF NOT EXISTS idx_files_repo     ON files(repo_id, path);
                """
            )
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()

    # -- repositories ------------------------------------------------------

    def ensure_repository(self, root_path: str) -> str:
        """
        Register a repository (idempotent) and return its repo_id.

        An existing row is left untouched. updated_at means "last indexed" -
        it is reported as last_indexed and orders "most recently indexed" -
        so only touch_repository(), at the end of a successful index, moves
        it. Bumping it here made every backend restart look like an index run.
        """
        repo_id = repo_id_for(root_path)
        # Identity is case-folded; the stored path keeps the user's casing.
        display = display_root(root_path)
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO repositories (repo_id, root_path, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(repo_id) DO NOTHING",
                (repo_id, display, now, now),
            )
            conn.commit()
        return repo_id

    def get_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT repo_id, root_path, created_at, updated_at FROM repositories WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()
        if not row:
            return None
        return {"repo_id": row[0], "root_path": row[1], "created_at": row[2], "updated_at": row[3]}

    def list_repositories(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT repo_id, root_path, created_at, updated_at FROM repositories ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {"repo_id": r[0], "root_path": r[1], "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    def most_recent_repository(self) -> Optional[Dict[str, Any]]:
        """Replaces the last_repo.json sidecar."""
        repos = self.list_repositories()
        return repos[0] if repos else None

    def touch_repository(self, repo_id: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE repositories SET updated_at = ? WHERE repo_id = ?", (_now(), repo_id))
            conn.commit()

    def delete_repository(self, repo_id: str) -> None:
        """Remove one repository and everything under it. Never touches others."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM index_metadata WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM repositories WHERE repo_id = ?", (repo_id,))
            conn.commit()

    def clear_repository_content(self, repo_id: str) -> None:
        """Drop a repository's files and chunks but keep the repository row."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            conn.commit()

    # -- files -------------------------------------------------------------

    def upsert_file(self, repo_id: str, rel_path: str, content_hash: str,
                    language: str = "") -> str:
        rel = normalize_rel_path(rel_path)
        file_id = file_id_for(repo_id, rel)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO files (file_id, repo_id, path, content_hash, language, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(file_id) DO UPDATE SET "
                "  content_hash = excluded.content_hash, "
                "  language     = excluded.language, "
                "  indexed_at   = excluded.indexed_at",
                (file_id, repo_id, rel, content_hash, language, _now()),
            )
            conn.commit()
        return file_id

    def get_file(self, repo_id: str, rel_path: str) -> Optional[Dict[str, Any]]:
        rel = normalize_rel_path(rel_path)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT file_id, path, content_hash, language, indexed_at "
                "FROM files WHERE repo_id = ? AND path = ?",
                (repo_id, rel),
            ).fetchone()
        if not row:
            return None
        return {"file_id": row[0], "path": row[1], "content_hash": row[2],
                "language": row[3], "indexed_at": row[4]}

    def list_files(self, repo_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT file_id, path, content_hash, language, indexed_at FROM files WHERE repo_id = ?",
                (repo_id,),
            ).fetchall()
        return [
            {"file_id": r[0], "path": r[1], "content_hash": r[2], "language": r[3], "indexed_at": r[4]}
            for r in rows
        ]

    def delete_file(self, repo_id: str, rel_path: str) -> None:
        """Remove a file and its chunks from one repository."""
        rel = normalize_rel_path(rel_path)
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE repo_id = ? AND file_path = ?", (repo_id, rel))
            conn.execute("DELETE FROM files WHERE repo_id = ? AND path = ?", (repo_id, rel))
            conn.commit()

    # -- chunks ------------------------------------------------------------

    def get_chunks_for_file(self, file_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Stored chunks for one file, keyed by chunk_id.

        Used by the incremental diff, so it returns content_hash and line
        numbers but deliberately not the embedding blobs.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, content_hash, start_line, end_line, symbol_name, "
                "       qualified_name, symbol_type "
                "FROM chunks WHERE file_id = ?",
                (file_id,),
            ).fetchall()
        return {
            r[0]: {
                "chunk_id": r[0], "content_hash": r[1], "start_line": r[2], "end_line": r[3],
                "symbol_name": r[4], "qualified_name": r[5], "symbol_type": r[6],
            }
            for r in rows
        }

    def upsert_chunks(self, repo_id: str, file_id: str,
                      chunks: Sequence[Dict[str, Any]],
                      embeddings: Sequence[Sequence[float]]) -> None:
        """
        Insert or replace chunks keyed by chunk_id.

        Unlike the previous implementation there is no UNIQUE constraint on
        content_hash: two byte-identical functions in different files are two
        distinct chunks and must both survive.
        """
        if not chunks:
            return
        rows = []
        for meta, emb in zip(chunks, embeddings):
            vec = np.asarray(emb, dtype=np.float32)
            rows.append((
                meta["chunk_id"],
                file_id,
                repo_id,
                meta.get("symbol_name", ""),
                meta.get("qualified_name", meta.get("symbol_name", "")),
                meta.get("symbol_type", ""),
                meta.get("parent_symbol"),
                normalize_rel_path(meta.get("file_path", "")),
                meta.get("start_line", 0),
                meta.get("end_line", 0),
                meta.get("language", ""),
                meta.get("chunk_text", ""),
                meta["content_hash"],
                vec.tobytes(),
            ))
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, file_id, repo_id, symbol_name, qualified_name, symbol_type, "
                " parent_symbol, file_path, start_line, end_line, language, chunk_text, "
                " content_hash, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    def update_chunk_lines(self, updates: Iterable[Dict[str, Any]]) -> None:
        """
        Move an unchanged chunk's line numbers without touching its embedding.

        This is the whole point of keeping lines out of chunk_id: an edit above
        a function shifts it, and that must not cost a re-embed.
        """
        rows = [(u["start_line"], u["end_line"], u["chunk_id"]) for u in updates]
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                "UPDATE chunks SET start_line = ?, end_line = ? WHERE chunk_id = ?", rows
            )
            conn.commit()

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._conn() as conn:
            conn.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(c,) for c in chunk_ids])
            conn.commit()

    # -- metadata ----------------------------------------------------------

    def set_index_metadata(self, repo_id: str, embedding_model: Optional[str] = None,
                           embedding_dimension: Optional[int] = None,
                           status: Optional[str] = None) -> None:
        existing = self.get_index_metadata(repo_id) or {}
        model = embedding_model if embedding_model is not None else existing.get("embedding_model")
        dim = embedding_dimension if embedding_dimension is not None else existing.get("embedding_dimension")
        state = status if status is not None else existing.get("status")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO index_metadata "
                "(repo_id, embedding_model, embedding_dimension, schema_version, status, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(repo_id) DO UPDATE SET "
                "  embedding_model = excluded.embedding_model, "
                "  embedding_dimension = excluded.embedding_dimension, "
                "  schema_version = excluded.schema_version, "
                "  status = excluded.status, "
                "  updated_at = excluded.updated_at",
                (repo_id, model, dim, SCHEMA_VERSION, state, _now()),
            )
            conn.commit()

    def get_index_metadata(self, repo_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT embedding_model, embedding_dimension, schema_version, status, updated_at "
                "FROM index_metadata WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()
        if not row:
            return None
        return {"embedding_model": row[0], "embedding_dimension": row[1],
                "schema_version": row[2], "status": row[3], "updated_at": row[4]}

    # -- read path ---------------------------------------------------------

    def search(self, embedding: Sequence[float], repo_id: str, top_k: int = 10,
               language: Optional[str] = None, path_prefix: Optional[str] = None,
               symbol_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Exact cosine-similarity search within ONE repository.

        repo_id is mandatory. The previous implementation had a single global
        chunks table with no repo dimension, so indexing repo A and then
        searching from repo B returned A's code.

        Optional filters are applied in SQL, before any scoring, so an
        excluded chunk is never read or multiplied.

        Two costs the earlier version paid needlessly:
          * it SELECTed chunk_text for every row purely to score it, then threw
            all but top_k away. Scoring now reads only (chunk_id, embedding) and
            the winners are hydrated afterwards.
          * it fully sorted N scores with argsort to keep 8. np.argpartition
            finds the top_k in linear time and only those are sorted.
        """
        query_vec = np.asarray(embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        query_vec = query_vec / q_norm

        where = ["repo_id = ?"]
        params: List[Any] = [repo_id]
        if language:
            where.append("language = ?")
            params.append(language)
        if symbol_type:
            where.append("symbol_type = ?")
            params.append(symbol_type)
        if path_prefix:
            prefix = normalize_rel_path(path_prefix)
            # Exact, case-sensitive prefix comparison. SQLite's LIKE ignores
            # ASCII case, so "backend/" also matched a different directory
            # "Backend/". substr() has no wildcards, so nothing needs escaping.
            where.append("substr(file_path, 1, ?) = ?")
            params.extend([len(prefix), prefix])

        clause = " AND ".join(where)

        # Phase 1: score against ids and vectors only.
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, embedding FROM chunks WHERE {clause}", params
            ).fetchall()

        if not rows:
            return []

        width = len(query_vec)
        ids, blobs = [], []
        for chunk_id, blob in rows:
            # A row whose width does not match the query cannot be scored. That
            # only happens if the model changed without a re-index; the query
            # layer refuses that case up front, so skip defensively here.
            if len(blob) != width * 4:
                continue
            ids.append(chunk_id)
            blobs.append(blob)

        if not ids:
            return []

        emb_matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(ids), width)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        scores = (emb_matrix / norms) @ query_vec

        k = min(top_k, len(scores))
        if k < len(scores):
            candidates = np.argpartition(scores, -k)[-k:]
        else:
            candidates = np.arange(len(scores))
        top_indices = candidates[np.argsort(scores[candidates])[::-1]]

        # Phase 2: hydrate only the winners.
        winners = [ids[i] for i in top_indices]
        placeholders = ",".join("?" * len(winners))
        with self._conn() as conn:
            hydrated = conn.execute(
                "SELECT chunk_id, symbol_name, qualified_name, symbol_type, file_path, "
                "       start_line, end_line, language, chunk_text "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                winners,
            ).fetchall()

        by_id = {row[0]: row for row in hydrated}
        results: List[Dict[str, Any]] = []
        for index in top_indices:
            row = by_id.get(ids[index])
            if row is None:
                continue  # deleted between the two queries
            results.append({
                "symbol_name": row[1],
                "qualified_name": row[2],
                "symbol_type": row[3],
                "file_path": row[4],
                "start_line": row[5],
                "end_line": row[6],
                "language": row[7],
                "chunk_text": row[8],
                "score": float(scores[index]),
            })
        return results

    def count(self, repo_id: Optional[str] = None) -> int:
        with self._conn() as conn:
            if repo_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE repo_id = ?", (repo_id,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def count_files(self, repo_id: Optional[str] = None) -> int:
        with self._conn() as conn:
            if repo_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM files WHERE repo_id = ?", (repo_id,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]


# Singleton - one store per process.
_db_instance: Optional[VectorStore] = None


def get_db() -> VectorStore:
    global _db_instance
    if _db_instance is None:
        _db_instance = LocalVectorStore()
    return _db_instance


def reset_db_singleton() -> None:
    """Drop the cached store. Used by tests that point at a temp directory."""
    global _db_instance
    _db_instance = None
