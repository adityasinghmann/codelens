"""LocalVectorStore: persistence, scoping, ranking and schema migration."""

import os
import sqlite3

import pytest

from backend.db_client import (
    SCHEMA_VERSION,
    LocalVectorStore,
    assign_chunk_ids,
    chunk_id_for,
    file_id_for,
    normalize_rel_path,
    normalize_root,
    repo_id_for,
)
from tests.conftest import fake_vector


def make_chunk(repo_id, rel_path, name, text, start=1, end=2, symbol_type="function"):
    chunk = {
        "symbol_name": name,
        "qualified_name": f"{rel_path.rsplit('.', 1)[0]}.{name}",
        "symbol_type": symbol_type,
        "parent_symbol": None,
        "file_path": rel_path,
        "start_line": start,
        "end_line": end,
        "language": "python",
        "chunk_text": text,
        "content_hash": f"hash-of-{text}",
    }
    assign_chunk_ids(repo_id, rel_path, [chunk])
    return chunk


# --- identity -------------------------------------------------------------

def test_repo_id_is_stable_across_path_spellings(tmp_path):
    root = str(tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    assert repo_id_for(root) == repo_id_for(root + "/.")
    assert repo_id_for(root) == repo_id_for(root.replace("\\", "/"))


def test_chunk_id_excludes_line_numbers():
    """Identity must survive a symbol moving, or incremental indexing breaks."""
    a = chunk_id_for("r", "m.py", "function", "m.f")
    b = chunk_id_for("r", "m.py", "function", "m.f")
    assert a == b


def test_chunk_id_distinguishes_files_types_and_names():
    base = chunk_id_for("r", "a.py", "function", "a.f")
    assert base != chunk_id_for("r", "b.py", "function", "b.f")
    assert base != chunk_id_for("r", "a.py", "method", "a.f")
    assert base != chunk_id_for("r", "a.py", "function", "a.g")
    assert base != chunk_id_for("other", "a.py", "function", "a.f")


def test_duplicate_qualified_names_get_an_occurrence_ordinal():
    chunks = [
        {"symbol_name": "f", "qualified_name": "m.f", "symbol_type": "function",
         "start_line": 1, "end_line": 2},
        {"symbol_name": "f", "qualified_name": "m.f", "symbol_type": "function",
         "start_line": 10, "end_line": 11},
    ]
    assign_chunk_ids("r", "m.py", chunks)
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]


def test_relative_paths_are_normalised_to_forward_slashes():
    assert normalize_rel_path("src\\pkg\\mod.py") == "src/pkg/mod.py"


# --- CRUD -----------------------------------------------------------------

def test_insert_and_read_back(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    file_id = store.upsert_file(repo_id, "a.py", "filehash", "python")
    chunk = make_chunk(repo_id, "a.py", "alpha", "def alpha(): pass")
    store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector("alpha")])

    stored = store.get_chunks_for_file(file_id)
    assert list(stored) == [chunk["chunk_id"]]
    assert store.count(repo_id) == 1
    assert store.count_files(repo_id) == 1


def test_update_replaces_in_place_without_duplicating(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    file_id = store.upsert_file(repo_id, "a.py", "h1", "python")
    chunk = make_chunk(repo_id, "a.py", "alpha", "v1")
    store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector("v1")])

    chunk["content_hash"] = "hash-of-v2"
    chunk["chunk_text"] = "v2"
    store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector("v2")])

    assert store.count(repo_id) == 1
    assert store.get_chunks_for_file(file_id)[chunk["chunk_id"]]["content_hash"] == "hash-of-v2"


def test_update_chunk_lines_leaves_the_embedding_untouched(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    file_id = store.upsert_file(repo_id, "a.py", "h1", "python")
    chunk = make_chunk(repo_id, "a.py", "alpha", "body", start=1, end=2)
    store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector("body")])

    with store._conn() as conn:
        before = conn.execute("SELECT embedding FROM chunks WHERE chunk_id = ?",
                              (chunk["chunk_id"],)).fetchone()[0]

    store.update_chunk_lines([{"chunk_id": chunk["chunk_id"], "start_line": 40, "end_line": 41}])

    with store._conn() as conn:
        row = conn.execute("SELECT embedding, start_line, end_line FROM chunks WHERE chunk_id = ?",
                           (chunk["chunk_id"],)).fetchone()
    assert row[0] == before
    assert (row[1], row[2]) == (40, 41)


def test_delete_chunks(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    file_id = store.upsert_file(repo_id, "a.py", "h", "python")
    chunks = [make_chunk(repo_id, "a.py", f"fn{i}", f"body{i}") for i in range(3)]
    store.upsert_chunks(repo_id, file_id, chunks, [fake_vector(c["chunk_text"]) for c in chunks])

    store.delete_chunks([chunks[0]["chunk_id"], chunks[2]["chunk_id"]])
    assert store.count(repo_id) == 1
    assert list(store.get_chunks_for_file(file_id)) == [chunks[1]["chunk_id"]]


def test_delete_file_removes_its_chunks_and_its_row(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    file_id = store.upsert_file(repo_id, "a.py", "h", "python")
    chunk = make_chunk(repo_id, "a.py", "alpha", "body")
    store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector("body")])

    store.delete_file(repo_id, "a.py")
    assert store.count(repo_id) == 0
    assert store.get_file(repo_id, "a.py") is None


def test_two_identical_chunks_in_different_files_both_survive(store, tmp_path):
    """
    Regression for the old content_hash UNIQUE constraint, which kept only one
    row and reported the wrong file_path for it.
    """
    repo_id = store.ensure_repository(str(tmp_path))
    text = "def helper(x):\n    return x * 2\n"

    for rel in ("alpha.py", "beta.py"):
        file_id = store.upsert_file(repo_id, rel, "h", "python")
        chunk = make_chunk(repo_id, rel, "helper", text)
        store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector(text)])

    with store._conn() as conn:
        paths = sorted(r[0] for r in conn.execute(
            "SELECT file_path FROM chunks WHERE symbol_name = 'helper'").fetchall())
    assert paths == ["alpha.py", "beta.py"]
    assert store.count(repo_id) == 2


# --- search ---------------------------------------------------------------

def _seed(store, root, rel_to_text):
    repo_id = store.ensure_repository(root)
    for rel, text in rel_to_text.items():
        file_id = store.upsert_file(repo_id, rel, "h", "python")
        chunk = make_chunk(repo_id, rel, rel.split(".")[0], text)
        store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector(text)])
    return repo_id


def test_search_returns_top_k_ordered_by_descending_score(store, tmp_path):
    repo_id = _seed(store, str(tmp_path), {f"f{i}.py": f"text number {i}" for i in range(10)})
    results = store.search(fake_vector("text number 3"), repo_id=repo_id, top_k=5)

    assert len(results) == 5
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # The exact text is stored, so it must rank first.
    assert results[0]["chunk_text"] == "text number 3"


def test_search_is_scoped_to_one_repository(store, tmp_path):
    repo_a = _seed(store, str(tmp_path / "a"), {"only_a.py": "alpha content"})
    repo_b = _seed(store, str(tmp_path / "b"), {"only_b.py": "beta content"})

    a_results = store.search(fake_vector("beta content"), repo_id=repo_a, top_k=10)
    b_results = store.search(fake_vector("beta content"), repo_id=repo_b, top_k=10)

    assert [r["file_path"] for r in a_results] == ["only_a.py"]
    assert [r["file_path"] for r in b_results] == ["only_b.py"]


def test_search_on_an_empty_repository_returns_nothing(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    assert store.search(fake_vector("anything"), repo_id=repo_id, top_k=5) == []


def test_search_with_a_zero_query_vector_returns_nothing(store, tmp_path):
    repo_id = _seed(store, str(tmp_path), {"a.py": "content"})
    assert store.search([0.0] * 32, repo_id=repo_id, top_k=5) == []


def test_search_returns_the_full_result_shape(store, tmp_path):
    repo_id = _seed(store, str(tmp_path), {"a.py": "content"})
    result = store.search(fake_vector("content"), repo_id=repo_id, top_k=1)[0]
    for key in ("symbol_name", "qualified_name", "symbol_type", "file_path",
                "start_line", "end_line", "language", "chunk_text", "score"):
        assert key in result, key


# --- repositories ---------------------------------------------------------

def test_delete_repository_leaves_other_repositories_intact(store, tmp_path):
    repo_a = _seed(store, str(tmp_path / "a"), {"a.py": "alpha"})
    repo_b = _seed(store, str(tmp_path / "b"), {"b.py": "beta"})

    store.delete_repository(repo_b)

    assert store.count(repo_a) == 1
    assert store.count(repo_b) == 0
    assert store.get_repository(repo_b) is None
    assert store.get_repository(repo_a) is not None


def test_clear_repository_content_keeps_the_repository_row(store, tmp_path):
    repo_id = _seed(store, str(tmp_path), {"a.py": "alpha"})
    store.clear_repository_content(repo_id)
    assert store.count(repo_id) == 0
    assert store.count_files(repo_id) == 0
    assert store.get_repository(repo_id) is not None


def test_most_recent_repository_tracks_the_latest_touch(store, tmp_path):
    repo_a = store.ensure_repository(str(tmp_path / "a"))
    repo_b = store.ensure_repository(str(tmp_path / "b"))
    store.touch_repository(repo_a)
    assert store.most_recent_repository()["repo_id"] == repo_a


def test_ensure_repository_is_idempotent(store, tmp_path):
    first = store.ensure_repository(str(tmp_path))
    second = store.ensure_repository(str(tmp_path))
    assert first == second
    assert len(store.list_repositories()) == 1


# --- metadata -------------------------------------------------------------

def test_embedding_metadata_round_trip(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    store.set_index_metadata(repo_id, embedding_model="nomic-embed-text",
                             embedding_dimension=768, status="ready")
    meta = store.get_index_metadata(repo_id)
    assert meta["embedding_model"] == "nomic-embed-text"
    assert meta["embedding_dimension"] == 768
    assert meta["status"] == "ready"
    assert meta["schema_version"] == SCHEMA_VERSION


def test_partial_metadata_update_preserves_other_fields(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    store.set_index_metadata(repo_id, embedding_model="m", embedding_dimension=64)
    store.set_index_metadata(repo_id, status="rebuilding")
    meta = store.get_index_metadata(repo_id)
    assert meta["embedding_model"] == "m"
    assert meta["embedding_dimension"] == 64
    assert meta["status"] == "rebuilding"


def test_metadata_is_absent_for_an_unknown_repository(store):
    assert store.get_index_metadata("no-such-repo") is None


# --- migration ------------------------------------------------------------

def test_an_old_schema_database_is_dropped_and_rebuilt(index_dir):
    """
    The pre-migration database had one global `chunks` table with a UNIQUE
    content_hash and no repo_id. Opening it must rebuild rather than leave an
    incompatible schema in place.
    """
    db_file = f"{index_dir}/codelens.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, symbol_name TEXT, chunk_text TEXT,
            file_path TEXT, start_line INTEGER, end_line INTEGER,
            language TEXT, content_hash TEXT UNIQUE, embedding BLOB NOT NULL
        );
        """
    )
    conn.execute("INSERT INTO chunks VALUES ('1','f','t','a.py',1,2,'python','h',x'00')")
    conn.commit()
    conn.close()

    store = LocalVectorStore(path=index_dir)

    with store._conn() as check:
        version = check.execute("PRAGMA user_version").fetchone()[0]
        columns = {row[1] for row in check.execute("PRAGMA table_info(chunks)").fetchall()}
        tables = {row[0] for row in check.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    assert version == SCHEMA_VERSION
    assert "repo_id" in columns
    assert "chunk_id" in columns
    assert {"repositories", "files", "chunks", "index_metadata"} <= tables
    # The stale row is gone; the index is a rebuildable cache.
    assert store.count() == 0


def test_reopening_a_current_database_preserves_its_contents(index_dir, tmp_path):
    store = LocalVectorStore(path=index_dir)
    repo_id = store.ensure_repository(str(tmp_path))
    file_id = store.upsert_file(repo_id, "a.py", "h", "python")
    chunk = make_chunk(repo_id, "a.py", "alpha", "body")
    store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector("body")])

    reopened = LocalVectorStore(path=index_dir)
    assert reopened.count(repo_id) == 1
    assert reopened.get_repository(repo_id) is not None


def test_stored_root_path_keeps_the_users_casing(store, tmp_path):
    """
    repo_id is case-folded for identity, but the path shown in /status must
    not come back lowercased on Windows.
    """
    root = tmp_path / "MixedCase" / "RepoName"
    root.mkdir(parents=True)

    repo_id = store.ensure_repository(str(root))
    stored = store.get_repository(repo_id)["root_path"]

    assert "MixedCase" in stored
    assert "RepoName" in stored
    # Identity is still case-insensitive on platforms where that matters.
    assert repo_id_for(str(root).lower()) == repo_id_for(str(root).upper()) or os.name != "nt"
