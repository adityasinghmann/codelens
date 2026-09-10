"""Optional search filters, result shape, and the two-phase scoring path."""

import pytest

from backend.db_client import assign_chunk_ids
from tests.conftest import fake_vector


def add(store, repo_id, rel_path, name, text, language="python", symbol_type="function"):
    chunk = {
        "symbol_name": name,
        "qualified_name": f"{rel_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.{name}",
        "symbol_type": symbol_type,
        "parent_symbol": None,
        "file_path": rel_path,
        "start_line": 1,
        "end_line": 5,
        "language": language,
        "chunk_text": text,
        "content_hash": f"h-{text}",
    }
    assign_chunk_ids(repo_id, rel_path, [chunk])
    file_id = store.upsert_file(repo_id, rel_path, "fh", language)
    store.upsert_chunks(repo_id, file_id, [chunk], [fake_vector(text)])
    return chunk


@pytest.fixture
def populated(store, tmp_path):
    repo_id = store.ensure_repository(str(tmp_path))
    add(store, repo_id, "backend/auth.py", "verify_token", "verify token python", "python", "function")
    add(store, repo_id, "backend/db.py", "connect", "connect database python", "python", "function")
    add(store, repo_id, "backend/models.py", "User", "user model class", "python", "class")
    add(store, repo_id, "frontend/api.ts", "fetchUser", "fetch user typescript", "typescript", "function")
    add(store, repo_id, "frontend/ui.ts", "Widget", "widget class typescript", "typescript", "class")
    add(store, repo_id, "frontend/ui.ts", "render", "render method typescript", "typescript", "method")
    return repo_id


def paths(results):
    return sorted(r["file_path"] for r in results)


def test_no_filters_searches_everything(store, populated):
    results = store.search(fake_vector("anything"), repo_id=populated, top_k=20)
    assert len(results) == 6


def test_language_filter(store, populated):
    results = store.search(fake_vector("anything"), repo_id=populated, top_k=20, language="python")
    assert len(results) == 3
    assert all(r["language"] == "python" for r in results)


def test_symbol_type_filter(store, populated):
    results = store.search(fake_vector("anything"), repo_id=populated, top_k=20, symbol_type="class")
    assert sorted(r["symbol_name"] for r in results) == ["User", "Widget"]


def test_path_prefix_filter(store, populated):
    results = store.search(fake_vector("anything"), repo_id=populated, top_k=20, path_prefix="backend/")
    assert paths(results) == ["backend/auth.py", "backend/db.py", "backend/models.py"]


def test_path_prefix_accepts_windows_separators(store, populated):
    results = store.search(fake_vector("x"), repo_id=populated, top_k=20, path_prefix="backend\\")
    assert len(results) == 3


def test_filters_combine(store, populated):
    results = store.search(fake_vector("anything"), repo_id=populated, top_k=20,
                           language="typescript", symbol_type="class", path_prefix="frontend/")
    assert [r["symbol_name"] for r in results] == ["Widget"]


def test_a_filter_matching_nothing_returns_nothing(store, populated):
    assert store.search(fake_vector("x"), repo_id=populated, top_k=20, language="rust") == []


def test_path_prefix_wildcards_are_escaped_not_interpreted(store, populated):
    """A '%' in the prefix must be a literal, not a LIKE wildcard."""
    assert store.search(fake_vector("x"), repo_id=populated, top_k=20, path_prefix="%") == []
    assert store.search(fake_vector("x"), repo_id=populated, top_k=20, path_prefix="_") == []


def test_filters_are_still_scoped_to_the_repository(store, tmp_path, populated):
    other = store.ensure_repository(str(tmp_path / "other"))
    add(store, other, "backend/secret.py", "leak_me", "python content", "python", "function")

    results = store.search(fake_vector("python content"), repo_id=populated,
                           top_k=20, language="python")
    assert "backend/secret.py" not in paths(results)


# --- result shape ---------------------------------------------------------

def test_results_carry_the_documented_fields(store, populated):
    result = store.search(fake_vector("verify token python"), repo_id=populated, top_k=1)[0]
    for key in ("symbol_name", "qualified_name", "file_path", "language",
                "symbol_type", "start_line", "end_line", "score"):
        assert key in result, key
    assert result["symbol_name"] == "verify_token"
    assert result["qualified_name"] == "auth.verify_token"
    assert result["symbol_type"] == "function"


def test_score_is_never_called_confidence(store, populated):
    result = store.search(fake_vector("x"), repo_id=populated, top_k=1)[0]
    assert "score" in result
    assert "confidence" not in result
    assert "probability" not in result


# --- ranking --------------------------------------------------------------

def test_top_k_limits_and_orders_by_descending_score(store, populated):
    results = store.search(fake_vector("connect database python"), repo_id=populated, top_k=3)
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["file_path"] == "backend/db.py"


def test_argpartition_path_returns_the_same_top_k_as_a_full_sort(store, tmp_path):
    """
    np.argpartition replaced a full argsort. The winners and their order must
    be identical, which is the only thing that change is allowed to affect.
    """
    import numpy as np

    repo_id = store.ensure_repository(str(tmp_path))
    for i in range(200):
        add(store, repo_id, f"f{i:03d}.py", f"fn{i}", f"text variant number {i}")

    query = fake_vector("text variant number 77")
    got = store.search(query, repo_id=repo_id, top_k=8)

    # Recompute the expected answer independently, with a full sort.
    with store._conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, embedding, file_path FROM chunks WHERE repo_id = ?", (repo_id,)
        ).fetchall()
    vectors = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    q = np.asarray(query, dtype=np.float32)
    q = q / np.linalg.norm(q)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = (vectors / np.where(norms == 0, 1.0, norms)) @ q
    expected = [rows[i][2] for i in np.argsort(scores)[::-1][:8]]

    assert [r["file_path"] for r in got] == expected


def test_top_k_larger_than_the_corpus_is_handled(store, populated):
    results = store.search(fake_vector("x"), repo_id=populated, top_k=500)
    assert len(results) == 6


def test_top_k_of_one(store, populated):
    assert len(store.search(fake_vector("x"), repo_id=populated, top_k=1)) == 1


def test_hydration_only_reads_the_winners(store, tmp_path, monkeypatch):
    """
    Phase 1 must not SELECT chunk_text. Assert on the SQL actually issued.
    """
    repo_id = store.ensure_repository(str(tmp_path))
    for i in range(50):
        add(store, repo_id, f"f{i}.py", f"fn{i}", f"body number {i}")

    issued = []
    original = store._conn

    def spy():
        conn = original()
        # sqlite3 exposes a trace hook for exactly this; Connection.execute
        # itself is read-only and cannot be wrapped.
        conn.set_trace_callback(lambda sql: issued.append(" ".join(sql.split())))
        return conn

    monkeypatch.setattr(store, "_conn", spy)
    store.search(fake_vector("body number 3"), repo_id=repo_id, top_k=5)

    selects = [s for s in issued if s.upper().startswith("SELECT")]
    scoring = selects[0]
    assert "chunk_text" not in scoring, scoring
    assert "embedding" in scoring

    hydration = [s for s in selects if "chunk_text" in s]
    assert len(hydration) == 1

    # The trace hook reports SQL with bound values already substituted, so
    # count the ids inside IN (...) rather than the placeholders.
    import re
    inside = re.search(r"IN \((.*?)\)", hydration[0], re.S).group(1)
    assert len(inside.split(",")) == 5, hydration[0]
