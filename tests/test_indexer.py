"""Incremental indexing: what gets re-embedded, deleted, kept and isolated."""

import os

import pytest

from backend.db_client import file_id_for
from backend.indexer import Indexer, chunk_file, md5_hash

IDENTICAL = "def helper(x):\n    return x * 2\n"

V1 = '''def A(x):
    return x + 1


def B(x):
    return x + 2


def C(x):
    return x + 3
'''

# The brief's worked example: A unchanged, B changed, C gone, D new.
V2 = '''def A(x):
    return x + 1


def B(x):
    return x + 999


def D(x):
    return x + 4
'''


def write(repo, rel, body):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    return path


def symbols(store, repo_id):
    with store._conn() as conn:
        return sorted(r[0] for r in conn.execute(
            "SELECT symbol_name FROM chunks WHERE repo_id = ?", (repo_id,)).fetchall())


# --- new / unchanged / modified -------------------------------------------

async def test_new_file_is_indexed(store, make_repo, index_repo):
    repo = make_repo({"m.py": V1})
    indexer = Indexer(repo)
    counts = await index_repo(indexer, repo)

    assert counts["stored"] == 3
    assert counts["skipped"] == 0
    assert symbols(store, indexer.repo_id) == ["A", "B", "C"]


async def test_reindexing_an_unchanged_file_stores_nothing_and_reports_skipped(
    store, make_repo, index_repo, embed_calls
):
    repo = make_repo({"m.py": V1})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    embed_calls.clear()
    counts = await index_repo(indexer, repo)

    assert counts["stored"] == 0
    assert counts["skipped"] == 3
    assert embed_calls == [], "an unchanged file must not be re-embedded"


async def test_modified_file_reembeds_only_the_changed_chunks(
    store, make_repo, index_repo, embed_calls
):
    """The brief's worked example, asserted on actual embed calls."""
    repo = make_repo({"m.py": V1})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    before = dict(store.get_chunks_for_file(file_id_for(indexer.repo_id, "m.py")))
    id_of_a = next(cid for cid, meta in before.items() if meta["symbol_name"] == "A")

    embed_calls.clear()
    write(repo, "m.py", V2)
    counts = await index_repo(indexer, repo)

    # A untouched and NOT re-embedded; B re-embedded; D embedded; C deleted.
    assert not any("x + 1" in text for text in embed_calls)
    assert any("x + 999" in text for text in embed_calls)
    assert any("x + 4" in text for text in embed_calls)
    assert len(embed_calls) == 2, embed_calls

    assert counts == {"stored": 2, "skipped": 1, "deleted": 1, "failed": 0}
    assert symbols(store, indexer.repo_id) == ["A", "B", "D"]

    after = store.get_chunks_for_file(file_id_for(indexer.repo_id, "m.py"))
    assert id_of_a in after, "A must keep its chunk_id across the edit"


async def test_shifting_every_symbol_costs_no_embeddings(
    store, make_repo, index_repo, embed_calls
):
    """Adding imports moves every line but changes no chunk's text."""
    repo = make_repo({"m.py": V1})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    embed_calls.clear()
    write(repo, "m.py", "import os\nimport sys\n\n" + V1)
    counts = await index_repo(indexer, repo)

    assert embed_calls == []
    assert counts["skipped"] == 3
    stored = store.get_chunks_for_file(file_id_for(indexer.repo_id, "m.py"))
    a_meta = next(m for m in stored.values() if m["symbol_name"] == "A")
    assert a_meta["start_line"] == 4, "line numbers must still be updated"


async def test_file_metadata_is_updated(store, make_repo, index_repo):
    repo = make_repo({"m.py": V1})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    record = store.get_file(indexer.repo_id, "m.py")
    assert record["content_hash"] == md5_hash(V1)
    assert record["indexed_at"]
    assert record["language"] == "python"


# --- delete / rename ------------------------------------------------------

async def test_deleted_file_loses_its_chunks(store, make_repo, index_repo):
    repo = make_repo({"keep.py": "def keep():\n    return 1\n",
                      "gone.py": "def gone():\n    return 2\n"})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)
    assert symbols(store, indexer.repo_id) == ["gone", "keep"]

    os.remove(os.path.join(repo, "gone.py"))
    indexer.remove_file(os.path.join(repo, "gone.py"))

    assert symbols(store, indexer.repo_id) == ["keep"]
    assert store.get_file(indexer.repo_id, "gone.py") is None


async def test_renamed_file_moves_to_its_new_path(store, make_repo, index_repo):
    repo = make_repo({"before.py": "def moves():\n    return 7\n"})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    old = os.path.join(repo, "before.py")
    new = os.path.join(repo, "after.py")
    os.rename(old, new)
    indexer.remove_file(old)
    await indexer.reindex_file_async(new)

    assert store.get_file(indexer.repo_id, "before.py") is None
    assert store.get_file(indexer.repo_id, "after.py") is not None
    assert symbols(store, indexer.repo_id) == ["moves"]

    with store._conn() as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT file_path FROM chunks WHERE repo_id = ?", (indexer.repo_id,)).fetchall()]
    assert paths == ["after.py"]


# --- identity regression --------------------------------------------------

async def test_two_files_with_byte_identical_functions_both_survive(
    store, make_repo, index_repo
):
    """
    Regression for content_hash UNIQUE: the second file's copy used to
    overwrite the first, leaving one row with the wrong file_path.
    """
    repo = make_repo({"alpha.py": IDENTICAL, "beta.py": IDENTICAL})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    with store._conn() as conn:
        rows = conn.execute(
            "SELECT file_path, content_hash FROM chunks "
            "WHERE repo_id = ? AND symbol_name = 'helper' ORDER BY file_path",
            (indexer.repo_id,)).fetchall()

    assert [r[0] for r in rows] == ["alpha.py", "beta.py"]
    # Same text, so the same content_hash - which is correct: it is a change
    # detector, not an identity.
    assert rows[0][1] == rows[1][1]


async def test_identical_functions_report_their_own_file_in_search(
    store, make_repo, index_repo
):
    from tests.conftest import fake_vector

    repo = make_repo({"alpha.py": IDENTICAL, "beta.py": IDENTICAL})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    hits = store.search(fake_vector(f"search_document: {IDENTICAL}"),
                        repo_id=indexer.repo_id, top_k=10)
    assert sorted(h["file_path"] for h in hits) == ["alpha.py", "beta.py"]


# --- force reindex --------------------------------------------------------

async def test_force_reindex_rebuilds_every_chunk(store, make_repo, index_repo, embed_calls):
    repo = make_repo({"m.py": V1})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    store.clear_repository_content(indexer.repo_id)
    assert store.count(indexer.repo_id) == 0

    embed_calls.clear()
    counts = await index_repo(indexer, repo)
    assert counts["stored"] == 3
    assert counts["skipped"] == 0
    assert len(embed_calls) == 3
    assert store.count(indexer.repo_id) == 3


async def test_force_reindex_does_not_touch_another_repository(
    store, make_repo, index_repo
):
    repo_a = make_repo({"a.py": "def alpha():\n    return 1\n"}, name="a")
    repo_b = make_repo({"b.py": "def beta():\n    return 2\n"}, name="b")
    indexer_a = Indexer(repo_a)
    indexer_b = Indexer(repo_b)
    await index_repo(indexer_a, repo_a)
    await index_repo(indexer_b, repo_b)

    store.clear_repository_content(indexer_a.repo_id)

    assert store.count(indexer_a.repo_id) == 0
    assert store.count(indexer_b.repo_id) == 1
    assert symbols(store, indexer_b.repo_id) == ["beta"]


# --- repository isolation -------------------------------------------------

async def test_two_repositories_do_not_cross_contaminate(store, make_repo, index_repo):
    from tests.conftest import fake_vector

    repo_a = make_repo({"a.py": "def only_in_alpha():\n    return 1\n"}, name="a")
    repo_b = make_repo({"b.py": "def only_in_beta():\n    return 2\n"}, name="b")
    indexer_a = Indexer(repo_a)
    indexer_b = Indexer(repo_b)
    await index_repo(indexer_a, repo_a)
    await index_repo(indexer_b, repo_b)

    assert symbols(store, indexer_a.repo_id) == ["only_in_alpha"]
    assert symbols(store, indexer_b.repo_id) == ["only_in_beta"]

    beta_vector = fake_vector("search_document: def only_in_beta():\n    return 2\n")
    from_a = store.search(beta_vector, repo_id=indexer_a.repo_id, top_k=10)
    assert all(hit["file_path"] != "b.py" for hit in from_a)


async def test_same_path_spelled_differently_is_one_repository(store, make_repo, index_repo):
    repo = make_repo({"m.py": V1})
    first = Indexer(repo)
    second = Indexer(repo + os.sep + ".")
    assert first.repo_id == second.repo_id
    assert len(store.list_repositories()) == 1


# --- walking --------------------------------------------------------------

async def test_ignored_directories_are_skipped(store, make_repo):
    repo = make_repo({
        "src/real.py": "def real():\n    return 1\n",
        "node_modules/pkg/junk.py": "def junk():\n    return 2\n",
        "__pycache__/cached.py": "def cached():\n    return 3\n",
    })
    indexer = Indexer(repo)
    found = [os.path.basename(p) for p in indexer.walk_repo()]
    assert found == ["real.py"]


async def test_unreadable_file_yields_no_chunks_without_raising(store, make_repo):
    repo = make_repo({"m.py": V1})
    indexer = Indexer(repo)
    parsed = chunk_file(os.path.join(repo, "does_not_exist.py"), repo, indexer.repo_id)
    assert parsed.chunks == []
