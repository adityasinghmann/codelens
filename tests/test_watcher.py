"""
Filesystem watcher: create, modify, delete, move, and debouncing.

These drive the real watchdog Observer against real files, because the bugs
being guarded against were in which events the handler implemented at all.
Each test waits for the debounce window and then flushes, so no test depends
on a race resolving in its favour.
"""

import os
import time

import pytest

from backend.indexer import Indexer, chunk_file

DEBOUNCE = 0.2
SETTLE = 1.5


@pytest.fixture
def watched(store, make_repo, fake_embedder, embed_calls):
    """A repository with an indexed seed file and a running watcher."""
    import asyncio

    repo = make_repo({"seed.py": "def seed():\n    return 1\n"})
    indexer = Indexer(repo)

    async def _seed():
        for path in indexer.walk_repo():
            await indexer.index_file(chunk_file(path, repo, indexer.repo_id))

    asyncio.run(_seed())
    indexer.start_watchdog(debounce_seconds=DEBOUNCE)
    try:
        yield repo, indexer
    finally:
        indexer.stop_watchdog()


def settle(indexer, seconds: float = SETTLE):
    """Let watchdog deliver, let the debounce expire, then force the work."""
    time.sleep(seconds)
    indexer.worker.flush()
    time.sleep(0.3)


def files_in(store, repo_id):
    return sorted(f["path"] for f in store.list_files(repo_id))


def symbols_in(store, repo_id):
    with store._conn() as conn:
        return sorted(r[0] for r in conn.execute(
            "SELECT symbol_name FROM chunks WHERE repo_id = ?", (repo_id,)).fetchall())


def write(repo, rel, body):
    with open(os.path.join(repo, rel), "w", encoding="utf-8") as handle:
        handle.write(body)


def test_created_file_is_indexed(store, watched):
    repo, indexer = watched
    write(repo, "created.py", "def brand_new():\n    return 42\n")
    settle(indexer)

    assert "created.py" in files_in(store, indexer.repo_id)
    assert "brand_new" in symbols_in(store, indexer.repo_id)


def test_modified_file_picks_up_new_symbols(store, watched):
    repo, indexer = watched
    write(repo, "seed.py", "def seed():\n    return 1\n\n\ndef added_later():\n    return 2\n")
    settle(indexer)

    assert "added_later" in symbols_in(store, indexer.repo_id)


def test_deleted_file_loses_its_row_and_chunks(store, watched):
    repo, indexer = watched
    write(repo, "temp.py", "def temporary():\n    return 1\n")
    settle(indexer)
    assert "temporary" in symbols_in(store, indexer.repo_id)

    os.remove(os.path.join(repo, "temp.py"))
    settle(indexer)

    assert "temp.py" not in files_in(store, indexer.repo_id)
    assert "temporary" not in symbols_in(store, indexer.repo_id)


def test_moved_file_leaves_the_old_path_and_appears_at_the_new_one(store, watched):
    repo, indexer = watched
    write(repo, "before.py", "def moves_around():\n    return 7\n")
    settle(indexer)
    assert "before.py" in files_in(store, indexer.repo_id)

    os.rename(os.path.join(repo, "before.py"), os.path.join(repo, "after.py"))
    settle(indexer)

    present = files_in(store, indexer.repo_id)
    assert "before.py" not in present
    assert "after.py" in present
    assert "moves_around" in symbols_in(store, indexer.repo_id)


def test_debouncing_coalesces_a_burst_of_saves(store, watched, embed_calls):
    """One editor save emits several events; they must cause one index pass."""
    repo, indexer = watched
    embed_calls.clear()

    for i in range(8):
        write(repo, "burst.py", f"def burst():\n    return {i}\n")
        time.sleep(0.02)

    settle(indexer)

    assert len(embed_calls) == 1, f"expected 1 coalesced embed, got {len(embed_calls)}"
    assert "return 7" in embed_calls[0], "the final content is what should be indexed"


def test_files_in_ignored_directories_are_not_indexed(store, watched):
    repo, indexer = watched
    os.makedirs(os.path.join(repo, "node_modules"), exist_ok=True)
    write(repo, "node_modules/junk.py", "def junk():\n    return 0\n")
    settle(indexer)

    assert "junk" not in symbols_in(store, indexer.repo_id)


def test_non_code_files_the_walker_indexes_are_watched_too(store, watched):
    """walk_repo() indexes .md/.toml/.yaml; the watcher must keep them current."""
    repo, indexer = watched
    write(repo, "NOTES.md", "# Deployment notes\n\nRotate the signing key monthly.\n")
    settle(indexer)

    assert "NOTES.md" in files_in(store, indexer.repo_id)


def test_non_source_files_are_ignored_by_the_watcher(store, watched, embed_calls):
    repo, indexer = watched
    embed_calls.clear()
    write(repo, "notes.log", "nothing to see")
    settle(indexer)

    assert embed_calls == []


def test_stop_watchdog_tears_everything_down(store, make_repo, fake_embedder):
    repo = make_repo({"a.py": "def a():\n    return 1\n"})
    indexer = Indexer(repo)
    indexer.start_watchdog(debounce_seconds=DEBOUNCE)
    assert indexer.observer is not None
    assert indexer.worker is not None

    indexer.stop_watchdog()
    assert indexer.observer is None
    assert indexer.worker is None


def test_watcher_works_for_a_repo_inside_a_directory_named_build(store, make_repo, fake_embedder):
    """Ignored names used to be matched against the whole absolute path."""
    repo = make_repo({"seed.py": "def seed():\n    return 1\n"}, name="build/myproject")
    indexer = Indexer(repo)
    indexer.start_watchdog(debounce_seconds=DEBOUNCE)
    try:
        write(repo, "new_file.py", "def brand_new():\n    return 42\n")
        settle(indexer)
        assert "new_file.py" in files_in(store, indexer.repo_id)
    finally:
        indexer.stop_watchdog()


def test_ignored_directories_inside_the_repo_are_still_ignored(store, make_repo, fake_embedder):
    from backend.indexer import RepoEventHandler

    repo = make_repo({"seed.py": "x = 1\n"}, name="plain")
    handler = RepoEventHandler(Indexer(repo), worker=None)
    assert handler._is_indexable(os.path.join(repo, "src", "a.py"))
    assert not handler._is_indexable(os.path.join(repo, "build", "a.py"))
    assert not handler._is_indexable(os.path.join(repo, "src", "node_modules", "a.py"))
    assert not handler._is_indexable(os.path.join(os.path.dirname(repo), "elsewhere.py"))


def test_a_file_gone_before_the_debounce_fires_leaves_no_row(store, make_repo, fake_embedder):
    repo = make_repo({"seed.py": "def seed():\n    return 1\n"})
    indexer = Indexer(repo)
    indexer.reindex_file(os.path.join(repo, "never_existed.py"))
    assert "never_existed.py" not in files_in(store, indexer.repo_id)
