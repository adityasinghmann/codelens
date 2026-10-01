"""Embedding failure handling, dimension derivation and model compatibility."""

import pytest

from backend.config import Settings
from backend.indexer import EmbeddingDimensionMismatch, Indexer
from backend.query import EmbeddingModelMismatch, run_query
from tests.conftest import FAKE_DIM

THREE_FUNCS = '''def one():
    return 1


def two():
    return 2


def three():
    return 3
'''


async def test_successful_embedding_stores_every_chunk(store, make_repo, index_repo):
    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    counts = await index_repo(indexer, repo)

    assert counts["stored"] == 3
    assert counts["failed"] == 0
    assert store.count(indexer.repo_id) == 3


async def test_ollama_unavailable_stores_nothing_and_counts_every_failure(
    store, make_repo, index_repo, fake_embedder
):
    """A zero vector must never be persisted in place of a real embedding."""
    fake_embedder.fail_all = True

    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    counts = await index_repo(indexer, repo)

    assert counts["stored"] == 0
    assert counts["failed"] == 3
    assert store.count(indexer.repo_id) == 0

    with store._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


async def test_chunks_that_failed_are_retried_on_the_next_pass(
    store, make_repo, index_repo, fake_embedder
):
    """Because nothing was written, nothing looks current on the retry."""
    fake_embedder.fail_all = True
    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    fake_embedder.fail_all = False
    counts = await index_repo(indexer, repo)

    assert counts["stored"] == 3
    assert counts["failed"] == 0
    assert store.count(indexer.repo_id) == 3


async def test_partial_failure_keeps_the_good_chunks(
    store, make_repo, index_repo, fake_embedder
):
    fake_embedder.fail_when = lambda text: "two" in text

    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    counts = await index_repo(indexer, repo)

    assert counts["stored"] == 2
    assert counts["failed"] == 1

    with store._conn() as conn:
        names = sorted(r[0] for r in conn.execute(
            "SELECT symbol_name FROM chunks WHERE repo_id = ?", (indexer.repo_id,)).fetchall())
    assert names == ["one", "three"]


async def test_dimension_is_derived_from_the_model_not_hardcoded(
    store, make_repo, index_repo
):
    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    meta = store.get_index_metadata(indexer.repo_id)
    assert meta["embedding_dimension"] == FAKE_DIM
    assert meta["embedding_dimension"] != 768, "768 must not be assumed"
    assert meta["embedding_model"] == Settings.EMBED_MODEL


async def test_a_changed_vector_width_raises_instead_of_being_written(
    store, make_repo, index_repo, fake_embedder
):
    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    # The model changed underneath the index.
    fake_embedder.dim = FAKE_DIM + 16
    with open(f"{repo}/m.py", "w", encoding="utf-8") as handle:
        handle.write("def one():\n    return 111\n")

    with pytest.raises(EmbeddingDimensionMismatch) as excinfo:
        await index_repo(indexer, repo)

    message = str(excinfo.value)
    assert str(FAKE_DIM) in message
    assert str(FAKE_DIM + 16) in message
    assert "Re-index" in message


async def test_query_refuses_an_index_built_with_another_model(
    store, make_repo, index_repo
):
    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    store.set_index_metadata(indexer.repo_id, embedding_model="some-other-model")

    with pytest.raises(EmbeddingModelMismatch) as excinfo:
        await run_query("anything", repo_id=indexer.repo_id)

    message = str(excinfo.value)
    assert "some-other-model" in message
    assert Settings.EMBED_MODEL in message
    assert "Re-index" in message


async def test_query_succeeds_when_the_model_matches(
    store, make_repo, index_repo, monkeypatch
):
    from backend import query as query_module
    from tests.conftest import fake_vector

    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    class FakeResponse:
        embedding = fake_vector("search_query: two")

    class FakeClient:
        async def embeddings(self, model, prompt):
            return FakeResponse()

    monkeypatch.setattr(query_module, "_get_ollama_client", lambda: FakeClient())

    result = await run_query("two", repo_id=indexer.repo_id, top_k=3)
    assert result["results"]
    assert all(0.0 <= hit["score"] <= 1.0 for hit in result["results"])


async def test_scores_are_clamped_into_zero_to_one(store, make_repo, index_repo, monkeypatch):
    from backend import query as query_module
    from tests.conftest import fake_vector

    repo = make_repo({"m.py": THREE_FUNCS})
    indexer = Indexer(repo)
    await index_repo(indexer, repo)

    class FakeResponse:
        # Deliberately anti-correlated, to drive raw cosine negative.
        embedding = [-v for v in fake_vector("search_document: def one():\n    return 1\n")]

    class FakeClient:
        async def embeddings(self, model, prompt):
            return FakeResponse()

    monkeypatch.setattr(query_module, "_get_ollama_client", lambda: FakeClient())

    result = await run_query("anything", repo_id=indexer.repo_id, top_k=3)
    assert all(hit["score"] >= 0.0 for hit in result["results"])


async def test_explain_model_comes_from_config():
    assert Settings.EXPLAIN_MODEL
    assert hasattr(Settings, "EMBED_MODEL")
    assert hasattr(Settings, "OLLAMA_HOST")


# --- Real HTTP across event loops -----------------------------------------

@pytest.fixture
def local_ollama(monkeypatch):
    """
    A real keep-alive HTTP server imitating Ollama's embeddings endpoint.

    The event-loop bug lives in the real HTTP client's connection pool, so it
    cannot be reproduced by monkeypatching _embed_text like the rest of this
    suite does.
    """
    import socket
    import threading
    import time
    import uvicorn
    from fastapi import FastAPI

    fake = FastAPI()

    @fake.post("/api/embeddings")
    async def embeddings(body: dict):
        return {"embedding": [0.5] * FAKE_DIM}

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "fake Ollama did not start"

    monkeypatch.setattr(Settings, "OLLAMA_HOST", f"http://127.0.0.1:{port}")
    yield
    server.should_exit = True
    thread.join(timeout=5)


def test_embedding_works_on_a_new_loop_after_the_first_is_closed(store, make_repo, local_ollama):
    """
    POST /index embeds on a private loop and closes it; the watcher then
    embeds on its own loop with the same Indexer. Sharing one HTTP client
    across them failed with "Event loop is closed".
    """
    import asyncio

    indexer = Indexer(make_repo({"a.py": "x = 1\n"}))

    first = asyncio.new_event_loop()
    try:
        assert first.run_until_complete(indexer._embed_text("first")) is not None
    finally:
        first.close()

    second = asyncio.new_event_loop()
    try:
        assert second.run_until_complete(indexer._embed_text("second")) is not None
    finally:
        second.close()
