"""
HTTP surface: status codes, the SSE contract, and the concurrency guard.

The app's lifespan is skipped where it would start a watcher; tests that need
the index populated drive the endpoints directly.
"""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from tests.conftest import fake_vector

SAMPLE = '''def alpha_one(x):
    """First function."""
    return x


def alpha_two(y):
    """Second function."""
    return y
'''


@pytest.fixture
def client(store, fake_embedder, monkeypatch):
    """A TestClient with Ollama stubbed out at every call site."""

    async def fake_probe(timeout: float = 1.5):
        return True, None

    monkeypatch.setattr(main_module, "probe_ollama", fake_probe)
    monkeypatch.setattr(main_module, "global_indexer", None)
    monkeypatch.setattr(main_module, "_active_indexing", set())

    # Never start a real watchdog from an API-driven index.
    from backend.indexer import Indexer
    monkeypatch.setattr(Indexer, "start_watchdog", lambda self, *a, **k: None)
    monkeypatch.setattr(Indexer, "stop_watchdog", lambda self, *a, **k: None)

    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def stub_query_embedding(monkeypatch):
    """Make /query embed deterministically without Ollama."""
    from backend import query as query_module

    def _install(text_for_vector: str):
        class FakeResponse:
            embedding = fake_vector(text_for_vector)

        class FakeClient:
            async def embeddings(self, model, prompt):
                return FakeResponse()

        monkeypatch.setattr(query_module, "_get_ollama_client", lambda: FakeClient())

    return _install


def sse_events(response):
    """Collect the JSON payloads from an SSE response body."""
    events = []
    for block in response.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[len("data: "):]))
    return events


# --- health / status ------------------------------------------------------

def test_health_reports_both_dependencies(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ollama"] is True
    assert "index" in body
    assert "vectorai" not in body, "the old field name must be gone"


def test_status_with_nothing_indexed(client):
    body = client.get("/status").json()
    assert body["indexed_chunks"] == 0
    assert body["repo_path"] is None
    assert body["watching"] is False
    assert body["embed_model"]


def test_status_reports_the_indexed_repository(client, make_repo):
    repo = make_repo({"m.py": SAMPLE})
    client.post("/index", json={"repo_path": repo})

    body = client.get("/status").json()
    assert body["indexed_chunks"] == 2
    assert body["repo_path"]
    assert body["last_indexed"]


# --- index ----------------------------------------------------------------

def test_index_streams_progress_then_complete(client, make_repo):
    repo = make_repo({"m.py": SAMPLE})
    response = client.post("/index", json={"repo_path": repo})
    assert response.status_code == 200

    events = sse_events(response)
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "complete"
    assert "progress" in kinds

    progress = [e for e in events if e["type"] == "progress"][-1]
    assert progress["processed_files"] == progress["total_files"]
    assert progress["processed_chunks"] == 2

    done = events[-1]
    assert done["stored"] == 2
    assert done["skipped"] == 0
    assert done["failed"] == 0
    assert done["total_chunks"] == 2
    assert "duration_ms" in done


def test_progress_units_are_comparable(client, make_repo):
    """processed_files / total_files must never exceed 1."""
    repo = make_repo({f"m{i}.py": SAMPLE for i in range(4)})
    response = client.post("/index", json={"repo_path": repo})

    for event in sse_events(response):
        if event["type"] == "progress":
            assert 0 < event["processed_files"] <= event["total_files"]


def test_reindexing_unchanged_reports_everything_skipped(client, make_repo):
    repo = make_repo({"m.py": SAMPLE})
    client.post("/index", json={"repo_path": repo})
    response = client.post("/index", json={"repo_path": repo})

    done = sse_events(response)[-1]
    assert done["stored"] == 0
    assert done["skipped"] == 2


def test_force_reindex_rebuilds_everything(client, make_repo):
    repo = make_repo({"m.py": SAMPLE})
    client.post("/index", json={"repo_path": repo})
    response = client.post("/index", json={"repo_path": repo, "force_reindex": True})

    done = sse_events(response)[-1]
    assert done["stored"] == 2
    assert done["skipped"] == 0


def test_index_rejects_a_nonexistent_path(client, tmp_path):
    response = client.post("/index", json={"repo_path": str(tmp_path / "nope")})
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_index_rejects_a_file(client, make_repo, tmp_path):
    target = tmp_path / "a_file.txt"
    target.write_text("x", encoding="utf-8")
    response = client.post("/index", json={"repo_path": str(target)})
    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"]


def test_index_rejects_an_empty_path(client):
    response = client.post("/index", json={"repo_path": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_concurrent_index_of_the_same_repository_returns_409(
    client, make_repo, fake_embedder, monkeypatch
):
    """
    Two overlapping runs used to race on the module-level global_indexer.
    The second must be refused, not interleaved.
    """
    import asyncio
    from backend.indexer import Indexer
    from tests.conftest import fake_vector as _vec

    started = threading.Event()
    release = threading.Event()

    async def slow_embed(self, text):
        started.set()
        await asyncio.get_running_loop().run_in_executor(None, release.wait, 10)
        return _vec(text)

    monkeypatch.setattr(Indexer, "_embed_text", slow_embed)

    repo = make_repo({"m.py": SAMPLE})
    outcome = {}

    def first():
        response = client.post("/index", json={"repo_path": repo})
        outcome["first"] = response.status_code

    worker = threading.Thread(target=first)
    worker.start()
    try:
        assert started.wait(timeout=10), "the first index never began embedding"
        second = client.post("/index", json={"repo_path": repo})
        assert second.status_code == 409
        assert "already running" in second.json()["detail"]
    finally:
        release.set()
        worker.join(timeout=20)

    assert outcome.get("first") == 200

    # Once it finishes, the guard is released.
    again = client.post("/index", json={"repo_path": repo})
    assert again.status_code == 200


# --- query ----------------------------------------------------------------

def test_query_returns_scoped_results(client, make_repo, stub_query_embedding):
    repo = make_repo({"m.py": SAMPLE})
    client.post("/index", json={"repo_path": repo})
    stub_query_embedding("search_document: def alpha_one(x):\n    \"\"\"First function.\"\"\"\n    return x")

    response = client.post("/query", json={"query": "first function", "repo_path": repo})
    assert response.status_code == 200
    body = response.json()

    assert body["results"]
    assert body["total_indexed"] == 2
    assert isinstance(body["query_ms"], int)

    hit = body["results"][0]
    for key in ("symbol_name", "qualified_name", "symbol_type", "file_path",
                "start_line", "end_line", "language", "chunk_text", "score"):
        assert key in hit, key
    assert 0.0 <= hit["score"] <= 1.0


def test_query_rejects_a_whitespace_only_query(client):
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_query_rejects_an_unindexed_repository(client, make_repo):
    repo = make_repo({"m.py": SAMPLE})
    response = client.post("/query", json={"query": "anything", "repo_path": repo})
    assert response.status_code == 400
    assert "not been indexed" in response.json()["detail"]


def test_query_without_any_indexed_repository_is_rejected(client):
    response = client.post("/query", json={"query": "anything"})
    assert response.status_code == 400


def test_query_enforces_the_top_k_range(client, make_repo):
    repo = make_repo({"m.py": SAMPLE})
    client.post("/index", json={"repo_path": repo})
    assert client.post("/query", json={"query": "x", "top_k": 0, "repo_path": repo}).status_code == 422
    assert client.post("/query", json={"query": "x", "top_k": 99, "repo_path": repo}).status_code == 422


def test_query_isolates_repositories(client, make_repo, stub_query_embedding):
    repo_a = make_repo({"a.py": "def only_alpha():\n    return 1\n"}, name="a")
    repo_b = make_repo({"b.py": "def only_beta():\n    return 2\n"}, name="b")
    client.post("/index", json={"repo_path": repo_a})
    client.post("/index", json={"repo_path": repo_b})

    stub_query_embedding("search_document: def only_beta():\n    return 2\n")

    from_a = client.post("/query", json={"query": "beta", "repo_path": repo_a}).json()
    from_b = client.post("/query", json={"query": "beta", "repo_path": repo_b}).json()

    assert [r["file_path"] for r in from_a["results"]] == ["a.py"]
    assert [r["file_path"] for r in from_b["results"]] == ["b.py"]


def test_query_refuses_a_model_mismatch_with_409(client, make_repo, store):
    from backend.db_client import repo_id_for

    repo = make_repo({"m.py": SAMPLE})
    client.post("/index", json={"repo_path": repo})
    store.set_index_metadata(repo_id_for(repo), embedding_model="a-different-model")

    response = client.post("/query", json={"query": "anything", "repo_path": repo})
    assert response.status_code == 409
    assert "Re-index" in response.json()["detail"]


# --- CORS -----------------------------------------------------------------

def test_cors_allows_the_vscode_webview_origin(client):
    response = client.get("/health", headers={"Origin": "vscode-webview://abc-123"})
    assert response.headers.get("access-control-allow-origin") == "vscode-webview://abc-123"


def test_cors_does_not_allow_an_arbitrary_web_origin(client):
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in response.headers


def test_requests_without_an_origin_still_work(client):
    assert client.get("/health").status_code == 200
