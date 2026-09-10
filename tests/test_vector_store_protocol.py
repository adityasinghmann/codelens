"""The storage interface, and that the SQLite implementation satisfies it."""

import inspect

import pytest

from backend.db_client import LocalVectorStore, get_db
from backend.vector_store import VectorStore


def protocol_methods():
    return sorted(
        name for name, _ in inspect.getmembers(VectorStore, predicate=inspect.isfunction)
        if not name.startswith("_")
    )


def test_local_store_implements_every_protocol_method():
    missing = [name for name in protocol_methods() if not hasattr(LocalVectorStore, name)]
    assert missing == []


def test_local_store_satisfies_the_protocol_at_runtime(store):
    assert isinstance(store, VectorStore)


def test_protocol_covers_the_documented_surface():
    """The interface must actually name the operations the brief calls for."""
    names = set(protocol_methods())
    for required in ("upsert_chunks", "delete_chunks", "delete_file",
                     "delete_repository", "search", "get_chunks_for_file",
                     "set_index_metadata", "get_index_metadata"):
        assert required in names, required


@pytest.mark.parametrize("name", protocol_methods())
def test_signatures_match_the_protocol(name):
    """
    A method whose signature drifts from the interface silently breaks any
    substitute implementation, so compare parameter names directly.
    """
    expected = inspect.signature(getattr(VectorStore, name))
    actual = inspect.signature(getattr(LocalVectorStore, name))
    assert list(actual.parameters) == list(expected.parameters), name


def test_a_fake_store_can_stand_in_for_the_real_one():
    """
    The point of the Protocol: something with no database can satisfy it.
    This is a test double, not a second backend.
    """

    class InMemoryStore:
        def __init__(self):
            self.chunks = {}

        def ensure_repository(self, root_path): return "repo"
        def get_repository(self, repo_id): return {"repo_id": repo_id}
        def list_repositories(self): return []
        def most_recent_repository(self): return None
        def touch_repository(self, repo_id): pass
        def delete_repository(self, repo_id): self.chunks.clear()
        def clear_repository_content(self, repo_id): self.chunks.clear()
        def upsert_file(self, repo_id, rel_path, content_hash, language=""): return "file"
        def get_file(self, repo_id, rel_path): return None
        def list_files(self, repo_id): return []
        def delete_file(self, repo_id, rel_path): pass
        def get_chunks_for_file(self, file_id): return {}
        def upsert_chunks(self, repo_id, file_id, chunks, embeddings):
            for chunk in chunks:
                self.chunks[chunk["chunk_id"]] = chunk
        def update_chunk_lines(self, updates): pass
        def delete_chunks(self, chunk_ids): pass
        def set_index_metadata(self, repo_id, embedding_model=None,
                               embedding_dimension=None, status=None): pass
        def get_index_metadata(self, repo_id): return None
        def search(self, embedding, repo_id, top_k=10): return []
        def count(self, repo_id=None): return len(self.chunks)
        def count_files(self, repo_id=None): return 0

    fake = InMemoryStore()
    assert isinstance(fake, VectorStore)

    fake.upsert_chunks("r", "f", [{"chunk_id": "c1"}], [[0.0]])
    assert fake.count() == 1


def test_get_db_returns_a_conforming_store(store):
    assert isinstance(get_db(), VectorStore)
