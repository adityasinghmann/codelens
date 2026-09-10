"""
Shared fixtures.

Two invariants hold for the whole suite:

  1. No test touches the network or requires a running Ollama. Embedding is
     always monkeypatched with a deterministic stand-in.
  2. No test touches a real index directory. Every store is created under
     pytest's tmp_path and the module-level singleton is reset around it.
"""

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db_client
from backend.config import Settings
from backend.db_client import LocalVectorStore


#: Width of the fake embedding vectors. Deliberately not 768, so a test that
#: accidentally depends on the old hardcoded dimension fails loudly.
FAKE_DIM = 32


def fake_vector(text: str, dim: int = FAKE_DIM) -> list[float]:
    """A deterministic vector for a string. Same text -> same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * ((dim // len(digest)) + 1))[:dim]
    return [b / 255.0 for b in raw]


@pytest.fixture
def index_dir(tmp_path):
    """An isolated index directory for one test."""
    path = tmp_path / "index"
    path.mkdir()
    return str(path)


@pytest.fixture
def store(index_dir, monkeypatch):
    """A LocalVectorStore on a temp directory, installed as the singleton."""
    monkeypatch.setattr(Settings, "INDEX_PATH", index_dir)
    instance = LocalVectorStore(path=index_dir)
    monkeypatch.setattr(db_client, "_db_instance", instance)
    yield instance
    monkeypatch.setattr(db_client, "_db_instance", None)


@pytest.fixture
def embed_calls():
    """Records every text passed to the embedder, so tests can count calls."""
    return []


@pytest.fixture
def fake_embedder(monkeypatch, embed_calls):
    """
    Replace Indexer._embed_text with a deterministic local function.

    Returns a small control object so a test can make embedding fail, change
    the vector width, or inspect what was embedded.
    """
    from backend import indexer as indexer_module

    class Control:
        dim = FAKE_DIM
        #: Predicate: return True for a chunk text that should fail to embed.
        fail_when = staticmethod(lambda text: False)
        fail_all = False

    control = Control()

    async def _embed(self, text):
        embed_calls.append(text)
        if control.fail_all or control.fail_when(text):
            return None
        return fake_vector(text, control.dim)

    monkeypatch.setattr(indexer_module.Indexer, "_embed_text", _embed)
    return control


@pytest.fixture
def make_repo(tmp_path):
    """Create a repository directory from a {relative path: contents} mapping."""
    counter = {"n": 0}

    def _make(files: dict[str, str], name: str | None = None) -> str:
        counter["n"] += 1
        root = tmp_path / (name or f"repo{counter['n']}")
        for rel, body in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        root.mkdir(exist_ok=True)
        return str(root)

    return _make


@pytest.fixture
def index_repo(fake_embedder):
    """Index every file in a repository, returning the accumulated counts."""

    async def _index(indexer, repo_path):
        from backend.indexer import chunk_file

        totals = {"stored": 0, "skipped": 0, "deleted": 0, "failed": 0}
        for path in indexer.walk_repo():
            counts = await indexer.index_file(chunk_file(path, repo_path, indexer.repo_id))
            for key in totals:
                totals[key] += counts.get(key, 0)
        return totals

    return _index


# Sample sources reused across the parser and indexer tests. ----------------

PY_CLASS = '''"""Module docstring."""


class Greeter:
    """Greets people."""

    prefix = "Hello"

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"{self.prefix}, {self.name}"

    def shout(self):
        return self.greet().upper()
'''

PY_DECORATED = '''import functools


@functools.cache
def cached_lookup(key):
    """Look something up."""
    return key * 2


@app.route("/thing")
@auth_required
def handler(request):
    return None
'''

JAVA_CLASS = '''public class Service {
    private int count;

    public Service(int c) {
        this.count = c;
    }

    public int getCount() {
        return count;
    }

    public void reset() {
        count = 0;
    }
}
'''

TS_MIXED = '''export interface Opts {
    a: number;
}

export class Widget {
    private id: string;

    constructor(id: string) {
        this.id = id;
    }

    render(): string {
        return this.id;
    }
}

const helper = (x: number) => x * 2;

export function topLevel(y: number): number {
    return y;
}
'''

PY_NESTED = '''class Outer:
    class Inner:
        def deep(self):
            return 1

    def shallow(self):
        return 2
'''
