<div align="center">

# 🔍 CodeLens

### *Natural-language search over your codebase — fully offline*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![VS Code](https://img.shields.io/badge/VS%20Code-Extension-007ACC?style=flat-square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com)
[![Ollama](https://img.shields.io/badge/Ollama-nomic--embed--text-FFA500?style=flat-square)](https://ollama.com)


**Ask a question in plain English. Get the functions that answer it. Jump to the line.**
No internet, no API keys, no data leaving the machine.

[Setup](#setup) · [Architecture](#architecture) · [API](#api-contract) · [Limitations](#limitations) · [Tests](#running-the-tests)

</div>

---

## What CodeLens is

CodeLens is a VS Code extension plus a local FastAPI backend that does semantic
search over a codebase. It parses source into whole functions, classes and
methods with tree-sitter, embeds each one with a local Ollama model, stores the
vectors in a local SQLite file, and ranks them against your query with exact
cosine similarity.

Keyword search matches *text*. CodeLens matches *intent*: `"where do user
sessions expire?"` can surface `def invalidate_jwt_token()` even though the
words do not overlap.

Everything runs on your machine. The only network calls are to
`localhost:11434` (Ollama) and `127.0.0.1:8000` (the backend, which binds
loopback only).

> **Scope.** This is a working single-developer tool, not a product. The
> [Limitations](#limitations) section is not boilerplate — read it before
> assuming a capability.

---

## Architecture

```
┌──────────────────────────── your machine, fully offline ────────────────────────────┐
│                                                                                      │
│   VS Code extension (TypeScript)                                                     │
│   ┌────────────────────────────────┐                                                 │
│   │ sidebar webview (panel.html)   │  query / reindex / jump-to-file                 │
│   │ backendProcess.ts              │  spawn + /health polling + restart              │
│   │ pythonEnv.ts, toolResolver.ts  │  interpreter + Ollama discovery                 │
│   │ sseParser.ts                   │  incremental Server-Sent Events                 │
│   └───────────────┬────────────────┘                                                 │
│                   │ HTTP  127.0.0.1:8000                                             │
│                   ▼                                                                  │
│   FastAPI backend (backend/main.py)                                                  │
│   ┌──────────────────────────┬───────────────────────────┐                           │
│   │  index service           │  search service           │                           │
│   │  (indexer.py)            │  (query.py)               │                           │
│   └───────────┬──────────────┴─────────────┬─────────────┘                           │
│               │                            │                                         │
│       ┌───────▼────────┐          ┌────────▼─────────┐                               │
│       │ tree-sitter    │          │ embedding service│                               │
│       │ chunker        │          │ (Ollama client)  │                               │
│       │ parser.py      │          │ bounded gather   │                               │
│       └───────┬────────┘          └────────┬─────────┘                               │
│               │        chunks + vectors    │                                         │
│               └─────────────┬──────────────┘                                         │
│                             ▼                                                        │
│              VectorStore  (backend/vector_store.py — Protocol)                       │
│                             │                                                        │
│                             ▼                                                        │
│              LocalVectorStore → SQLite  (.codelens_index/codelens.db)                │
│                                                                                      │
│                             ▲  HTTP                                                  │
│                             └──── Ollama (localhost:11434) ─── nomic-embed-text      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### How the data is organised

```
repositories (repo_id PK, root_path UNIQUE, created_at, updated_at)
     │ 1
     │
     │ N
files (file_id PK, repo_id FK, path, content_hash, language, indexed_at)
     │ 1
     │
     │ N
chunks (chunk_id PK, file_id FK, repo_id FK,
        symbol_name, qualified_name, symbol_type, parent_symbol,
        file_path, start_line, end_line, language,
        chunk_text, content_hash, embedding BLOB)

index_metadata (repo_id FK, embedding_model, embedding_dimension,
                schema_version, status, updated_at)
```

Every chunk carries `repo_id`, and `search()` requires one — there is no
unscoped search.

---

## Design decisions

### Why tree-sitter, not fixed-size splits

Splitting code every N lines cuts functions in half and glues unrelated
fragments together, which produces chunks that mean nothing on their own.
CodeLens parses the real AST and extracts complete semantic units.

Node types are split into two kinds, which matters more than it sounds:

- **Leaves** — functions, methods, constructors, arrow functions, type aliases.
  Emitted whole; the walk stops there, because the body belongs to the symbol.
- **Containers** — classes, interfaces, `impl` blocks, structs, traits, enums.
  Emitted *header-only*: the declaration, docstring and field declarations,
  with the bodies of nested methods and nested classes removed (each is its
  own chunk). The walk continues inside. A decorated class, such as a
  `@dataclass`, is a container like any other.
- **Module-level code** — anything no symbol covers: module-level assignments
  and config, handlers passed inline to a call such as `app.post(...)`, Go
  `var` blocks. Each contiguous run becomes a `code` chunk (windowed if long),
  so a file with one function does not lose everything around it. Runs made
  only of imports or closing brackets are skipped.

Go types are one chunk each (`type ( A struct{}; B int )` gives `A` and `B`),
typed `struct`, `interface` or `type`. A Rust `impl Trait for Type` is named and
scopes its methods by `Type`.

Without that split, a class is emitted once containing all its methods and then
again as one chunk per method — the same source embedded twice, competing with
itself in results. On this repository the split removes about 32% of the chunk
text that would otherwise be embedded.

Files with no recognised symbols, unsupported extensions, and any parse failure
fall back to a 40-line sliding window with 10 lines of overlap. A malformed file
never raises.

### Why local embeddings

`nomic-embed-text` via Ollama runs on the machine, so source code never leaves
it and there are no API keys or rate limits. The trade is that embedding is a
local HTTP round-trip per chunk, which makes the first index of a repository
the slow part.

### Why SQLite and exact cosine similarity

The index is one file. No server, no daemon, no container — copy it, back it
up, delete it. SQLite ships with Python, so the store adds zero operational
surface.

Search is **exhaustive and exact**: every embedding for the repository is read
into a numpy matrix and scored against the query with one matrix-vector
product, then the top *k* are returned. There is **no ANN index** — no HNSW, no
IVF, no quantisation. Cost grows linearly with the number of chunks. That is
the honest trade, and an approximate index is the first thing that would need
to change to scale much further.

Two things `search()` does to avoid needless work: it scores against
`(chunk_id, embedding)` and hydrates only the winning rows, and it uses
`np.argpartition` rather than fully sorting N scores to keep 8. Measured over
20,000 chunks at 768 dimensions that is 265.6 ms → 224.2 ms per query, with
identical results.

The store sits behind a `VectorStore` Protocol
([backend/vector_store.py](backend/vector_store.py)) so the indexer and query
layer depend on a named interface rather than on SQLite. `LocalVectorStore` is
the only implementation. **No second backend exists.**

### Incremental indexing

Chunk identity is separated from change detection:

```
chunk_id     = hash(repo_id, file_path, symbol_type, qualified_name[, ordinal])
content_hash = hash(chunk_text)
```

`chunk_id` deliberately **excludes line numbers**. Adding an import at the top
of a file shifts every symbol below it; line-based identity would change every
id and force a full re-embed. Lines are mutable metadata, updated in place.

Re-indexing a file diffs against what is stored:

| Case | Action |
|---|---|
| same `chunk_id`, same `content_hash` | keep the embedding; update line numbers only |
| same `chunk_id`, different `content_hash` | re-embed, update in place |
| new `chunk_id` | embed and insert |
| stored `chunk_id` no longer present | delete |

So `{A:h1, B:h2, C:h3}` → `{A:h1, B:h99, D:h4}` costs exactly **two**
embeddings — B and D. A is untouched, C is deleted. Re-indexing an unchanged
file costs **zero** embeddings.

### Repository isolation

`repo_id` derives from the resolved, `normcase`-normalised root path, so
`C:\Repo` and `c:\repo` are one repository. Two repositories indexed by the
same backend never see each other's chunks. Deleting or force-rebuilding one
never touches another.

### Watcher behaviour

A `watchdog` observer handles **create, modify, delete and move** — a move is a
removal of the old path plus an index of the new one. It covers every file type
the full index walks, including `.md`, `.txt`, `.toml` and `.yaml`. Each
repository indexed during a backend session gets its own watcher, so every
folder of a multi-root workspace stays current. Events are debounced per
path (500 ms) so the burst an editor emits for one save causes one index pass,
and all work is serialised onto a single long-lived background loop, so two
operations for the same file can never overlap.

### Embedding robustness

A chunk that fails to embed returns `None` and is **dropped, never stored**, and
counted as a failure. It is retried on the next pass precisely because no row
exists to make it look current. The embedding dimension is derived from the
first successful vector and recorded; a different width raises rather than
writing a wrong-size blob. Querying an index built with a different embedding
model is refused with a clear message, because vectors from two models are not
comparable.

---

## Setup

### Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com)** running locally, with `nomic-embed-text` pulled
- **VS Code 1.86+** (for the extension)
- Node 18+ (only to build the extension from source)

### Using the extension

The backend ships **inside** the `.vsix`. On first activation the extension
creates a private virtual environment under its own global-storage directory
and installs `requirements.txt` into it. That environment is keyed by a hash of
the requirements plus the interpreter version, so it is rebuilt when either
changes, and removed when the extension is uninstalled.

```bash
ollama pull nomic-embed-text     # one time, ~274 MB

npm install
npm run compile
npx vsce package                 # produces codelens-<version>.vsix
code --install-extension codelens-0.2.0.vsix
```

Open a folder, then run **CodeLens: Re-index Workspace**.

If Python cannot be found, the venv cannot be created, pip fails, or the
backend does not answer `/health` in time, you get a specific error naming the
fix — never a silent hang.

### Running the backend directly

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`setup.sh` and `start.sh` do the same for macOS and Linux. On Windows, use the
extension, which provisions its own environment.

### Running with Docker

```bash
CODELENS_REPOS_DIR=/path/to/your/code docker compose up --build
curl -X POST http://127.0.0.1:8000/index -H 'Content-Type: application/json' \
     -d '{"repo_path": "/repos"}'
```

This starts the backend and an Ollama container, which pulls
`nomic-embed-text` on first start. Both ports are published on the host's
loopback only. `CODELENS_REPOS_DIR` (default: this directory) is mounted
read-only at `/repos`, and repositories are indexed and queried by that
container path, so this mode is for driving the API directly — the VS Code
extension sends host paths and runs its own backend. On Docker Desktop, file
change events from the host may not reach the container, so re-index after
editing.

### Configuration

VS Code settings:

| Setting | Default | Purpose |
|---|---|---|
| `codelens.pythonPath` | `""` | Interpreter to build the venv from. Empty = search `PATH`, then platform defaults. |
| `codelens.ollamaPath` | `""` | Path to the `ollama` executable, used to put it on the backend's `PATH`. |
| `codelens.ollamaHost` | `http://localhost:11434` | Ollama server address. |
| `codelens.port` | `8000` | Loopback port for the backend. |
| `codelens.startupTimeoutSeconds` | `30` | How long to wait for `/health` before reporting failure. |

Backend environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `CODELENS_INDEX_PATH` | `./.codelens_index` | Index directory. (`VECTORAI_DB_PATH` still honoured.) The extension sets it to `index/` in its VS Code global storage, so the index survives extension updates. |
| `CODELENS_ALLOWED_HOSTS` | `127.0.0.1,localhost` | `Host` header values the API answers to. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama address. |
| `EMBED_MODEL` | `nomic-embed-text` | Model for chunks *and* queries. |
| `EXPLAIN_MODEL` | `mistral` | Model for optional explain summaries. |
| `EMBED_CONCURRENCY` | `4` | Concurrent embedding requests, bounded. |
| `EMBED_BATCH_SIZE` | `20` | Chunks per batch. |
| `TOP_K` | `10` | Result count for a `/query` that omits `top_k`, clamped to 1–20. The sidebar always asks for 8. |

### Supported languages

AST chunking: **Python, TypeScript, TSX, JavaScript, JSX, Go, Rust, Java**, and
**C/C++** if `tree-sitter-cpp` is installed. Everything else the walker picks up
(`.md`, `.txt`, `.toml`, `.yaml`, `.yml`) uses the sliding-window fallback.

> `tree-sitter` is pinned to `>=0.25,<0.26`. Version 0.26.0 against the current
> grammar wheels corrupts memory and segfaults after a few hundred parses.

---

## API contract

Defined once, as the Pydantic models in [backend/main.py](backend/main.py) —
those models are what FastAPI validates against at runtime, so they cannot
drift from the server's behaviour.
[extension/src/apiTypes.ts](extension/src/apiTypes.ts) is a hand-maintained
TypeScript mirror.

### `POST /index`

```json
{"repo_path": "/abs/path/to/repo", "force_reindex": false}
```

Responds with `text/event-stream`, each event `data: <json>\n\n`:

```json
{"type":"progress","file":"src/auth.py","processed_files":12,"total_files":240,"processed_chunks":318}
{"type":"error","file":"src/broken.py","message":"..."}
{"type":"complete","total_files":240,"processed_files":240,"total_chunks":6104,
 "stored":83,"skipped":6021,"failed":0,"removed_files":0,"duration_ms":41200}
```

`processed_files` and `total_files` share a unit, so their ratio is real
progress; a file that fails to parse still advances it, so the stream always
reaches 100%. `stored`, `skipped` and `failed` are actual counts from the diff;
what is searchable afterwards is `stored + skipped`, not `total_chunks`.
`removed_files` counts stored files no longer on disk, which every run prunes.

- `400` — `repo_path` missing, nonexistent, or not a directory
- `409` — an index of that same repository is already running

### `POST /query`

```json
{"query": "where is JWT validated?", "top_k": 8, "explain": false,
 "repo_path": "/abs/path/to/repo",
 "language": "python", "path_prefix": "backend/", "symbol_type": "method"}
```

`top_k` is 1–20; omitted, the `TOP_K` setting is used (clamped to that range).
`repo_path` scopes the search; omitted, the most recently
indexed repository is used. The three filters are optional and applied in SQL
before scoring. `path_prefix` is an exact, case-sensitive prefix of the
repository-relative path.

```json
{
  "results": [{
    "symbol_name": "validate_jwt_token",
    "qualified_name": "middleware.AuthMiddleware.validate_jwt_token",
    "symbol_type": "method",
    "file_path": "auth/middleware.py",
    "start_line": 34, "end_line": 67,
    "language": "python",
    "chunk_text": "def validate_jwt_token(self, token): ...",
    "score": 0.71
  }],
  "explain_text": null, "query_ms": 38, "total_indexed": 6104
}
```

`score` is cosine similarity clamped to [0, 1]. **It is a similarity, not a
probability and not a confidence.**

- `400` — blank query, or a repository that has not been indexed
- `409` — the index was built with a different embedding model

### `GET /status`

```json
{"indexed_chunks":6104,"last_indexed":"2026-04-16T14:20:16+00:00",
 "repo_path":"/abs/path/to/repo","db_path":"./.codelens_index",
 "embed_model":"nomic-embed-text","watching":true}
```

### `GET /health`

```json
{"ollama": true, "index": true, "ollama_error": null}
```

`index` is true only when a query against the index database succeeds.

CORS is restricted to `vscode-webview://*`; the extension host sends no
`Origin` and is unaffected. Requests must also carry a loopback `Host`
(`127.0.0.1` or `localhost`, set by `CODELENS_ALLOWED_HOSTS`); anything else
gets `400`. CORS alone does not stop DNS rebinding, where a web page re-points
its own domain at `127.0.0.1` and so counts as same-origin.

---

## VS Code integration

| Feature | Behaviour |
|---|---|
| Sidebar search | Natural-language input, ranked cards with symbol kind, language and line range |
| Jump to file | Opens at the exact line. The path is resolved and verified to stay inside its workspace folder before opening. |
| Re-index | `POST /index` with a properly buffered SSE progress bar. In a multi-root workspace, pick one folder or all of them. Run from the Command Palette or editor title bar, it opens the sidebar first. A file that fails is reported under the bar without stopping the run; chunks that could not be embedded are reported, never counted as indexed. |
| Copy | Copies a result's code to the clipboard |
| Several VS Code windows | Later windows use the first window's backend. If it goes away, a remaining window starts its own within seconds. |
| Multi-root workspaces | Search covers every indexed folder, merged into one ranking; each hit shows which folder it came from |
| Sidebar safety | Every value from the index or the LLM is HTML-escaped, and a Content-Security-Policy with a per-load nonce means only the panel's own script can run |
| Status bar | Chunk count while healthy; distinct text for starting, offline, stopped and crashed |
| Restart backend | Offered automatically when the backend crashes mid-session |
| Explain mode | Optional local-LLM summary of the top hits |
| Keyboard | `↑↓` navigate, `Enter` search, `Esc` clear |

---

## Running the tests

```bash
python -m pip install -e ".[dev]"
python -m pytest              # 176 tests

npm install
npm test                      # 55 Jest tests
```

**No test requires a running Ollama, a model, or any network access** —
embedding is monkeypatched throughout, and every test runs against a store in a
temporary directory.

Coverage: parser (containers vs leaves, decorated names, nested scopes, Java,
TypeScript, fallbacks), store (CRUD, ranking, `repo_id` filtering, metadata,
old-schema rebuild), indexer (the incremental diff asserted on actual embed
calls, delete, rename, force reindex, two-repository isolation, and the
byte-identical-functions regression), embedding (failure counting, dimension
derivation, model mismatch), API (status codes, SSE shape, CORS, a genuinely
concurrent index returning 409), watcher (all four event types plus debouncing),
and on the extension side SSE parsing at every chunk boundary, path containment,
and the backend readiness/crash/shutdown state machine.

CI runs the Python suite on Ubuntu and Windows across Python 3.11 and 3.12,
type-checks and Jest-tests the extension, and unzips the built `.vsix` to assert
the backend is actually inside it.

---

## Limitations

Current, real, and worth knowing before relying on this:

- **Search is exhaustive.** Every chunk in the repository is read and scored on
  every query. Measured at ~224 ms per query over 20,000 chunks at 768
  dimensions. There is no ANN index; this is linear in corpus size.
- **First index is slow**, because it is one Ollama round-trip per chunk.
  Bounded concurrency (default 4) helps but does not remove it.
- **Ollama is a hard dependency.** With it unreachable, indexing stores nothing
  and querying fails. This is deliberate — the alternative was storing zero
  vectors, which silently corrupted the index.
- **Changing `EMBED_MODEL` invalidates every index.** Queries are refused until
  you re-index, rather than returning meaningless scores.
- **The schema version drops and rebuilds on mismatch.** The index is a cache;
  an upgrade means re-indexing.
- **Watchers last for one backend session.** Every repository indexed while the
  backend runs is watched live, but after a restart only the most recently
  indexed one is watched again until the others are re-indexed.
- **No ranking beyond cosine similarity** — no reranking, no hybrid keyword
  scoring, no call-graph awareness.
- **Container chunks are header-only**, so a query matching text that only
  exists inside a method body matches the method, never its class.
- **`symbol_type` and `qualified_name` are best-effort** per grammar. Where two
  same-named symbols in a file cannot be told apart, an occurrence ordinal
  disambiguates, which is positional and can shift if they are reordered.
- **Not published to the Marketplace**; install the `.vsix` locally.
- **No authentication.** The backend binds loopback and assumes a single
  trusted local user.

---

## Possible future extensions

Not implemented. Nothing in this section exists in the code.

- An approximate index (HNSW/IVF) for repositories where linear scan stops being
  acceptable
- Hybrid retrieval combining keyword and vector scoring
- A `.codelensignore` file
- Reusing git to bound re-indexing to changed files
- Cross-file relationships ("callers of this")

---

## Project structure

```
CodeLens/
├── backend/
│   ├── config.py               # Environment-backed settings
│   ├── vector_store.py         # VectorStore Protocol - the storage contract
│   ├── db_client.py            # LocalVectorStore: SQLite + exact cosine search
│   ├── indexer.py              # Walker, incremental diff, embedding, watcher
│   ├── main.py                 # FastAPI app + the API contract models
│   ├── query.py                # Query embedding + search + explain mode
│   └── tree_sitter_parser.py   # AST chunking (containers vs leaves)
├── extension/
│   ├── src/
│   │   ├── extension.ts        # Activation, commands, status bar
│   │   ├── backendProcess.ts   # Spawn, /health polling, crash, shutdown
│   │   ├── pythonEnv.ts        # Interpreter discovery + venv provisioning
│   │   ├── toolResolver.ts     # Cross-platform Python/Ollama resolution
│   │   ├── sseParser.ts        # Incremental SSE parser
│   │   ├── paths.ts            # Workspace containment
│   │   ├── multiRoot.ts        # Multi-root result merge + folder validation
│   │   ├── apiTypes.ts         # TypeScript mirror of the contract
│   │   ├── config.ts           # Settings + API base URL
│   │   └── searchPanel.ts      # Webview provider
│   ├── media/panel.html        # Sidebar UI
│   └── test/                   # Jest tests
├── tests/                      # pytest suite
├── requirements.txt            # Bundled into the .vsix, installed on first run
├── backend.Dockerfile          # Backend image for docker-compose.yml
├── docker-compose.yml          # Backend + Ollama, loopback-only
└── .github/workflows/ci.yml
```

---



