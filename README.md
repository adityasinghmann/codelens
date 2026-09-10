<div align="center">

<img src="https://img.shields.io/badge/CodeLens-Offline%20Semantic%20Search-6C63FF?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="CodeLens">

# 🔍 CodeLens

### *Natural-language search for your entire codebase — 100% offline, instant results*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![VS Code](https://img.shields.io/badge/VS%20Code-Extension-007ACC?style=flat-square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com)
[![Ollama](https://img.shields.io/badge/Ollama-nomic--embed--text-FFA500?style=flat-square)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**Type a question. Get the exact code. Jump to the line. No internet. No API keys.**

[Quick Start](#-one-command-setup) · [Architecture](#-architecture) · [The vector store](#-the-vector-store) · [Demo](#-demo)

</div>

---

## 🎯 The Problem

I join a large codebase. I know there's *some* function that handles JWT token refresh — but `Ctrl+F "token"` returns 847 matches. I spend 20 minutes hunting. **That's broken.**

Traditional code search is keyword-based. It matches *text*, not *intent*. It can't understand that **"find where user sessions expire"** and `def invalidate_jwt_token()` are the same concept.

**CodeLens fixes this.**

---

## ✨ What CodeLens Does

> Type any natural language question about your codebase. Get the exact functions, classes, and blocks that answer it — **ranked by semantic similarity**, with a one-click jump to the file and line.

```
Query: "where does the app handle database connection errors?"

Result 1  backend/db.py  line 42–78   score 0.94  ████████████ ✓
          def handle_db_exception(err: DatabaseError) → None

Result 2  services/retry.py  line 12–34   score 0.87  ██████████   ✓
          class ConnectionRetryPolicy

Result 3  middleware/errors.py  line 89–102  score 0.79  █████████    ✓
          async def global_error_handler(request, exc)

→ [Jump to file] button opens the file at the exact line in VS Code
```

All of this runs **offline**. Close your WiFi. It still works.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE — FULLY OFFLINE               │
│                                                                  │
│  ┌──────────────────────────────┐   ┌──────────────────────┐   │
│  │      INDEXING PIPELINE       │   │   LOCAL VECTOR STORE  │   │
│  │   (runs once + on file save) │   │   (SQLite, one file)  │   │
│  │                              │   │                       │   │
│  │  📁 File Walker              │   │  • chunk_text         │   │
│  │     .py .ts .go .rs .java   │──▶│  • file_path          │   │
│  │         │                    │   │  • line_start/end     │   │
│  │         ▼                    │   │  • symbol_name        │   │
│  │  🌳 AST Chunker (tree-sitter)│   │  • language           │   │
│  │     functions, classes,      │   │  • content_hash       │   │
│  │     methods, interfaces      │   │  • embedding          │   │
│  │         │                    │   │                       │   │
│  │         ▼                    │   └──────────┬────────────┘   │
│  │  🤖 Ollama Embedder          │              │                 │
│  │     nomic-embed-text (~270MB)│◀─────────────┘                │
│  │     runs 100% offline        │              │                 │
│  └──────────────────────────────┘              │                 │
│                                                │ exact cosine    │
│  ┌──────────────────────────────┐              │ top-k results   │
│  │      QUERY PIPELINE          │              │                 │
│  │   (triggered by user)        │◀─────────────┘                │
│  │                              │                               │
│  │  💬 VS Code Sidebar          │   ┌──────────────────────┐   │
│  │     natural language input   │   │  Local LLM (optional) │   │
│  │         │                    │   │  Ollama + Mistral 7B  │   │
│  │         ▼                    │──▶│  "Explain mode"       │   │
│  │  🔢 Query Embedder           │   │  plain-English summary│   │
│  │     same model as indexer    │   └──────────────────────┘   │
│  │         │                    │                               │
│  │         ▼                    │                               │
│  │  📊 Result Renderer          │                               │
│  │     snippets + file links    │                               │
│  │     syntax highlighting      │                               │
│  │     [Jump to file] button   │                               │
│  └──────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### Two Pipelines, One Core

| Pipeline | Trigger | What it does |
|---|---|---|
| **Indexing** | Once on setup, then auto on file save | Walks repo → AST chunking → embed → store in the local index |
| **Query** | Every user search | Embed query → exact cosine scoring → render ranked results |

---

## 🗄️ The Vector Store

The index is **SQLite plus exact cosine similarity computed with numpy**. There
is no external database, no server to run, and no network dependency: the whole
index is one file on disk that you can copy, back up, or delete.

**Why this and not a vector database?** The requirement was that the tool work
fully offline on a developer's laptop with nothing else installed. SQLite ships
with Python, so the store adds zero operational surface. At the scale one
repository reaches, scoring every row is fast enough, and it is exactly correct
rather than approximately correct.

**How search actually works.** Every embedding for the repository is read into
one numpy matrix and scored against the query vector with a single
matrix-vector product, then the top *k* are returned. This is an **exhaustive
(exact) search, not an ANN index** - there is no HNSW, no IVF, no quantisation.
Cost grows linearly with the number of indexed chunks. That is the honest
trade, and it is the first thing that would need to change to scale much
further.

The store sits behind a narrow interface (`VectorStore`), so a different
backend could be substituted without touching the indexer or the query path.
No such backend is implemented.

### What the store holds per chunk

```python
{
    "chunk_id":     "…",                 # stable identity: repo + file + symbol
    "symbol_name":  "handle_db_exception",
    "qualified_name": "backend.db.handle_db_exception",
    "symbol_type":  "function",
    "chunk_text":   "def handle_db_exception…",
    "file_path":    "backend/db.py",     # repository-relative
    "start_line":   42,
    "end_line":     78,
    "language":     "python",
    "content_hash": "a3f1c9d…",          # change detection, not identity
    "embedding":    [0.021, -0.134, …],  # width comes from the model
}
```

---

## 🌳 AST-Based Chunking (Not Line Splits)

Most naive RAG systems split code every N lines. This produces garbage chunks that split functions in half, include imports in the middle of class bodies, and destroy semantic coherence.

CodeLens uses **tree-sitter** to parse the actual AST of each file, then extracts *complete semantic units*:

```
Python   → function_definition, class_definition, decorated_definition
TypeScript → function_declaration, method_definition, class_declaration, arrow_function
Go       → function_declaration, method_declaration, type_declaration
Rust     → function_item, impl_item, struct_item
Java     → method_declaration, class_declaration, constructor_declaration
```

Each chunk is a **complete, meaningful code unit** — never half a function, never a fragment. This is why search results are actually useful.

> **Fallback**: For files with no recognized AST nodes (config, markdown, etc.), CodeLens falls back to a 40-line sliding window with 10-line overlap.

---

## 🚀 One-Command Setup

```bash
git clone https://github.com/TryingtobeingNikhil/CodeLens.git
cd CodeLens
chmod +x setup.sh
./setup.sh
```

That's it. The script:
1. Checks for [Ollama](https://ollama.com) (installs prompt if missing)
2. Starts the Ollama service
3. Pulls `nomic-embed-text` (~270 MB, one-time)
4. Installs all Python dependencies
5. Initialises the local index directory
6. Builds the VS Code extension

Then launch the backend:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

---

## 📦 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Vector store** | SQLite + numpy | Local, file-based; exact cosine similarity over stored embeddings |
| **Embedding Model** | `nomic-embed-text` via Ollama | 768-dim code embeddings, fully offline |
| **Code Parsing** | `tree-sitter` (6 languages) | AST-based semantic chunking |
| **API Server** | FastAPI + Uvicorn | `POST /index`, `POST /query`, `GET /status` |
| **File Watching** | `watchdog` | Auto-reindex on file save |
| **IDE Integration** | VS Code Extension (TypeScript) | Sidebar UI, jump-to-file, status bar |
| **Optional LLM** | A local chat model via Ollama (default `mistral`) | "Explain mode" — plain English summaries |
| **Progress UI** | `rich` | Beautiful CLI progress bars during indexing |

---

## 🔌 API Contract

The contract is defined once, as the Pydantic models in `backend/main.py`.
Those models are what FastAPI validates against at runtime, so they cannot
drift from the server's behaviour. `extension/src/apiTypes.ts` is a
hand-maintained TypeScript mirror of the same shapes and points back at them.

### `POST /index` - index a repository

Request: `{"repo_path": "/abs/path/to/repo", "force_reindex": false}`

Responds with a `text/event-stream`. Each event is `data: <json>

`:

```json
{"type": "progress", "file": "src/auth.py", "processed_files": 12, "total_files": 240, "processed_chunks": 318}
{"type": "error",    "file": "src/broken.py", "message": "..."}
{"type": "complete", "total_files": 240, "processed_files": 240, "total_chunks": 6104, "duration_ms": 41200}
```

`processed_files` and `total_files` are in the same unit, so
`processed_files / total_files` is the real progress fraction. A file that
fails to parse still advances `processed_files`, so the stream always reaches
100%. `file` is repository-relative.

Returns 400 if `repo_path` does not exist or is not a directory.

### `POST /query` - semantic search

Request: `{"query": "where is JWT validated?", "top_k": 8, "explain": false}`
(`top_k` must be 1..20; a blank query returns 400.)

```json
{
  "results": [
    {
      "symbol_name": "validate_jwt_token",
      "file_path": "auth/middleware.py",
      "start_line": 34,
      "end_line": 67,
      "language": "python",
      "chunk_text": "def validate_jwt_token(token: str) -> User: ...",
      "score": 0.71
    }
  ],
  "explain_text": null,
  "query_ms": 38,
  "total_indexed": 6104
}
```

`score` is cosine similarity clamped to [0, 1]. It is a similarity, not a
probability and not a confidence.

### `GET /status`

```json
{"indexed_chunks": 6104, "last_indexed": "2026-04-16T14:20:16", "db_path": "./.codelens_index",
 "embed_model": "nomic-embed-text", "watching": true}
```

### `GET /health`

```json
{"ollama": true, "index": true, "ollama_error": null}
```

> The former `shared/types.py` and `shared/types.ts` declared a different
> contract (`workspace_path`, `include_patterns`, `exclude_patterns`,
> `IndexStatus`) that neither side ever imported. They have been deleted.

---

## 💡 VS Code Extension Features

| Feature | Description |
|---|---|
| **Semantic Search Sidebar** | Natural language input, ranked results with syntax highlighting |
| **Jump to File** | Click any result → VS Code opens the file at the exact line |
| **Live Re-indexing** | "Re-index" button triggers `POST /index` with SSE progress bar |
| **Status Bar** | Shows `⬡ CodeLens: 9,432 chunks` — always visible |
| **Offline Syntax Highlighting** | Built-in highlight.js stub — no CDN, no internet |
| **Keyboard Navigation** | `↑↓` arrows navigate results, `Enter` to search, `Esc` to clear |
| **AI Explain Mode** | Optional "Include AI explanation" checkbox → Mistral 7B summarizes results |
| **Shimmer Loading** | Skeleton cards while waiting — feels production-quality |
| **Example Queries** | Clickable chips on empty state to onboard new users instantly |

---

## 🔁 Incremental Indexing

CodeLens uses MD5 content hashing to **skip unchanged chunks**. When the file watcher triggers on a save:

1. The modified file is re-parsed by tree-sitter
2. New content hashes are generated for each chunk
3. Only chunks with **new hashes** are embedded and upserted
4. Deleted/modified functions are removed by `delete_by_filepath` before re-insert

This means **re-indexing a large codebase after a single file edit takes milliseconds**, not minutes.

---

## 🔒 Privacy by Design

| What never leaves your machine |
|---|
| Your source code |
| Your query text |
| Your embeddings |
| Your search history |

There are no telemetry calls, no analytics pings, no cloud sync. The only network calls CodeLens makes are to `localhost:11434` (Ollama) and `localhost:8000` (its own backend).

---

## 📁 Project Structure

```
CodeLens/
├── backend/                    # Python FastAPI backend
│   ├── __init__.py
│   ├── config.py               # Environment-based settings
│   ├── db_client.py            # Local SQLite vector store
│   ├── indexer.py              # File walker + AST chunker + embedder
│   ├── main.py                 # FastAPI server (index/query/status/health)
│   ├── query.py                # Query embedding + exact cosine search + explain mode
│   └── tree_sitter_parser.py   # Multi-language AST parsing utilities
│
├── extension/                  # VS Code extension (TypeScript)
│   ├── src/
│   │   ├── extension.ts        # Extension entry, command + status bar wiring
│   │   ├── backendProcess.ts   # Backend lifecycle: spawn, readiness, shutdown
│   │   ├── pythonEnv.ts        # Interpreter discovery + venv provisioning
│   │   ├── toolResolver.ts     # Cross-platform Python/Ollama resolution
│   │   ├── sseParser.ts        # Incremental Server-Sent Events parser
│   │   ├── apiTypes.ts         # TypeScript mirror of the backend contract
│   │   ├── config.ts           # Settings + API base URL (single source)
│   │   └── searchPanel.ts      # Webview provider, message handler
│   └── media/
│       └── panel.html          # Full sidebar UI (search, results, progress)
│
├── setup.sh                    # ⭐ One-command full setup script
├── docker-compose.yml          # Docker deployment option
├── pyproject.toml              # Python dependency manifest
├── package.json                # Extension manifest + VS Code contribution points
├── tsconfig.json               # TypeScript compiler config
└── .env.example                # Environment variable template
```

---

## 🎬 Demo

> **The killer demo**: Turn off WiFi. Open VS Code. Type a question. Watch it work.

Three queries that demonstrate semantic understanding (not keyword matching):

| Query | What it finds | Why it's impressive |
|---|---|---|
| `"where does the app validate user sessions?"` | `def validate_jwt_token()` in `auth/middleware.py` | "validate" ≠ "sessions" ≠ "jwt" — semantic match |
| `"show me all database write operations"` | `INSERT`, `UPDATE`, `batch_upsert` across 6 files | Finds writes by *concept*, not by SQL keyword |
| `"which functions handle error logging?"` | Logger calls across multiple modules | "handle" + "error" + "logging" as a concept, not words |

---

## 🗺️ Roadmap

- [ ] **Dead code detector** — embed all function signatures, flag those with zero similarity to the rest of the codebase
- [ ] **Query history** — persisted locally, shown in sidebar with timestamps
- [ ] **Multi-workspace support** — index and switch between multiple repos
- [ ] **`.codelens` ignore file** — like `.gitignore` but for indexing
- [ ] **Commit-aware indexing** — only re-embed chunks changed since last git commit
- [ ] **Cross-file semantic linking** — show "functions that call this" / "functions called by this"

---

## 🤝 Contributing

```bash
# Backend development
pip install -e ".[dev]"
uvicorn backend.main:app --reload

# Extension development
npm install
npm run watch
# F5 in VS Code to launch Extension Development Host
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**Built For hackthon & I literally killed it. ** · April 2026

*CodeLens is proof that developer tools don't need the cloud to be powerful.*

</div>
