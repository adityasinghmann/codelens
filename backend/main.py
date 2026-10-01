import os
import time
import json
import asyncio
import logging
import threading
from datetime import datetime
from contextlib import asynccontextmanager

import httpx

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load explicitly before importing internal config
load_dotenv()

from backend.config import Settings
from backend.db_client import get_db, repo_id_for, normalize_rel_path
from backend.indexer import Indexer, chunk_file
from backend.query import run_query, EmbeddingModelMismatch

logger = logging.getLogger(__name__)

# The last_repo.json sidecar is gone: the repositories table is the record of
# what has been indexed, and unlike the sidecar it can hold more than one.
#
# One live watcher per repository indexed during this process, keyed by
# repo_id. A multi-root workspace indexes several folders, and every one of
# them must stay current - not just whichever was indexed last.
_watchers: dict[str, Indexer] = {}
_watchers_lock = threading.Lock()

# repo_ids with an index run in flight. Two overlapping POST /index calls for
# the same repository previously raced on global_indexer and on each other's
# rows; the second is now rejected with 409. Different repositories may run
# concurrently - that falls out of the per-repo scoping and needs no queue.
_active_indexing: set[str] = set()
_active_lock = threading.Lock()

async def probe_ollama(timeout: float = 1.5) -> tuple[bool, str | None]:
    """
    Check that Ollama answers, without blocking the event loop.

    Both the health endpoint and lifespan startup previously called
    urllib.request.urlopen() inside async functions. That is a synchronous
    socket call: for the length of its timeout it blocked the whole event loop,
    so with Ollama down every /health request stalled every other request the
    server was handling.
    """
    url = f"{Settings.OLLAMA_HOST}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            await client.get(url)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_repo_path(raw: str) -> str:
    """
    Resolve and validate a caller-supplied repository path.

    Returns the normalised absolute path. Raises HTTPException(400) if it is
    empty, does not exist, or is not a directory. The previous code checked
    existence but never normalised, so the same repository reached the store
    under several spellings.
    """
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="repo_path must not be empty.")

    try:
        resolved = os.path.realpath(os.path.abspath(os.path.expanduser(raw.strip())))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"repo_path could not be resolved: {e}")

    if not os.path.exists(resolved):
        raise HTTPException(status_code=400, detail=f"repo_path does not exist: {resolved}")
    if not os.path.isdir(resolved):
        raise HTTPException(status_code=400, detail=f"repo_path is not a directory: {resolved}")

    return resolved


def watch_repository(indexer: Indexer) -> None:
    """Start a live watcher for this indexer's repository, unless one runs."""
    with _watchers_lock:
        if indexer.repo_id in _watchers:
            return
        _watchers[indexer.repo_id] = indexer
    indexer.start_watchdog()


def stop_all_watchers() -> None:
    with _watchers_lock:
        indexers = list(_watchers.values())
        _watchers.clear()
    for indexer in indexers:
        indexer.stop_watchdog()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init DB explicitly to load resources safely
    db = get_db()
    
    # Log rather than print, and keep it ASCII: on Windows the console encoding
    # is cp1252 by default and a non-encodable character raised
    # UnicodeEncodeError from inside lifespan, which aborted application startup
    # entirely -- the backend never came up at all.
    logger.info("CodeLens backend starting")
    logger.info("Index path: %s", Settings.INDEX_PATH)
    logger.info("Total chunks in index: %d", db.count())

    reachable, err = await probe_ollama(timeout=2.0)
    if reachable:
        logger.info("Ollama status: online")
    else:
        logger.warning("Ollama status: offline (%s)", err)

    recent = db.most_recent_repository()
    if recent and os.path.isdir(recent["root_path"]):
        logger.info("Restoring live file watcher for: %s", recent["root_path"])
        watch_repository(Indexer(recent["root_path"]))

    yield

    # Teardown: stop every watcher and its background index loop.
    stop_all_watchers()

app = FastAPI(title="CodeLens Offline Core", lifespan=lifespan)

# The backend binds 127.0.0.1 and serves exactly one client: the extension's
# webview. VS Code webviews send Origin: vscode-webview://<uuid>, and requests
# from the extension host itself carry no Origin at all (and so are unaffected
# by CORS). allow_origins=["*"] on a loopback service let any web page the user
# visited probe and read this API; the regex below is the narrowest rule that
# still works for the real client.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^vscode-webview://.*$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# CORS alone does not stop DNS rebinding: a page on attacker.example that
# re-resolves its own name to 127.0.0.1 is same-origin with this server from
# the browser's point of view, so no CORS check ever runs. Its requests still
# carry "Host: attacker.example", so only loopback Host values are served.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=Settings.ALLOWED_HOSTS)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    status_code = 500
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
    return JSONResponse(
        status_code=status_code,
        content={"error": type(exc).__name__, "detail": str(exc)}
    )

# ---------------------------------------------------------------------------
# API contract - the single source of truth.
#
# These models are what FastAPI validates against at runtime, so they cannot
# drift from the server's actual behaviour. extension/src/apiTypes.ts mirrors
# them for the TypeScript side and points back here; keep the two in step when
# changing anything below.
#
# The former shared/types.py and shared/types.ts declared a different, unused
# contract (workspace_path, include_patterns, exclude_patterns, IndexStatus)
# that no code on either side imported. They have been deleted.
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    """Body of POST /index."""
    repo_path: str
    force_reindex: bool = False


class QueryRequest(BaseModel):
    """Body of POST /query."""
    query: str
    # 1..20. When omitted, the TOP_K setting decides (clamped to that range).
    top_k: int | None = Field(None, ge=1, le=20)
    explain: bool = False
    # Which repository to search. Results are always scoped to exactly one
    # repository; when omitted the most recently indexed one is used.
    repo_path: str | None = None

    # Optional filters, applied in SQL before scoring.
    language: str | None = None
    path_prefix: str | None = None
    symbol_type: str | None = None


class QueryResult(BaseModel):
    """One search hit."""
    symbol_name: str
    qualified_name: str = ""
    symbol_type: str = ""
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_text: str
    # Cosine similarity in [0, 1]. A similarity, not a probability.
    score: float


class QueryResponse(BaseModel):
    """Body of the POST /query response."""
    results: list[QueryResult]
    explain_text: str | None = None
    query_ms: int
    total_indexed: int


class StatusResponse(BaseModel):
    """Body of the GET /status response."""
    indexed_chunks: int
    last_indexed: str | None
    # Root of the repository these numbers describe, or null if none indexed.
    repo_path: str | None
    db_path: str
    embed_model: str
    watching: bool


class HealthResponse(BaseModel):
    """Body of the GET /health response."""
    ollama: bool
    index: bool
    ollama_error: str | None = None

def indexer_worker(repo_path: str, repo_id: str, force: bool,
                   q: asyncio.Queue, main_loop: asyncio.AbstractEventLoop):
    def send(event: dict | None):
        asyncio.run_coroutine_threadsafe(q.put(event), main_loop)

    try:
        indexer = Indexer(repo_path)
        db = get_db()

        if force:
            # Drop only THIS repository's files and chunks, then reset its
            # metadata so the model/dimension are re-derived from the rebuild.
            # Other repositories indexed by the same backend are untouched.
            db.clear_repository_content(indexer.repo_id)
            db.set_index_metadata(
                indexer.repo_id,
                embedding_model=Settings.EMBED_MODEL,
                embedding_dimension=0,
                status="rebuilding",
            )
            logger.info("Force reindex: cleared existing index for %s", repo_path)

        files = list(indexer.walk_repo())
        total_files = len(files)
        processed_files = 0
        processed_chunks = 0
        stored = skipped = failed = 0
        start_time = time.time()

        local_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(local_loop)

        for file_path in files:
            rel = os.path.relpath(file_path, repo_path)
            # Reset per file: a failure must never be charged with the chunk
            # count of the previous file.
            parsed = None
            try:
                parsed = chunk_file(file_path, repo_path, indexer.repo_id)
                rel = parsed.rel_path
                processed_chunks += len(parsed.chunks)
                counts = local_loop.run_until_complete(indexer.index_file(parsed))
                stored += counts["stored"]
                skipped += counts["skipped"]
                failed += counts["failed"]
            except Exception as err:
                # A file that could not be parsed or stored at all counts as a
                # failure of every chunk it would have produced.
                failed += max(1, len(parsed.chunks)) if parsed is not None else 1
                send({"type": "error", "message": str(err), "file": rel})
            finally:
                # Count the file as processed even when it failed, so the
                # progress denominator and numerator stay in the same units and
                # the bar always reaches 100%.
                processed_files += 1
                send({
                    "type": "progress",
                    "file": rel,
                    "processed_files": processed_files,
                    "total_files": total_files,
                    "processed_chunks": processed_chunks,
                })

        local_loop.close()

        # Files deleted while nothing was watching are still stored; drop
        # everything this walk did not find. (A forced run starts empty.)
        removed_files = indexer.prune_missing(
            {normalize_rel_path(os.path.relpath(f, repo_path)) for f in files}
        )

        db.set_index_metadata(indexer.repo_id, status="ready")
        db.touch_repository(indexer.repo_id)

        duration = int((time.time() - start_time) * 1000)
        send({
            "type": "complete",
            "total_files": total_files,
            "processed_files": processed_files,
            "total_chunks": processed_chunks,
            # Real accumulated counts from the per-file diff. The previous
            # implementation hardcoded "skipped": 0 with a comment claiming it
            # was tracked elsewhere; it was not tracked at all.
            "stored": stored,
            "skipped": skipped,
            "failed": failed,
            "removed_files": removed_files,
            "duration_ms": duration,
        })

        # Keep this repository live alongside any others already watched. A
        # re-index of an already-watched repository keeps its existing watcher
        # rather than starting a second one on the same tree.
        watch_repository(indexer)

    except Exception as e:
        logger.exception("Indexing failed for %s", repo_path)
        send({"type": "error", "message": str(e), "file": "system"})
    finally:
        with _active_lock:
            _active_indexing.discard(repo_id)
        send(None)


def default_top_k() -> int:
    """TOP_K from the environment, held to the range the API accepts."""
    return max(1, min(20, Settings.TOP_K))


def resolve_repo_id(repo_path: str | None) -> str:
    """
    Work out which repository a query targets.

    Explicit repo_path wins. Without one, fall back to the most recently
    indexed repository so a single-repo setup keeps working. Either way the
    search is scoped to exactly one repo_id.
    """
    db = get_db()
    if repo_path:
        repo_id = repo_id_for(repo_path)
        if db.get_repository(repo_id) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Repository has not been indexed yet: {repo_path}. Run indexing first.",
            )
        return repo_id

    recent = db.most_recent_repository()
    if recent is None:
        raise HTTPException(
            status_code=400,
            detail="No repository has been indexed yet. Run indexing first.",
        )
    return recent["repo_id"]


@app.post("/index")
async def api_index(req: IndexRequest):
    repo_path = validate_repo_path(req.repo_path)
    repo_id = repo_id_for(repo_path)

    # Reject a second concurrent index of the SAME repository. Two overlapping
    # runs raced on the module-level global_indexer and interleaved writes to
    # the same rows. Different repositories are allowed to proceed in parallel.
    with _active_lock:
        if repo_id in _active_indexing:
            raise HTTPException(
                status_code=409,
                detail=f"An index of this repository is already running: {req.repo_path}",
            )
        _active_indexing.add(repo_id)

    q: asyncio.Queue = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    t = threading.Thread(
        target=indexer_worker,
        args=(repo_path, repo_id, req.force_reindex, q, main_loop),
        daemon=True,
    )
    t.start()

    async def sse_gen():
        while True:
            event = await q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")

@app.post("/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query payload empty.")

    repo_id = resolve_repo_id(req.repo_path)

    start_time = time.time()
    db = get_db()

    try:
        data = await run_query(
            req.query,
            repo_id=repo_id,
            top_k=req.top_k if req.top_k is not None else default_top_k(),
            explain=req.explain,
            language=req.language,
            path_prefix=req.path_prefix,
            symbol_type=req.symbol_type,
        )
    except EmbeddingModelMismatch as e:
        # 409: the index is in a state incompatible with the request, and the
        # user has a concrete action (re-index). Not a 500.
        raise HTTPException(status_code=409, detail=str(e))

    query_ts = int((time.time() - start_time) * 1000)
    # Scoped to the repository actually searched, not the whole database.
    total_indexed = db.count(repo_id)

    logger.info(
        "Query: %r | repo %s | results: %d | %dms",
        req.query, repo_id[:8], len(data["results"]), query_ts,
    )

    return {
        "results": data["results"],
        "explain_text": data.get("explain_text"),
        "query_ms": query_ts,
        "total_indexed": total_indexed,
    }


@app.get("/status", response_model=StatusResponse)
async def api_status():
    db = get_db()

    # Report on the most recently indexed repository; `watching` says whether
    # that repository has a live watcher.
    repo = db.most_recent_repository()
    with _watchers_lock:
        watcher = _watchers.get(repo["repo_id"]) if repo else None

    return {
        "indexed_chunks": db.count(repo["repo_id"]) if repo else 0,
        "last_indexed": repo["updated_at"] if repo else None,
        "repo_path": repo["root_path"] if repo else None,
        "db_path": Settings.INDEX_PATH,
        "embed_model": Settings.EMBED_MODEL,
        "watching": watcher is not None and watcher.observer is not None,
    }


@app.get("/health", response_model=HealthResponse)
async def api_health():
    ollama_ok, ollama_err = await probe_ollama(timeout=1.5)

    # Actually query the index. The previous check accepted the index path's
    # parent directory, which for the default "./.codelens_index" is "." and
    # always exists, so it reported true even with no usable database.
    try:
        get_db().count_files()
        db_ok = True
    except Exception as e:
        logger.warning("Index health check failed: %s", e)
        db_ok = False

    return {
        "ollama": ollama_ok,
        "index": db_ok,
        "ollama_error": ollama_err
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
