import os
import time
import json
import asyncio
import logging
import threading
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load explicitly before importing internal config
load_dotenv()

from backend.config import Settings
from backend.db_client import get_db, repo_id_for
from backend.indexer import Indexer, chunk_file
from backend.query import run_query

logger = logging.getLogger(__name__)

# The last_repo.json sidecar is gone: the repositories table is the record of
# what has been indexed, and unlike the sidecar it can hold more than one.
global_indexer: Indexer | None = None

# repo_ids with an index run in flight. Two overlapping POST /index calls for
# the same repository previously raced on global_indexer and on each other's
# rows; the second is now rejected with 409. Different repositories may run
# concurrently - that falls out of the per-repo scoping and needs no queue.
_active_indexing: set[str] = set()
_active_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_indexer
    
    # Init DB explicitly to load resources safely
    db = get_db()
    
    # Log rather than print, and keep it ASCII: on Windows the console encoding
    # is cp1252 by default and a non-encodable character raised
    # UnicodeEncodeError from inside lifespan, which aborted application startup
    # entirely -- the backend never came up at all.
    logger.info("CodeLens backend starting")
    logger.info("Index path: %s", Settings.INDEX_PATH)
    logger.info("Total chunks in index: %d", db.count())

    import urllib.request
    try:
        urllib.request.urlopen(f"{Settings.OLLAMA_HOST}/api/tags", timeout=2)
        logger.info("Ollama status: online")
    except Exception as e:
        logger.warning("Ollama status: offline (%s)", e)

    recent = db.most_recent_repository()
    if recent and os.path.isdir(recent["root_path"]):
        logger.info("Restoring live file watcher for: %s", recent["root_path"])
        global_indexer = Indexer(recent["root_path"])
        global_indexer.start_watchdog()

    yield
    
    # Teardown: stop the watcher and its background index loop.
    if global_indexer is not None:
        global_indexer.stop_watchdog()

app = FastAPI(title="CodeLens Offline Core", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    top_k: int = Field(8, ge=1, le=20)
    explain: bool = False
    # Which repository to search. Results are always scoped to exactly one
    # repository; when omitted the most recently indexed one is used.
    repo_path: str | None = None


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
                failed += max(1, len(parsed.chunks) if "parsed" in dir() else 1)
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
            "duration_ms": duration,
        })

        global global_indexer
        if global_indexer is not None and global_indexer.repo_id != indexer.repo_id:
            global_indexer.stop_watchdog()
        global_indexer = indexer
        global_indexer.start_watchdog()

    except Exception as e:
        logger.exception("Indexing failed for %s", repo_path)
        send({"type": "error", "message": str(e), "file": "system"})
    finally:
        with _active_lock:
            _active_indexing.discard(repo_id)
        send(None)


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
    if not os.path.exists(req.repo_path) or not os.path.isdir(req.repo_path):
        raise HTTPException(
            status_code=400,
            detail=f"repo_path does not exist or is not a directory: {req.repo_path}",
        )

    repo_id = repo_id_for(req.repo_path)

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
        args=(req.repo_path, repo_id, req.force_reindex, q, main_loop),
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

    data = await run_query(req.query, repo_id=repo_id, top_k=req.top_k, explain=req.explain)

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

    # Report on the repository the watcher is attached to, falling back to the
    # most recently indexed one.
    if global_indexer is not None:
        repo = db.get_repository(global_indexer.repo_id)
    else:
        repo = db.most_recent_repository()

    return {
        "indexed_chunks": db.count(repo["repo_id"]) if repo else 0,
        "last_indexed": repo["updated_at"] if repo else None,
        "repo_path": repo["root_path"] if repo else None,
        "db_path": Settings.INDEX_PATH,
        "embed_model": Settings.EMBED_MODEL,
        "watching": global_indexer is not None and global_indexer.observer is not None,
    }


@app.get("/health", response_model=HealthResponse)
async def api_health():
    import urllib.request
    ollama_ok = False
    ollama_err = None
    
    try:
        urllib.request.urlopen(f"{Settings.OLLAMA_HOST}/api/tags", timeout=1)
        ollama_ok = True
    except Exception as e:
        ollama_err = str(e)
        
    db_ok = False
    db_dir = os.path.dirname(Settings.INDEX_PATH)
    if os.path.exists(Settings.INDEX_PATH) or (db_dir and os.path.exists(db_dir)):
        db_ok = True
        
    return {
        "ollama": ollama_ok,
        "index": db_ok,
        "ollama_error": ollama_err
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
