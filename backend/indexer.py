"""
Indexer - walks a repo, AST-chunks files, embeds with Ollama,
and upserts into the local SQLite vector store.

Everything here is scoped to a repository. The store keys files and chunks by
repo_id, so two repositories indexed by the same backend never see each
other's code.
"""

import os
import hashlib
import asyncio
import logging
import threading
from typing import List, Generator, Dict, Any, NamedTuple, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from ollama import AsyncClient

from backend.db_client import (
    get_db,
    assign_chunk_ids,
    normalize_rel_path,
)
from backend.config import Settings
from backend.tree_sitter_parser import LANGUAGES, extract_chunks, _sliding_window
from backend.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IGNORE_DIRS = {"node_modules", ".git", "vendor", "dist", "__pycache__", "build",
               ".venv", "venv", ".vectorai_db", ".codelens_index", "out"}

SUPPORTED_EXTENSIONS = set(LANGUAGES.keys()) | {".md", ".txt", ".toml", ".yaml", ".yml"}


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised when a vector's width disagrees with what the index recorded."""


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class ParsedFile(NamedTuple):
    """One file's parse result, ready to be diffed against the store."""
    rel_path: str
    language: str
    #  Hash of the whole file - cheap "did anything change at all?" check.
    content_hash: str
    chunks: List[Dict[str, Any]]


def chunk_file(file_path: str, repo_root: str, repo_id: str) -> ParsedFile:
    """
    Read one file, AST-chunk it, and give every chunk a stable identity.

    Two hashes are produced and they mean different things:
      - ParsedFile.content_hash is the whole file, for skipping untouched files
      - chunk["content_hash"] is that chunk's text, for detecting which
        individual symbols changed
    Identity itself is chunk["chunk_id"], which is independent of both.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return ParsedFile(normalize_rel_path(os.path.relpath(file_path, repo_root)), "", "", [])

    rel_path = normalize_rel_path(os.path.relpath(file_path, repo_root))
    ext = os.path.splitext(file_path)[1].lower()

    if not content.strip():
        return ParsedFile(rel_path, "", md5_hash(content), [])

    if ext in LANGUAGES:
        raw_chunks = extract_chunks(rel_path, content)
    else:
        raw_chunks = _sliding_window(rel_path, content, "unknown")

    for chunk in raw_chunks:
        chunk["content_hash"] = md5_hash(chunk["chunk_text"])
        chunk["file_path"] = rel_path

    assign_chunk_ids(repo_id, rel_path, raw_chunks)

    language = raw_chunks[0].get("language", "") if raw_chunks else ""
    return ParsedFile(rel_path, language, md5_hash(content), raw_chunks)


class Indexer:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        # Typed as the interface, not the SQLite class: nothing in the indexer
        # may depend on how storage is implemented.
        self.db: VectorStore = get_db()
        self.repo_id = self.db.ensure_repository(repo_path)
        self.observer: Optional[Observer] = None
        self.worker: Optional["_IndexWorker"] = None
        self.ollama_client = AsyncClient(host=Settings.OLLAMA_HOST)

    def walk_repo(self) -> Generator[str, None, None]:
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    yield os.path.join(root, file)

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """
        Embed one text string, or return None if the embedding could not be
        produced.

        None, not a zero vector. The previous implementation returned
        [0.0] * 768 on failure and that vector was persisted as a real chunk:
        it never matched anything, was never retried because its content_hash
        looked current, and silently corrupted the index. The 768 was also
        hardcoded regardless of the model's actual width.
        """
        try:
            resp = await self.ollama_client.embeddings(
                model=Settings.EMBED_MODEL,
                prompt=f"search_document: {text}"
            )
            vector = list(resp.embedding) if hasattr(resp, "embedding") else resp["embedding"]
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
            return None

        if not vector:
            logger.error("Ollama returned an empty embedding for a chunk; dropping it.")
            return None
        return vector

    async def _embed_many(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Embed many chunks with bounded concurrency.

        The previous code looked batched but was not:
            embeddings = [await self._embed_text(c) for c in batch]
        That awaits each request before starting the next, so a 20-chunk
        "batch" was 20 sequential HTTP round-trips.

        Requests now run under asyncio.gather bounded by a Semaphore of
        EMBED_CONCURRENCY. Concurrency is bounded rather than unbounded because
        Ollama is one local process: firing hundreds of requests at it makes it
        slower, not faster.

        There is no retry here. _embed_text already swallows its own errors and
        returns None, so a failure cannot raise into gather, cancel siblings, or
        trigger a retry storm; each chunk is attempted exactly once per pass and
        a failed one is simply not written, so the next pass picks it up.
        """
        semaphore = asyncio.Semaphore(Settings.EMBED_CONCURRENCY)

        async def one(text: str) -> Optional[List[float]]:
            async with semaphore:
                return await self._embed_text(text)

        results: List[Optional[List[float]]] = []
        batch_size = Settings.EMBED_BATCH_SIZE
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            results.extend(await asyncio.gather(*(one(t) for t in batch)))
        return results

    def _check_dimension(self, vector: List[float]) -> int:
        """
        Reconcile an embedding's width against what this repository recorded.

        The dimension is derived from the first successful embedding rather
        than assumed. A later vector of a different width means the model
        changed underneath the index, which would silently produce garbage
        scores, so it raises instead.
        """
        width = len(vector)
        meta = self.db.get_index_metadata(self.repo_id) or {}
        recorded = meta.get("embedding_dimension") or 0

        if not recorded:
            self.db.set_index_metadata(
                self.repo_id,
                embedding_model=Settings.EMBED_MODEL,
                embedding_dimension=width,
            )
            return width

        if recorded != width:
            raise EmbeddingDimensionMismatch(
                f"Embedding model returned {width}-dimensional vectors but this "
                f"index was built with {recorded} dimensions "
                f"(model {meta.get('embedding_model')!r}, now {Settings.EMBED_MODEL!r}). "
                f"Re-index this repository to rebuild it."
            )
        return width

    async def index_file(self, parsed: ParsedFile) -> Dict[str, int]:
        """
        Reconcile one file's chunks against what is already stored.

        This is a real diff, not a delete-and-re-embed. For each chunk_id:

          in both, content_hash equal  -> keep the embedding, update lines only
          in both, content_hash differs-> re-embed, update in place
          new chunk_id                 -> embed and insert
          stored but no longer present -> delete

        Embedding is the expensive step (a network round-trip per chunk), so
        the whole point is that an unchanged chunk is never re-embedded, even
        when the edit moved it.

        Returns counts of {stored, skipped, deleted, failed}.
        """
        file_id = self.db.upsert_file(
            self.repo_id, parsed.rel_path, parsed.content_hash, parsed.language
        )

        stored_chunks = self.db.get_chunks_for_file(file_id)
        current = {c["chunk_id"]: c for c in parsed.chunks}

        to_embed: List[Dict[str, Any]] = []
        line_updates: List[Dict[str, Any]] = []
        skipped = 0

        for chunk_id, chunk in current.items():
            existing = stored_chunks.get(chunk_id)
            if existing is None:
                to_embed.append(chunk)
            elif existing["content_hash"] != chunk["content_hash"]:
                to_embed.append(chunk)
            else:
                # Unchanged: keep the vector, but the symbol may have moved.
                skipped += 1
                if (existing["start_line"] != chunk.get("start_line")
                        or existing["end_line"] != chunk.get("end_line")):
                    line_updates.append({
                        "chunk_id": chunk_id,
                        "start_line": chunk.get("start_line", 0),
                        "end_line": chunk.get("end_line", 0),
                    })

        removed = [cid for cid in stored_chunks if cid not in current]

        if removed:
            self.db.delete_chunks(removed)
        if line_updates:
            self.db.update_chunk_lines(line_updates)

        stored = failed = 0
        if to_embed:
            keep_chunks: List[Dict[str, Any]] = []
            keep_vectors: List[List[float]] = []

            vectors = await self._embed_many([c["chunk_text"] for c in to_embed])

            for chunk, vector in zip(to_embed, vectors):
                if vector is None:
                    # Dropped, never written. It will be retried on the next
                    # index because no row exists to make it look current.
                    failed += 1
                    continue
                self._check_dimension(vector)
                keep_chunks.append(chunk)
                keep_vectors.append(vector)

            if keep_chunks:
                self.db.upsert_chunks(self.repo_id, file_id, keep_chunks, keep_vectors)
            stored = len(keep_chunks)

            if failed:
                logger.warning(
                    "%d chunk(s) in %s could not be embedded and were not stored",
                    failed, parsed.rel_path,
                )

        return {"stored": stored, "skipped": skipped, "deleted": len(removed), "failed": failed}

    async def embed_and_store(self, parsed: ParsedFile) -> Dict[str, int]:
        """Backwards-compatible name used by the API worker."""
        return await self.index_file(parsed)

    async def reindex_file_async(self, file_path: str) -> Dict[str, int]:
        parsed = chunk_file(file_path, self.repo_path, self.repo_id)
        result = await self.index_file(parsed)
        logger.info(
            "Re-indexed <%s> (%d chunks) in repo %s",
            parsed.rel_path, len(parsed.chunks), self.repo_id[:8],
        )
        return result

    def reindex_file(self, file_path: str) -> Dict[str, int]:
        return asyncio.run(self.reindex_file_async(file_path))

    def remove_file(self, file_path: str) -> None:
        rel = normalize_rel_path(os.path.relpath(file_path, self.repo_path))
        self.db.delete_file(self.repo_id, rel)
        logger.info("Removed <%s> from repo %s", rel, self.repo_id[:8])

    def start_watchdog(self, debounce_seconds: float = 0.5):
        if self.observer is not None:
            return
        self.worker = _IndexWorker(self, debounce_seconds=debounce_seconds)
        self.worker.start()
        handler = RepoEventHandler(self, self.worker)
        self.observer = Observer()
        self.observer.schedule(handler, self.repo_path, recursive=True)
        self.observer.start()
        logger.info(f"Watchdog watching: {self.repo_path}")

    def stop_watchdog(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
        if self.worker is not None:
            self.worker.stop()
            self.worker = None


class _IndexWorker:
    """
    Serialises watcher-driven indexing onto one long-lived background loop.

    Replaces the previous asyncio.run() per filesystem event, which built and
    tore down an event loop inside the watchdog thread for every keystroke-save
    and gave no ordering guarantee between overlapping events.

    Two properties matter here:
      - per-path debouncing, so one editor save (which typically emits several
        events) causes one index pass, not several
      - never two operations for the same path at once, which the single
        consuming task guarantees by construction
    """

    def __init__(self, indexer: "Indexer", debounce_seconds: float = 0.5):
        self.indexer = indexer
        self.debounce = debounce_seconds
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._timers: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._started = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="codelens-index-worker")
        self._thread.start()
        self._started.wait(timeout=5)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def stop(self) -> None:
        with self._lock:
            for handle in self._timers.values():
                handle.cancel()
            self._timers.clear()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None

    def submit(self, action: str, path: str) -> None:
        """
        Queue an action for a path, coalescing repeats within the debounce
        window. A later event for the same path replaces the pending one, so a
        create-then-modify burst resolves to a single index pass.
        """
        if self._loop is None:
            return
        key = os.path.normcase(os.path.abspath(path))
        with self._lock:
            pending = self._timers.pop(key, None)
            if pending is not None:
                pending.cancel()
            timer = threading.Timer(self.debounce, self._dispatch, args=(key, action, path))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _dispatch(self, key: str, action: str, path: str) -> None:
        with self._lock:
            self._timers.pop(key, None)
        loop = self._loop
        if loop is None:
            return
        # Hand the work to the single worker loop. Because there is exactly one
        # consuming loop, two operations for the same file never overlap.
        asyncio.run_coroutine_threadsafe(self._execute(action, path), loop)

    async def _execute(self, action: str, path: str) -> None:
        try:
            if action == "remove":
                self.indexer.remove_file(path)
            else:
                await self.indexer.reindex_file_async(path)
        except Exception as e:
            logger.error("Watcher %s failed for %s: %s", action, path, e)

    def flush(self, timeout: float = 5.0) -> None:
        """Run every pending debounced action now. Used by tests."""
        with self._lock:
            pending = list(self._timers.items())
            self._timers.clear()
        futures = []
        for key, timer in pending:
            timer.cancel()
            args = getattr(timer, "args", None)
            if args and self._loop is not None:
                _, action, path = args
                futures.append(asyncio.run_coroutine_threadsafe(
                    self._execute(action, path), self._loop))
        for fut in futures:
            try:
                fut.result(timeout=timeout)
            except Exception as e:
                logger.error("Flush failed: %s", e)


class RepoEventHandler(FileSystemEventHandler):
    """
    Turns filesystem events into index operations.

    The previous handler implemented on_modified only, so deleted files kept
    their chunks forever, new files were never indexed until something modified
    them, and a rename left the old path orphaned and the new path absent.
    """

    def __init__(self, indexer: "Indexer", worker: "_IndexWorker"):
        self.indexer = indexer
        self.worker = worker

    def _is_indexable(self, src_path: str) -> bool:
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in LANGUAGES:
            return False
        parts = os.path.normpath(src_path).split(os.sep)
        return not any(ign in parts for ign in IGNORE_DIRS)

    def _in_repo(self, path: str) -> bool:
        """Guard against events for paths outside this indexer's repository."""
        try:
            root = os.path.realpath(self.indexer.repo_path)
            return os.path.commonpath([root, os.path.realpath(path)]) == root
        except (ValueError, OSError):
            return False

    def on_created(self, event):
        if event.is_directory or not self._is_indexable(event.src_path):
            return
        if self._in_repo(event.src_path):
            logger.info("File created: %s", event.src_path)
            self.worker.submit("index", event.src_path)

    def on_modified(self, event):
        if event.is_directory or not self._is_indexable(event.src_path):
            return
        if self._in_repo(event.src_path):
            logger.info("File changed: %s", event.src_path)
            self.worker.submit("index", event.src_path)

    def on_deleted(self, event):
        if event.is_directory or not self._is_indexable(event.src_path):
            return
        if self._in_repo(event.src_path):
            logger.info("File deleted: %s", event.src_path)
            self.worker.submit("remove", event.src_path)

    def on_moved(self, event):
        """A rename is a removal of the old path plus an index of the new one."""
        if event.is_directory:
            return
        src, dest = event.src_path, event.dest_path
        if self._is_indexable(src) and self._in_repo(src):
            logger.info("File moved from: %s", src)
            self.worker.submit("remove", src)
        if self._is_indexable(dest) and self._in_repo(dest):
            logger.info("File moved to: %s", dest)
            self.worker.submit("index", dest)
