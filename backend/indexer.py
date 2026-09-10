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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IGNORE_DIRS = {"node_modules", ".git", "vendor", "dist", "__pycache__", "build",
               ".venv", "venv", ".vectorai_db", ".codelens_index", "out"}

SUPPORTED_EXTENSIONS = set(LANGUAGES.keys()) | {".md", ".txt", ".toml", ".yaml", ".yml"}


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
        self.db = get_db()
        self.repo_id = self.db.ensure_repository(repo_path)
        self.observer: Optional[Observer] = None
        self.ollama_client = AsyncClient(host=Settings.OLLAMA_HOST)

    def walk_repo(self) -> Generator[str, None, None]:
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    yield os.path.join(root, file)

    async def _embed_text(self, text: str) -> List[float]:
        """Embed one text string. Returns a zero vector on failure."""
        try:
            resp = await self.ollama_client.embeddings(
                model=Settings.EMBED_MODEL,
                prompt=f"search_document: {text}"
            )
            return list(resp.embedding) if hasattr(resp, "embedding") else resp["embedding"]
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
            return [0.0] * 768

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
            embeddings = []
            for chunk in to_embed:
                embeddings.append(await self._embed_text(chunk["chunk_text"]))
            self.db.upsert_chunks(self.repo_id, file_id, to_embed, embeddings)
            stored = len(to_embed)

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

    def start_watchdog(self):
        if self.observer is not None:
            return
        handler = RepoEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(handler, self.repo_path, recursive=True)
        self.observer.start()
        logger.info(f"Watchdog watching: {self.repo_path}")

    def stop_watchdog(self):
        if self.observer is None:
            return
        self.observer.stop()
        self.observer.join(timeout=5)
        self.observer = None


class RepoEventHandler(FileSystemEventHandler):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def _is_indexable(self, src_path: str) -> bool:
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in LANGUAGES:
            return False
        parts = src_path.split(os.sep)
        return not any(ign in parts for ign in IGNORE_DIRS)

    def on_modified(self, event):
        if event.is_directory or not self._is_indexable(event.src_path):
            return
        logger.info(f"File changed: {event.src_path} - re-indexing")
        self.reindex_safe(event.src_path)

    def reindex_safe(self, path: str):
        try:
            self.indexer.reindex_file(path)
        except Exception as e:
            logger.error(f"Re-index failed for {path}: {e}")
