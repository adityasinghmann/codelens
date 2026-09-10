"""
The storage interface the rest of the backend depends on.

This exists for architectural separation and testability, not for new
infrastructure. `LocalVectorStore` is the only implementation, and no second
backend is planned or claimed. What the Protocol buys:

  * the indexer and query layer depend on a named interface rather than on a
    concrete SQLite class, so what storage owes them is written down in one
    place instead of being implied by whichever methods happen to be called
  * a test can substitute a fake store without a database
  * a future backend has an explicit contract to satisfy

Protocol rather than an abstract base class deliberately: LocalVectorStore does
not inherit from anything, and structural typing keeps the dependency pointing
one way - storage knows nothing about this module.
"""

from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Everything the indexer and query layer need from storage."""

    # -- repositories ------------------------------------------------------

    def ensure_repository(self, root_path: str) -> str:
        """Register a repository (idempotent) and return its repo_id."""

    def get_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """One repository's record, or None."""

    def list_repositories(self) -> List[Dict[str, Any]]:
        """Every known repository, most recently updated first."""

    def most_recent_repository(self) -> Optional[Dict[str, Any]]:
        """The most recently indexed repository, or None."""

    def touch_repository(self, repo_id: str) -> None:
        """Mark a repository as just used."""

    def delete_repository(self, repo_id: str) -> None:
        """Remove a repository and everything under it. Must not touch others."""

    def clear_repository_content(self, repo_id: str) -> None:
        """Drop a repository's files and chunks, keeping the repository row."""

    # -- files -------------------------------------------------------------

    def upsert_file(self, repo_id: str, rel_path: str, content_hash: str,
                    language: str = "") -> str:
        """Record a file and return its file_id."""

    def get_file(self, repo_id: str, rel_path: str) -> Optional[Dict[str, Any]]:
        """One file's record, or None."""

    def list_files(self, repo_id: str) -> List[Dict[str, Any]]:
        """Every file recorded for a repository."""

    def delete_file(self, repo_id: str, rel_path: str) -> None:
        """Remove a file and its chunks from one repository."""

    # -- chunks ------------------------------------------------------------

    def get_chunks_for_file(self, file_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Stored chunks for one file, keyed by chunk_id.

        Returns identity and change-detection metadata, not embeddings: this
        feeds the incremental diff, which never needs the vectors.
        """

    def upsert_chunks(self, repo_id: str, file_id: str,
                      chunks: Sequence[Dict[str, Any]],
                      embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks, keyed by chunk_id."""

    def update_chunk_lines(self, updates: Iterable[Dict[str, Any]]) -> None:
        """Move a chunk's line numbers without touching its embedding."""

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks by id."""

    # -- metadata ----------------------------------------------------------

    def set_index_metadata(self, repo_id: str, embedding_model: Optional[str] = None,
                           embedding_dimension: Optional[int] = None,
                           status: Optional[str] = None) -> None:
        """Record how this repository's index was built. Fields are merged."""

    def get_index_metadata(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """How this repository's index was built, or None."""

    # -- read path ---------------------------------------------------------

    def search(self, embedding: Sequence[float], repo_id: str, top_k: int = 10,
               language: Optional[str] = None, path_prefix: Optional[str] = None,
               symbol_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Rank this repository's chunks against a query vector.

        repo_id is mandatory: there is no unscoped search. Optional filters
        narrow the candidate set before scoring. Results are ordered by
        descending similarity and carry a `score`, which is a similarity in
        [0, 1] - never a probability or a confidence.
        """

    def count(self, repo_id: Optional[str] = None) -> int:
        """Number of stored chunks, for one repository or overall."""

    def count_files(self, repo_id: Optional[str] = None) -> int:
        """Number of recorded files, for one repository or overall."""
