import os


class Settings:
    """
    Environment-backed configuration.

    CODELENS_INDEX_PATH replaces the former VECTORAI_DB_PATH, which named a
    database this project does not use. The old variable is still honoured so
    an existing .env keeps working.
    """

    INDEX_PATH: str = os.getenv(
        "CODELENS_INDEX_PATH",
        os.getenv("VECTORAI_DB_PATH", "./.codelens_index"),
    )

    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Model used to embed both indexed chunks and incoming queries. Both sides
    # must use the same model for the vectors to be comparable at all.
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # Model used only for optional "explain" summaries. Previously hardcoded
    # at the call site in query.py.
    EXPLAIN_MODEL: str = os.getenv("EXPLAIN_MODEL", "mistral")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    TOP_K: int = int(os.getenv("TOP_K", "10"))
