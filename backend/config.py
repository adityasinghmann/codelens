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

    # Host header values the API answers to. Loopback names only by default:
    # a DNS-rebinding page reaches 127.0.0.1 under its own domain name, and
    # rejecting that Host is what stops it reading indexed code.
    ALLOWED_HOSTS: list[str] = [
        h.strip() for h in os.getenv("CODELENS_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
        if h.strip()
    ]

    # Model used to embed both indexed chunks and incoming queries. Both sides
    # must use the same model for the vectors to be comparable at all.
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # Model used only for optional "explain" summaries. Previously hardcoded
    # at the call site in query.py.
    EXPLAIN_MODEL: str = os.getenv("EXPLAIN_MODEL", "mistral")

    # Result count for a POST /query that does not send top_k. Clamped to the
    # API's 1..20 range where it is used, in main.default_top_k().
    TOP_K: int = int(os.getenv("TOP_K", "10"))

    # How many embedding requests may be in flight at once. Bounded on purpose:
    # Ollama is a single local process and unbounded concurrency degrades it
    # rather than helping.
    EMBED_CONCURRENCY: int = max(1, int(os.getenv("EMBED_CONCURRENCY", "4")))

    # Chunks handed to the embedder per batch.
    EMBED_BATCH_SIZE: int = max(1, int(os.getenv("EMBED_BATCH_SIZE", "20")))
