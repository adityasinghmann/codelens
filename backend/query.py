"""
Query engine - embeds a natural-language question and scores it against
every stored chunk with exact cosine similarity. Optionally summarises the
top results with a local chat model.

Changes vs original:
 • ollama SDK 0.5+: resp.embedding (not resp['embedding'])
 • chat_resp.message.content (not chat_resp['message']['content'])
"""

import asyncio
import logging
from typing import Dict, Any, List

from ollama import AsyncClient
from backend.db_client import get_db
from backend.config import Settings

logger = logging.getLogger(__name__)


def _get_ollama_client() -> AsyncClient:
    return AsyncClient(host=Settings.OLLAMA_HOST)


async def run_query(
    query: str,
    repo_id: str,
    top_k: int = 10,
    explain: bool = False,
) -> Dict[str, Any]:
    """
    1. Embed the query with the same model used during indexing.
    2. Score it against every stored chunk (exact cosine similarity).
    3. Optionally summarise the top hits with a local chat model.
    """
    client = _get_ollama_client()

    # --- Embed query ---
    embed_prompt = f"search_query: {query}"
    try:
        resp = await client.embeddings(model=Settings.EMBED_MODEL, prompt=embed_prompt)
        # ollama SDK 0.5+: EmbeddingsResponse object
        query_vec = list(resp.embedding) if hasattr(resp, "embedding") else resp["embedding"]
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise

    # --- Exact cosine search over the stored chunks ---
    db = get_db()
    raw_results = db.search(query_vec, repo_id=repo_id, top_k=top_k)

    # Normalise scores to [0, 1]
    results = []
    for r in raw_results:
        score = r.get("score", 0.0)
        # Cosine scores are already in[-1, 1]; clamp to [0, 1]
        r["score"] = max(0.0, min(1.0, score))
        results.append(r)

    # --- Optional local-LLM explanation ---
    explanation: str | None = None
    if explain and results:
        context = "\n\n".join(r.get("chunk_text", "") for r in results[:3])
        system_msg = (
            "You are a code navigation assistant. "
            "Answer in 2-3 sentences. Reference specific function names and files."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"Codebase context:\n{context}\n\nQuestion: {query}"},
        ]
        try:
            chat_resp = await client.chat(model=Settings.EXPLAIN_MODEL, messages=messages)
            # ollama SDK 0.5+: ChatResponse object
            if hasattr(chat_resp, "message"):
                explanation = chat_resp.message.content
            else:
                explanation = chat_resp["message"]["content"]
        except Exception as e:
            logger.warning(f"Explain step failed (not critical): {e}")
            explanation = f"(Explain mode unavailable - is the {Settings.EXPLAIN_MODEL} model pulled?)"

    return {"results": results, "explain_text": explanation}


def run_query_sync(query: str, repo_id: str, top_k: int = 10, explain: bool = False) -> Dict[str, Any]:
    """Synchronous wrapper for CLI usage."""
    return asyncio.run(run_query(query, repo_id, top_k, explain))
