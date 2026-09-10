/**
 * TypeScript mirror of the CodeLens HTTP contract.
 *
 * The authoritative definition is the Pydantic model block in
 * `backend/main.py` - those models are what FastAPI validates every request
 * and response against at runtime, so they cannot drift from the server's
 * behaviour. This file exists only so the extension side is type-checked
 * against the same shapes; when you change a model there, change it here.
 *
 * This replaces the former shared/types.ts, which declared a contract
 * (workspace_path, include_patterns, exclude_patterns, IndexStatus) that
 * neither the backend nor the extension ever used.
 */

/** POST /index request body. */
export interface IndexRequest {
    /** Absolute path to the repository root to index. */
    repo_path: string;
    /** Discard this repository's existing index and rebuild it from scratch. */
    force_reindex: boolean;
}

/** POST /query request body. */
export interface QueryRequest {
    query: string;
    /** 1..20 inclusive; the backend rejects anything outside that range. */
    top_k: number;
    explain: boolean;
}

/** One search hit. */
export interface QueryResult {
    symbol_name: string;
    file_path: string;
    start_line: number;
    end_line: number;
    language: string;
    chunk_text: string;
    /** Cosine similarity in [0, 1]. A similarity, not a probability. */
    score: number;
}

/** POST /query response body. */
export interface QueryResponse {
    results: QueryResult[];
    explain_text: string | null;
    query_ms: number;
    total_indexed: number;
}

/** GET /status response body. */
export interface StatusResponse {
    indexed_chunks: number;
    last_indexed: string | null;
    db_path: string;
    embed_model: string;
    watching: boolean;
}

/** GET /health response body. */
export interface HealthResponse {
    /** Whether the Ollama server answered. */
    ollama: boolean;
    /** Whether the local index directory is present and readable. */
    index: boolean;
    ollama_error: string | null;
}

/**
 * POST /index streams these as Server-Sent Events (`data: <json>\n\n`).
 * They are not FastAPI response models, so the backend emits them as plain
 * dicts from indexer_worker(); the shapes below are that emission.
 */
export type IndexEvent = IndexProgressEvent | IndexCompleteEvent | IndexErrorEvent;

export interface IndexProgressEvent {
    type: 'progress';
    /** Repository-relative path of the file just processed. */
    file: string;
    processed_files: number;
    total_files: number;
    processed_chunks: number;
}

export interface IndexCompleteEvent {
    type: 'complete';
    total_files: number;
    processed_files: number;
    total_chunks: number;
    duration_ms: number;
}

export interface IndexErrorEvent {
    type: 'error';
    message: string;
    /** Repository-relative path, or 'system' for a whole-run failure. */
    file: string;
}
