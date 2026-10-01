/**
 * Multi-root workspace helpers.
 *
 * A workspace can hold several folders, and each is indexed as its own
 * repository. Searching means one query per folder and a merge; jumping to a
 * hit means knowing which folder its repository-relative path belongs to.
 *
 * Kept free of any `vscode` import so the logic can be unit tested directly.
 */
import * as path from 'path';
import { QueryResult } from './apiTypes';

/** A search hit tagged with the workspace folder it came from. */
export interface FolderHit extends QueryResult {
    /** Absolute root of the folder the hit belongs to; undefined when no folder is open. */
    root?: string;
    /** Folder display name, set only when there is more than one folder to tell apart. */
    folder?: string;
}

export interface FolderResults {
    root?: string;
    name?: string;
    results: QueryResult[];
}

/**
 * Merge per-folder results into one ranking.
 *
 * Scores are directly comparable across folders: every folder is searched by
 * the same backend with the same embedding model (the backend refuses a query
 * against an index built with a different one), and each score is a cosine
 * similarity.
 */
export function mergeFolderResults(perFolder: FolderResults[], limit: number): FolderHit[] {
    const labelFolders = perFolder.length > 1;
    const hits: FolderHit[] = [];
    for (const entry of perFolder) {
        for (const result of entry.results) {
            hits.push({ ...result, root: entry.root, folder: labelFolders ? entry.name : undefined });
        }
    }
    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, Math.max(0, limit));
}

/**
 * Pick the workspace root a jump-to-file request refers to.
 *
 * `requested` round-trips through the webview, so it is untrusted: it is
 * accepted only if it names one of the open workspace folders exactly. With no
 * root given (a result from a search made with no folder open), the first
 * folder is used, which is what jump-to-file always did before multi-root.
 */
export function selectWorkspaceRoot(workspaceRoots: string[], requested: string | undefined): string | null {
    if (requested === undefined || requested === '') {
        return workspaceRoots[0] ?? null;
    }
    const wanted = path.resolve(requested);
    return workspaceRoots.find((root) => path.resolve(root) === wanted) ?? null;
}
