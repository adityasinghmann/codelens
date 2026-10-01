import * as vscode from 'vscode';
import * as fs from 'fs';
import * as crypto from 'crypto';
import axios from 'axios';
import * as http from 'http';
import { apiUrl } from './config';
import { SseParser, parseJsonEvent, SseEvent } from './sseParser';
import { IndexEvent, IndexCompleteEvent, QueryResponse, IndexRequest, QueryRequest } from './apiTypes';
import { resolveInsideRoot } from './paths';
import { mergeFolderResults, selectWorkspaceRoot, FolderResults } from './multiRoot';

/** Results shown per search, across all workspace folders. Within the backend's 1..20. */
const RESULTS_PER_QUERY = 8;

export class SearchPanelProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    /** Resolves once the current webview's script has loaded and can receive messages. */
    private _ready: Promise<void> = Promise.resolve();
    private _markReady: () => void = () => undefined;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;
        this._ready = new Promise((resolve) => { this._markReady = resolve; });
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this.getHtmlForWebview();

        webviewView.onDidDispose(() => {
            if (this._view === webviewView) {
                this._view = undefined;
            }
        });

        // 1. Process Messages dispatched from panel.html WebView runtime
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'ready':
                    this._markReady();
                    break;
                case 'copy':
                    await vscode.env.clipboard.writeText(String(data.text ?? ''));
                    vscode.window.setStatusBarMessage('CodeLens: copied to clipboard', 2000);
                    break;
                case 'query':
                    this.handleQuery(data.text, data.explain);
                    break;
                case 'jumpTo':
                    this.handleJumpTo(data.root, data.file, data.line);
                    break;
                case 'reindex':
                    this.handleReindex();
                    break;
            }
        });
    }

    public triggerReindex() {
        this.handleReindex();
    }

    /**
     * Make sure the sidebar exists and its script is listening.
     *
     * "Re-index Workspace" can run from the Command Palette or the editor
     * title bar before the sidebar was ever opened. Every progress update and
     * error then went to a view that did not exist and was silently dropped.
     */
    private async ensureView(): Promise<void> {
        if (!this._view) {
            await vscode.commands.executeCommand('codelens.sidebar.focus');
        }
        if (this._view) {
            const timeout = new Promise<void>((resolve) => setTimeout(resolve, 5000));
            await Promise.race([this._ready, timeout]);
        }
    }

    /**
     * Search every workspace folder and merge the hits.
     *
     * Each folder is its own repository to the backend, so a multi-root
     * workspace needs one query per folder. Folders that have not been indexed
     * yet are skipped; the search fails only if no folder could be searched.
     */
    private async handleQuery(text: string, explain: boolean) {
        const folders = vscode.workspace.workspaceFolders ?? [];
        // With no folder open, one unscoped query searches the most recently
        // indexed repository, as before.
        const targets: (vscode.WorkspaceFolder | undefined)[] = folders.length ? [...folders] : [undefined];
        // Explaining is a local-LLM call per request. Across several folders,
        // ask only the folder holding the best hit, once the ranking is known.
        const explainInline = explain && targets.length === 1;

        const query = (folder: vscode.WorkspaceFolder | undefined, withExplain: boolean) => {
            const body: QueryRequest = {
                query: text,
                top_k: RESULTS_PER_QUERY,
                explain: withExplain,
                repo_path: folder?.uri.fsPath,
            };
            return axios.post<QueryResponse>(`${apiUrl()}/query`, body);
        };

        const settled = await Promise.allSettled(targets.map((folder) => query(folder, explainInline)));

        const perFolder: FolderResults[] = [];
        let queryMs = 0;
        let totalIndexed = 0;
        let explainText: string | null = null;
        let firstError: unknown;
        // Folders that could not be searched. Skipping them silently hid, for
        // example, a folder whose index needs rebuilding after a model change.
        const notices: string[] = [];

        settled.forEach((outcome, i) => {
            if (outcome.status === 'rejected') {
                firstError ??= outcome.reason;
                const name = targets[i]?.name;
                if (name) {
                    notices.push(`${name}: ${describeRequestError(outcome.reason)}`);
                }
                return;
            }
            const folder = targets[i];
            perFolder.push({ root: folder?.uri.fsPath, name: folder?.name, results: outcome.value.data.results });
            // The queries ran in parallel, so the slowest one is the wall time.
            queryMs = Math.max(queryMs, outcome.value.data.query_ms);
            totalIndexed += outcome.value.data.total_indexed;
            if (explainInline) {
                explainText = outcome.value.data.explain_text;
            }
        });

        if (perFolder.length === 0) {
            this.postError(describeRequestError(firstError));
            return;
        }

        const hits = mergeFolderResults(perFolder, RESULTS_PER_QUERY);

        if (explain && !explainInline && hits.length > 0) {
            const best = folders.find((f) => f.uri.fsPath === hits[0].root);
            try {
                explainText = (await query(best, true)).data.explain_text;
            } catch {
                explainText = '(Explain mode failed for this search.)';
            }
        }

        this.post({
            type: 'results',
            data: hits,
            explain_text: explainText,
            query_ms: queryMs,
            total_indexed: totalIndexed,
            notices,
        });
    }

    /**
     * Re-index the workspace. With several folders, ask which one - or all -
     * rather than silently indexing only the first.
     */
    private async handleReindex() {
        // The command can run before the sidebar was ever opened; open it
        // first so progress and errors have somewhere to go.
        await this.ensureView();

        const folders = vscode.workspace.workspaceFolders ?? [];
        if (folders.length === 0) {
            this.postError('No valid workspace currently open.');
            return;
        }

        let targets: readonly vscode.WorkspaceFolder[] = folders;
        if (folders.length > 1) {
            const pick = await vscode.window.showQuickPick(
                [
                    { label: 'All workspace folders', description: `${folders.length} folders`, targets: [...folders] },
                    ...folders.map((f) => ({ label: f.name, description: f.uri.fsPath, targets: [f] })),
                ],
                { placeHolder: 'Re-index which workspace folder?' },
            );
            if (!pick) {
                this.post({ type: 'indexCancelled' });
                return;
            }
            targets = pick.targets;
        }

        // Shows the progress view whatever triggered the run - the sidebar
        // button, the Command Palette or the editor title bar.
        this.post({ type: 'indexStarted' });

        let stored = 0;
        let skipped = 0;
        let failed = 0;
        for (const folder of targets) {
            const done = await this.indexFolder(folder, targets.length > 1 ? folder.name : undefined);
            if (done === null) {
                return; // The failure has already been shown.
            }
            stored += Number(done.stored) || 0;
            skipped += Number(done.skipped) || 0;
            failed += Number(done.failed) || 0;
        }

        // Report what is actually searchable. total_chunks counts what was
        // parsed, so with Ollama down the sidebar used to announce every chunk
        // as indexed while none had been stored.
        const searchable = stored + skipped;
        if (failed > 0 && searchable === 0) {
            this.postError(
                `None of the ${failed} chunks could be embedded, so nothing is searchable yet. `
                + 'Check that Ollama is running and the embedding model is pulled '
                + '(by default: ollama pull nomic-embed-text), then re-index.',
            );
            return;
        }
        this.post({ type: 'status', chunks: searchable, failed, watching: true });
    }

    /**
     * Index one folder, streaming progress to the webview.
     * Resolves to the backend's completion summary, or null if indexing failed.
     */
    private indexFolder(folder: vscode.WorkspaceFolder, label: string | undefined): Promise<IndexCompleteEvent | null> {
        const body: IndexRequest = { repo_path: folder.uri.fsPath, force_reindex: true };
        const payload = JSON.stringify(body);
        const parser = new SseParser();

        return new Promise((resolve) => {
            // Set by the backend's 'complete' event. If the socket closes
            // without one, the index was truncated and the user must hear
            // about it rather than watching a progress bar stall forever.
            let completion: IndexCompleteEvent | undefined;
            // A whole-run error is followed by the stream closing; do not bury
            // the real cause under a generic "ended early" message.
            let sawFatalError = false;
            let settled = false;
            const finish = (value: IndexCompleteEvent | null) => {
                if (!settled) {
                    settled = true;
                    resolve(value);
                }
            };

            const handle = (event: SseEvent) => {
                const outcome = this.dispatchIndexEvent(event, label);
                if (outcome.complete) {
                    completion = outcome.complete;
                }
                if (outcome.fatal) {
                    sawFatalError = true;
                }
            };

            const req = http.request(`${apiUrl()}/index`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(payload)
                }
            }, (res) => {
                // A non-2xx response is a JSON error body, not an event stream.
                if (res.statusCode && (res.statusCode < 200 || res.statusCode >= 300)) {
                    let text = '';
                    res.setEncoding('utf-8');
                    res.on('data', (c) => { text += c; });
                    res.on('end', () => {
                        let detail = text.trim();
                        try {
                            const parsedBody = JSON.parse(text);
                            detail = parsedBody.detail ?? parsedBody.error ?? detail;
                        } catch {
                            // Not JSON; fall back to the raw body text.
                        }
                        const what = label ? `Indexing ${label} failed` : 'Indexing failed';
                        this.postError(`${what} (HTTP ${res.statusCode}): ${detail || 'no detail returned'}`);
                        finish(null);
                    });
                    return;
                }

                res.setEncoding('utf-8');
                res.on('data', (chunk) => {
                    for (const event of parser.push(chunk)) {
                        handle(event);
                    }
                });

                res.on('end', () => {
                    // Flush any final event the server did not terminate with a
                    // blank line before closing.
                    for (const event of parser.flush()) {
                        handle(event);
                    }
                    if (!completion) {
                        if (!sawFatalError) {
                            this.postError('The indexing stream ended before the backend reported completion. The index may be incomplete - try re-indexing.');
                        }
                        finish(null);
                        return;
                    }
                    finish(completion);
                });

                res.on('error', (err) => {
                    this.postError(`The indexing stream failed: ${err.message}`);
                    finish(null);
                });
            });

            req.on('error', (err) => {
                this.postError(`Could not reach the CodeLens backend at ${apiUrl()}: ${err.message}`);
                finish(null);
            });

            req.write(payload);
            req.end();
        });
    }

    /**
     * Turn one SSE event into a webview message.
     * Returns the completion summary when this was the terminal 'complete'
     * event, and whether it was a whole-run failure.
     */
    private dispatchIndexEvent(event: SseEvent, label: string | undefined): { complete?: IndexCompleteEvent; fatal?: boolean } {
        const parsed = parseJsonEvent<IndexEvent>(event);
        if (!parsed.ok) {
            // Log rather than swallow: a malformed payload is a backend bug and
            // silently dropping it is what hid the old parser's data loss.
            console.error(`[CodeLens] Discarding malformed SSE payload (${parsed.failure.error}): ${parsed.failure.data}`);
            return {};
        }

        const message = parsed.value;
        switch (message.type) {
            case 'progress': {
                // files/files, not chunks/files: the old formula divided one
                // file's chunk count by the repo's file count, which is
                // dimensionally meaningless and routinely exceeded 100%.
                const total = Math.max(1, Number(message.total_files) || 0);
                const done = Number(message.processed_files) || 0;
                const pct = Math.max(0, Math.min(100, (done / total) * 100));
                this.post({
                    type: 'indexProgress',
                    file: label ? `${label}/${message.file}` : message.file,
                    pct,
                    processedFiles: done,
                    totalFiles: total,
                    processedChunks: Number(message.processed_chunks) || 0,
                });
                return {};
            }
            case 'complete':
                return { complete: message };
            case 'error': {
                const fatal = !message.file || message.file === 'system';
                if (fatal) {
                    this.postError(label ? `${label}: ${message.message}` : message.message);
                } else {
                    // One file failing does not stop the run: report it under
                    // the progress bar instead of replacing the progress view.
                    this.post({
                        type: 'indexFileError',
                        file: label ? `${label}/${message.file}` : message.file,
                        message: message.message,
                    });
                }
                return { fatal };
            }
            default: {
                // Unreachable for the contract in apiTypes.ts - TypeScript
                // narrows `message` to never here, which is the exhaustiveness
                // check. Kept as a runtime guard against a backend newer than
                // the extension emitting an event type we do not know yet.
                const unknown = message as { type?: unknown };
                console.warn(`[CodeLens] Ignoring unknown index event type: ${JSON.stringify(unknown.type)}`);
                return {};
            }
        }
    }

    private post(message: Record<string, unknown>) {
        this._view?.webview.postMessage(message);
    }

    /** Show an error in the sidebar, or as a notification when there is no sidebar. */
    private postError(message: string) {
        if (this._view) {
            this.post({ type: 'error', message });
        } else {
            vscode.window.showErrorMessage(`CodeLens: ${message}`);
        }
    }

    private async handleJumpTo(root: string | undefined, file: string, line: number) {
        const roots = (vscode.workspace.workspaceFolders ?? []).map((f) => f.uri.fsPath);
        // `root` comes back from the webview, so it must name an open folder.
        const workspace = selectWorkspaceRoot(roots, root);
        if (!workspace) {
            if (root) {
                vscode.window.showErrorMessage(`CodeLens refused to open "${file}": "${root}" is not an open workspace folder.`);
            }
            return;
        }

        // Containment check. `file` comes from a search result, which comes
        // from the index, which is built from files on disk - but it is still
        // untrusted input by the time it round-trips through the webview, and
        // path.join happily resolves "../../../etc/passwd".
        const absolutePath = resolveInsideRoot(workspace, file);
        if (absolutePath === null) {
            console.error(`[CodeLens] Refusing to open a path outside the workspace: ${file}`);
            vscode.window.showErrorMessage(
                `CodeLens refused to open "${file}" because it resolves outside the workspace.`
            );
            return;
        }

        try {
            const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(absolutePath));
            const editor = await vscode.window.showTextDocument(doc);

            // Adjust to VS Code's 0-indexed positions.
            const vsLine = Math.max(0, line - 1);
            const range = new vscode.Range(vsLine, 0, vsLine, 0);

            editor.selection = new vscode.Selection(range.start, range.end);
            editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
        } catch (e: any) {
            vscode.window.showErrorMessage(`Cannot open file: ${e.message}`);
        }
    }

    private getHtmlForWebview() {
        // The webview markup lives at <extensionRoot>/extension/media/panel.html
        // (media/ at the root holds only the activity-bar icon).
        const htmlPath = vscode.Uri.joinPath(this._extensionUri, 'extension', 'media', 'panel.html');
        try {
            // A fresh nonce per load: the page's Content-Security-Policy runs
            // only the script carrying it, so injected markup cannot execute.
            const nonce = crypto.randomBytes(16).toString('base64');
            return fs.readFileSync(htmlPath.fsPath, 'utf-8').replace(/\{\{nonce\}\}/g, nonce);
        } catch (err: any) {
            const message = `CodeLens could not load its sidebar UI from ${htmlPath.fsPath}: ${err.message}`;
            vscode.window.showErrorMessage(`${message} Reinstall the CodeLens extension.`);
            return `<!DOCTYPE html><html><body style="font-family: var(--vscode-font-family); padding: 16px;">
                <p>CodeLens failed to load its interface.</p>
                <p style="opacity:.7">${message}</p></body></html>`;
        }
    }
}

/** The backend's own explanation for a failed request, when it sent one. */
function describeRequestError(err: unknown): string {
    const e = err as { response?: { data?: { detail?: unknown } }; message?: string } | undefined;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) {
        return detail;
    }
    return e?.message ?? String(err);
}
