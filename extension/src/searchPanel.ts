import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import * as http from 'http';
import { apiUrl } from './config';
import { SseParser, parseJsonEvent, SseEvent } from './sseParser';
import { IndexEvent, QueryResponse, IndexRequest, QueryRequest } from './apiTypes';
import { resolveInsideRoot } from './paths';

export class SearchPanelProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;
        webviewView.webview.options = { 
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this.getHtmlForWebview();

        // 1. Process Messages dispatched from panel.html WebView runtime
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'query':
                    this.handleQuery(data.text, data.explain);
                    break;
                case 'jumpTo':
                    this.handleJumpTo(data.file, data.line);
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

    private async handleQuery(text: string, explain: boolean) {
        try {
            // Native Axios bindings directly parsing localhost backend
            // Scope the search to the open workspace so a backend that has
            // indexed several repositories returns this one's code.
            const workspace = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
            const body: QueryRequest = { query: text, top_k: 8, explain, repo_path: workspace };
            const res = await axios.post<QueryResponse>(`${apiUrl()}/query`, body);
            this._view?.webview.postMessage({
                type: 'results',
                data: res.data.results,
                explain_text: res.data.explain_text,
                query_ms: res.data.query_ms,
                total_indexed: res.data.total_indexed
            });
        } catch (error: any) {
             this._view?.webview.postMessage({ type: 'error', message: error.message });
        }
    }

    private async handleReindex() {
        const workspaceDir = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
        if (!workspaceDir) {
            this._view?.webview.postMessage({ type: 'error', message: 'No valid workspace currently open.' });
            return;
        }

        const body: IndexRequest = { repo_path: workspaceDir, force_reindex: true };
        const payload = JSON.stringify(body);
        const parser = new SseParser();
        // Tracks whether the backend told us it finished. If the socket closes
        // without a 'complete' event, the index was truncated and the user must
        // hear about it rather than watching a progress bar stall forever.
        let sawComplete = false;

        const req = http.request(`${apiUrl()}/index`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload)
            }
        }, (res) => {
            // A non-2xx response is a JSON error body, not an event stream.
            if (res.statusCode && (res.statusCode < 200 || res.statusCode >= 300)) {
                let body = '';
                res.setEncoding('utf-8');
                res.on('data', (c) => { body += c; });
                res.on('end', () => {
                    let detail = body.trim();
                    try {
                        const parsedBody = JSON.parse(body);
                        detail = parsedBody.detail ?? parsedBody.error ?? detail;
                    } catch {
                        // Not JSON; fall back to the raw body text.
                    }
                    this.postError(`Indexing failed (HTTP ${res.statusCode}): ${detail || 'no detail returned'}`);
                });
                return;
            }

            res.setEncoding('utf-8');
            res.on('data', (chunk) => {
                for (const event of parser.push(chunk)) {
                    if (this.dispatchIndexEvent(event)) {
                        sawComplete = true;
                    }
                }
            });

            res.on('end', () => {
                // Flush any final event the server did not terminate with a
                // blank line before closing.
                for (const event of parser.flush()) {
                    if (this.dispatchIndexEvent(event)) {
                        sawComplete = true;
                    }
                }
                if (!sawComplete) {
                    this.postError('The indexing stream ended before the backend reported completion. The index may be incomplete - try re-indexing.');
                }
            });

            res.on('error', (err) => {
                this.postError(`The indexing stream failed: ${err.message}`);
            });
        });

        req.on('error', (err) => {
            this.postError(`Could not reach the CodeLens backend at ${apiUrl()}: ${err.message}`);
        });

        req.write(payload);
        req.end();
    }

    /**
     * Turn one SSE event into a webview message.
     * Returns true if this was the terminal 'complete' event.
     */
    private dispatchIndexEvent(event: SseEvent): boolean {
        const parsed = parseJsonEvent<IndexEvent>(event);
        if (!parsed.ok) {
            // Log rather than swallow: a malformed payload is a backend bug and
            // silently dropping it is what hid the old parser's data loss.
            console.error(`[CodeLens] Discarding malformed SSE payload (${parsed.failure.error}): ${parsed.failure.data}`);
            return false;
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
                this._view?.webview.postMessage({
                    type: 'indexProgress',
                    file: message.file,
                    pct,
                    processedFiles: done,
                    totalFiles: total,
                    processedChunks: Number(message.processed_chunks) || 0,
                });
                return false;
            }
            case 'complete':
                this._view?.webview.postMessage({
                    type: 'status',
                    chunks: message.total_chunks,
                    watching: true
                });
                return true;
            case 'error':
                this.postError(message.file && message.file !== 'system'
                    ? `${message.file}: ${message.message}`
                    : message.message);
                return false;
            default: {
                // Unreachable for the contract in apiTypes.ts - TypeScript
                // narrows `message` to never here, which is the exhaustiveness
                // check. Kept as a runtime guard against a backend newer than
                // the extension emitting an event type we do not know yet.
                const unknown = message as { type?: unknown };
                console.warn(`[CodeLens] Ignoring unknown index event type: ${JSON.stringify(unknown.type)}`);
                return false;
            }
        }
    }

    private postError(message: string) {
        this._view?.webview.postMessage({ type: 'error', message });
    }

    private async handleJumpTo(file: string, line: number) {
        const workspace = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
        if (!workspace) return;

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
            return fs.readFileSync(htmlPath.fsPath, 'utf-8');
        } catch (err: any) {
            const message = `CodeLens could not load its sidebar UI from ${htmlPath.fsPath}: ${err.message}`;
            vscode.window.showErrorMessage(`${message} Reinstall the CodeLens extension.`);
            return `<!DOCTYPE html><html><body style="font-family: var(--vscode-font-family); padding: 16px;">
                <p>CodeLens failed to load its interface.</p>
                <p style="opacity:.7">${message}</p></body></html>`;
        }
    }
}
