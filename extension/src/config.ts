/**
 * Single source of truth for everything the extension needs to talk to the
 * backend, and for every user-facing setting.
 *
 * Both extension.ts and searchPanel.ts import from here so the API base URL is
 * defined exactly once.
 */
import * as vscode from 'vscode';

export const DEFAULT_PORT = 8000;

/** The configured backend port (`codelens.port`). */
export function getPort(): number {
    const configured = vscode.workspace.getConfiguration('codelens').get<number>('port');
    if (typeof configured === 'number' && Number.isInteger(configured) && configured > 0 && configured < 65536) {
        return configured;
    }
    return DEFAULT_PORT;
}

/** Base URL of the local backend. Never remote — the backend binds 127.0.0.1. */
export function apiUrl(): string {
    return `http://127.0.0.1:${getPort()}`;
}

/** How long to wait for the backend to answer /health before giving up (ms). */
export function getStartupTimeoutMs(): number {
    const seconds = vscode.workspace.getConfiguration('codelens').get<number>('startupTimeoutSeconds');
    if (typeof seconds === 'number' && seconds > 0) {
        return Math.round(seconds * 1000);
    }
    return 30_000;
}

/** Explicit interpreter override (`codelens.pythonPath`), or '' when unset. */
export function getPythonSetting(): string {
    return (vscode.workspace.getConfiguration('codelens').get<string>('pythonPath') ?? '').trim();
}

/** Explicit Ollama binary override (`codelens.ollamaPath`), or '' when unset. */
export function getOllamaSetting(): string {
    return (vscode.workspace.getConfiguration('codelens').get<string>('ollamaPath') ?? '').trim();
}

/** Ollama server address handed to the backend process. */
export function getOllamaHost(): string {
    const configured = (vscode.workspace.getConfiguration('codelens').get<string>('ollamaHost') ?? '').trim();
    return configured || 'http://localhost:11434';
}
