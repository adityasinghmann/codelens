import * as vscode from 'vscode';
import axios from 'axios';
import { SearchPanelProvider } from './searchPanel';
import { BackendController, probeHealth } from './backendProcess';
import { apiUrl } from './config';

let statusBarItem: vscode.StatusBarItem;
let pollInterval: NodeJS.Timeout | undefined;
let backend: BackendController | undefined;

/** How often the status bar refreshes its chunk count once the backend is up. */
const STATUS_POLL_MS = 30000;

export async function activate(context: vscode.ExtensionContext) {
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = '$(sync~spin) CodeLens starting...';
    statusBarItem.command = 'codelens.showStatus';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    const provider = new SearchPanelProvider(context.extensionUri);
    context.subscriptions.push(
        // Keep the sidebar alive while collapsed, so a long re-index keeps its
        // progress and messages instead of being reset when the view is hidden.
        vscode.window.registerWebviewViewProvider('codelens.sidebar', provider, {
            webviewOptions: { retainContextWhenHidden: true },
        })
    );

    context.subscriptions.push(vscode.commands.registerCommand('codelens.search', () => {
        vscode.commands.executeCommand('codelens.sidebar.focus');
    }));

    context.subscriptions.push(vscode.commands.registerCommand('codelens.reindex', () => {
        provider.triggerReindex();
    }));

    context.subscriptions.push(vscode.commands.registerCommand('codelens.showStatus', () => {
        vscode.commands.executeCommand('codelens.sidebar.focus');
    }));

    context.subscriptions.push(vscode.commands.registerCommand('codelens.restartBackend', async () => {
        stopStatusPolling();
        statusBarItem.text = '$(sync~spin) CodeLens restarting...';
        await backend?.restart();
    }));

    backend = new BackendController(context);
    context.subscriptions.push(backend);

    // Reflect backend lifecycle in the status bar. A crash must not leave a
    // stale "N chunks" reading sitting there as if everything were fine.
    context.subscriptions.push(
        backend.onDidChangeState((event) => {
            switch (event.state) {
                case 'starting':
                    statusBarItem.text = '$(sync~spin) CodeLens starting...';
                    statusBarItem.tooltip = 'CodeLens is starting its local backend';
                    break;
                case 'ready':
                    statusBarItem.text = '$(database) CodeLens: ready';
                    statusBarItem.tooltip = 'CodeLens backend is running';
                    startStatusPolling();
                    break;
                case 'failed':
                    stopStatusPolling();
                    statusBarItem.text = '$(error) CodeLens: offline';
                    statusBarItem.tooltip = event.detail ?? 'CodeLens backend is not running';
                    break;
                case 'stopped':
                    stopStatusPolling();
                    statusBarItem.text = '$(circle-slash) CodeLens: stopped';
                    statusBarItem.tooltip = 'CodeLens backend is stopped';
                    break;
            }
        })
    );

    const started = await backend.start();
    if (started) {
        await checkOllama();
    }
}

/** Warn once if the backend is up but Ollama is not reachable. */
async function checkOllama() {
    const health = await probeHealth();
    if (health && health.ollama === false) {
        const install = 'Install Ollama';
        const choice = await vscode.window.showWarningMessage(
            'CodeLens cannot reach Ollama, which generates the embeddings it searches with. '
            + 'Start Ollama, or set "codelens.ollamaHost" if it runs elsewhere.',
            install
        );
        if (choice === install) {
            vscode.env.openExternal(vscode.Uri.parse('https://ollama.com'));
        }
    }
}

function startStatusPolling() {
    stopStatusPolling();
    void pollStatus();
    pollInterval = setInterval(() => void pollStatus(), STATUS_POLL_MS);
}

function stopStatusPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = undefined;
    }
}

async function pollStatus() {
    // Only meaningful while the controller believes the backend is up; the
    // state machine owns the offline/failed text.
    if (backend?.getState() !== 'ready') {
        return;
    }
    try {
        const res = await axios.get(`${apiUrl()}/status`, { timeout: 5000 });
        const chunks = res.data.indexed_chunks ?? 0;
        statusBarItem.text = `$(database) CodeLens: ${chunks} chunks`;
    } catch {
        statusBarItem.text = '$(warning) CodeLens: not responding';
        statusBarItem.tooltip = 'The CodeLens backend stopped answering /status';
    }
}

export async function deactivate() {
    stopStatusPolling();
    await backend?.stop();
}
