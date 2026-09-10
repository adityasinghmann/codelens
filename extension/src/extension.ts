import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import axios from 'axios';
import { SearchPanelProvider } from './searchPanel';
import { ensureBackendEnv, BackendSetupError } from './pythonEnv';

let backendProcess: ChildProcess | undefined;
let statusBarItem: vscode.StatusBarItem;
let pollInterval: NodeJS.Timeout;

const PORT = 8000;
const API_URL = `http://127.0.0.1:${PORT}`;

export async function activate(context: vscode.ExtensionContext) {
    // 1. Boot Python Backend Context
    startBackendServer(context);

    // 2. Register Webview sidebar
    const provider = new SearchPanelProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("codelens.sidebar", provider)
    );

    // 3. Mount Status Bar precisely formatted
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "$(sync~spin) CodeLens starting...";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // 4. Command Registrations
    context.subscriptions.push(vscode.commands.registerCommand('codelens.search', () => {
        vscode.commands.executeCommand('codelens.sidebar.focus');
    }));

    context.subscriptions.push(vscode.commands.registerCommand('codelens.reindex', () => {
        provider.triggerReindex();
    }));

    context.subscriptions.push(vscode.commands.registerCommand('codelens.showStatus', () => {
        vscode.commands.executeCommand('codelens.sidebar.focus');
    }));

    // 5. Pre-flight initialization & UI checks
    setTimeout(async () => {
        try {
            const health = await axios.get(`${API_URL}/健康`, { timeout: 1000 }).catch(() => axios.get(`${API_URL}/health`));
            
            if (health.data && health.data.ollama === false) {
                const actionBtn = "Install Ollama";
                const installRes = await vscode.window.showWarningMessage(
                    "CodeLens Engine Offline: Ollama was not found on your system. This is strictly required for local vector generation.",
                    actionBtn
                );
                if (installRes === actionBtn) {
                    vscode.env.openExternal(vscode.Uri.parse("https://ollama.ai")); // or ollama.com depending on routing
                }
            }
        } catch (e) {
            console.error("CodeLens Backend Health Check Missed");
        }
        
        startPolling();
    }, 4500); // giving uvicorn maximum startup leeway
}

/**
 * Start the backend that ships inside this extension.
 *
 * The backend is resolved from the extension's own install directory, never
 * from the user's workspace: the `backend` package is bundled in the .vsix and
 * its dependencies live in an extension-owned virtual environment provisioned
 * on first activation.
 */
async function startBackendServer(context: vscode.ExtensionContext) {
    let env;
    try {
        env = await ensureBackendEnv(context);
    } catch (err) {
        reportSetupFailure(err);
        return;
    }

    backendProcess = spawn(
        env.pythonPath,
        ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', PORT.toString()],
        {
            // cwd is the extension root so that `backend.main` imports as a package.
            cwd: env.extensionRoot,
            detached: false,
            env: {
                ...process.env,
                OLLAMA_HOST: 'http://localhost:11434',
            }
        }
    );

    backendProcess.stdout?.on('data', (d) => console.log(`[CodeLens]: ${d}`));
    backendProcess.stderr?.on('data', (d) => console.error(`[CodeLens ERR]: ${d}`));
    backendProcess.on('error', (err) => {
        console.error(`[CodeLens] Backend failed to spawn: ${err.message}`);
        statusBarItem.text = '$(error) CodeLens: failed to start';
        vscode.window.showErrorMessage(
            `CodeLens could not start its backend process (${env.pythonPath}): ${err.message}`,
        );
    });
    backendProcess.on('exit', (code) => {
        console.warn(`[CodeLens] Backend exited with code ${code}`);
        statusBarItem.text = '$(database) CodeLens: Offline';
    });
}

/** Surface a BackendSetupError as a message that names the fix. */
function reportSetupFailure(err: unknown) {
    if (statusBarItem) {
        statusBarItem.text = '$(error) CodeLens: setup failed';
    }
    if (err instanceof BackendSetupError) {
        console.error(`[CodeLens] ${err.message}`);
        vscode.window.showErrorMessage(`${err.message} ${err.remedy}`);
    } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error(`[CodeLens] Backend setup failed: ${message}`);
        vscode.window.showErrorMessage(`CodeLens backend setup failed: ${message}`);
    }
}

function startPolling() {
    pollStatus();
    pollInterval = setInterval(pollStatus, 30000); // updates every 30s
}

async function pollStatus() {
    try {
        const res = await axios.get(`${API_URL}/status`);
        const chunks = res.data.indexed_chunks || 0;
        statusBarItem.text = `$(database) CodeLens: ${chunks} chunks`;
    } catch(e) {
        statusBarItem.text = `$(database) CodeLens: Offline`;
    }
}

export function deactivate() {
    if (pollInterval) clearInterval(pollInterval);
    if (backendProcess && !backendProcess.killed) {
        backendProcess.kill('SIGINT');
    }
}
