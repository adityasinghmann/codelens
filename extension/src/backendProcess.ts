/**
 * Owns the lifetime of the bundled backend process.
 *
 * Responsibilities, all of which the previous inline implementation got wrong:
 *  - wait for readiness by polling /health, not by sleeping a fixed 4.5s
 *  - distinguish "failed to spawn", "exited during startup", "never became
 *    ready" and "crashed mid-session", and report each one specifically
 *  - shut down for real: SIGINT, then SIGTERM after a grace period, then
 *    SIGKILL, because uvicorn's reloader-less child does not always honour the
 *    first signal (and Windows has no signals at all - see stop()).
 */
import * as vscode from 'vscode';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';
import axios from 'axios';
import { apiUrl, getStartupTimeoutMs, getOllamaHost, getOllamaSetting, getPort } from './config';
import { ensureBackendEnv, BackendSetupError } from './pythonEnv';
import { resolveOllama, pathPrefixFor, ToolResolutionError } from './toolResolver';

/** How often to re-probe /health while waiting for startup. */
const POLL_INTERVAL_MS = 250;
/**
 * Per-probe HTTP timeout. The server is on loopback, but /health itself probes
 * Ollama with its own timeout, so this must comfortably exceed that or a
 * healthy backend with Ollama down would look unreachable.
 */
const PROBE_TIMEOUT_MS = 2500;
/** Grace period between SIGINT and SIGTERM, and between SIGTERM and SIGKILL. */
const SIGNAL_GRACE_MS = 2000;

export type BackendState = 'stopped' | 'starting' | 'ready' | 'failed';

export interface BackendStatusEvent {
    state: BackendState;
    /** Human-readable detail, present on 'failed'. */
    detail?: string;
    /** Actionable next step, present on 'failed' when one is known. */
    remedy?: string;
}

/** Probe /health once. Returns the payload, or null if it did not answer. */
export async function probeHealth(): Promise<any | null> {
    try {
        const res = await axios.get(`${apiUrl()}/health`, { timeout: PROBE_TIMEOUT_MS });
        return res.data ?? {};
    } catch {
        return null;
    }
}

export class BackendController {
    private child: ChildProcess | undefined;
    private state: BackendState = 'stopped';
    /** Set while stop() is running so the exit handler stays quiet. */
    private shuttingDown = false;
    /** Guards against two concurrent start() calls racing on one port. */
    private starting: Promise<boolean> | undefined;
    private stderrTail: string[] = [];

    private readonly emitter = new vscode.EventEmitter<BackendStatusEvent>();
    /** Fires on every state transition. */
    public readonly onDidChangeState = this.emitter.event;

    constructor(private readonly context: vscode.ExtensionContext) {}

    public getState(): BackendState {
        return this.state;
    }

    private setState(state: BackendState, detail?: string, remedy?: string) {
        this.state = state;
        this.emitter.fire({ state, detail, remedy });
    }

    /**
     * Provision the environment, spawn uvicorn and wait until /health answers.
     * Resolves true once the backend is serving, false on any failure (the
     * failure is reported through onDidChangeState and a notification).
     */
    public async start(): Promise<boolean> {
        if (this.state === 'ready') {
            return true;
        }
        if (this.starting) {
            return this.starting;
        }
        this.starting = this.doStart().finally(() => {
            this.starting = undefined;
        });
        return this.starting;
    }

    private async doStart(): Promise<boolean> {
        this.setState('starting');
        this.stderrTail = [];

        let env;
        try {
            env = await ensureBackendEnv(this.context);
        } catch (err) {
            return this.fail(err);
        }

        // If something is already serving on the port (a leftover process, or a
        // backend the developer started by hand), adopt it rather than fighting
        // over the port.
        if (await probeHealth()) {
            this.setState('ready');
            return true;
        }

        let exited: { code: number | null; signal: NodeJS.Signals | null } | undefined;

        // Locating Ollama is best-effort at spawn time: the backend talks to it
        // over HTTP, so a missing CLI is not fatal here. Finding it lets us put
        // its directory on PATH, which matters when VS Code was launched from a
        // GUI and did not inherit a login shell's PATH. /health reports the
        // real reachability, and activate() warns from that.
        const childEnv: NodeJS.ProcessEnv = { ...process.env, OLLAMA_HOST: getOllamaHost() };
        try {
            const ollama = await resolveOllama(getOllamaSetting());
            const prefix = pathPrefixFor(ollama);
            if (prefix) {
                childEnv.PATH = `${prefix}${path.delimiter}${process.env.PATH ?? ''}`;
            }
        } catch (err) {
            const detail = err instanceof ToolResolutionError ? `${err.message} ${err.remedy}` : String(err);
            console.warn(`[CodeLens] ${detail}`);
        }

        try {
            this.child = spawn(
                env.pythonPath,
                ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(getPort())],
                {
                    cwd: env.extensionRoot,
                    detached: false,
                    env: childEnv,
                },
            );
        } catch (err) {
            return this.fail(
                new BackendSetupError(
                    `CodeLens could not launch the backend using ${env.pythonPath}: ${err instanceof Error ? err.message : String(err)}`,
                    'Check the "codelens.pythonPath" setting.',
                ),
            );
        }

        this.child.stdout?.on('data', (d) => console.log(`[CodeLens] ${d}`));
        this.child.stderr?.on('data', (d) => {
            const text = String(d);
            console.error(`[CodeLens ERR] ${text}`);
            // Keep a short tail so a startup failure can quote the real cause.
            this.stderrTail.push(text);
            if (this.stderrTail.length > 40) {
                this.stderrTail.shift();
            }
        });

        this.child.on('error', (err) => {
            this.fail(
                new BackendSetupError(
                    `CodeLens could not launch the backend process: ${err.message}`,
                    `Verify that ${env.pythonPath} is executable, or set "codelens.pythonPath".`,
                ),
            );
        });

        this.child.on('exit', (code, signal) => {
            exited = { code, signal };
            this.child = undefined;
            if (this.shuttingDown) {
                return;
            }
            if (this.state === 'ready') {
                // Crashed mid-session: surface it and offer a restart.
                this.reportCrash(code, signal);
            }
        });

        // Poll /health until it answers, the child dies, or we run out of time.
        const deadline = Date.now() + getStartupTimeoutMs();
        while (Date.now() < deadline) {
            if (exited) {
                return this.fail(
                    new BackendSetupError(
                        `The CodeLens backend exited during startup (${describeExit(exited.code, exited.signal)}).${this.stderrSummary()}`,
                        'Check the CodeLens output in the Developer Tools console for the full traceback.',
                    ),
                );
            }
            if (await probeHealth()) {
                this.setState('ready');
                return true;
            }
            await delay(POLL_INTERVAL_MS);
        }

        const waited = Math.round(getStartupTimeoutMs() / 1000);
        await this.stop();
        return this.fail(
            new BackendSetupError(
                `The CodeLens backend did not respond on ${apiUrl()}/health within ${waited}s.${this.stderrSummary()}`,
                'Increase "codelens.startupTimeoutSeconds" if the machine is slow, or check that nothing else is bound to the configured port.',
            ),
        );
    }

    /** Report a mid-session crash and offer to restart. */
    private reportCrash(code: number | null, signal: NodeJS.Signals | null) {
        const detail = `The CodeLens backend stopped unexpectedly (${describeExit(code, signal)}).`;
        this.setState('failed', detail, 'Restart the backend to resume searching.');
        vscode.window.showErrorMessage(detail, 'Restart Backend').then((choice) => {
            if (choice === 'Restart Backend') {
                void this.restart();
            }
        });
    }

    public async restart(): Promise<boolean> {
        await this.stop();
        return this.start();
    }

    /**
     * Terminate the backend and confirm it is actually gone.
     *
     * SIGINT is what uvicorn documents for a clean shutdown, but a wedged
     * worker can ignore it, so we escalate: SIGINT -> SIGTERM -> SIGKILL, each
     * with a grace period. On Windows none of these are real signals -
     * ChildProcess.kill() maps them onto TerminateProcess - so the escalation
     * collapses to a single effective kill, which is why we still confirm exit
     * rather than assuming it.
     */
    public async stop(): Promise<void> {
        const child = this.child;
        if (!child || child.exitCode !== null || child.killed) {
            this.child = undefined;
            this.shuttingDown = false;
            this.setState('stopped');
            return;
        }

        this.shuttingDown = true;
        const dead = new Promise<void>((resolve) => child.once('exit', () => resolve()));

        for (const signal of ['SIGINT', 'SIGTERM', 'SIGKILL'] as const) {
            try {
                child.kill(signal);
            } catch {
                // Already reaped between the check and the call.
                break;
            }
            const settled = await Promise.race([dead.then(() => true), delay(SIGNAL_GRACE_MS).then(() => false)]);
            if (settled) {
                break;
            }
            console.warn(`[CodeLens] Backend ignored ${signal}; escalating.`);
        }

        this.child = undefined;
        this.shuttingDown = false;
        this.setState('stopped');
    }

    public dispose() {
        void this.stop();
        this.emitter.dispose();
    }

    /** Report a startup failure once, as state + notification. */
    private fail(err: unknown): boolean {
        if (this.state === 'failed') {
            return false;
        }
        const isSetup = err instanceof BackendSetupError;
        const detail = err instanceof Error ? err.message : String(err);
        const remedy = isSetup ? (err as BackendSetupError).remedy : undefined;

        console.error(`[CodeLens] ${detail}`);
        this.setState('failed', detail, remedy);
        vscode.window.showErrorMessage(remedy ? `${detail} ${remedy}` : detail);
        return false;
    }

    private stderrSummary(): string {
        const tail = this.stderrTail.join('').trim();
        if (!tail) {
            return '';
        }
        const lastLines = tail.split('\n').slice(-4).join(' ').trim();
        return ` Last output: ${lastLines}`;
    }
}

function describeExit(code: number | null, signal: NodeJS.Signals | null): string {
    if (signal) {
        return `killed by ${signal}`;
    }
    return `exit code ${code ?? 'unknown'}`;
}

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
