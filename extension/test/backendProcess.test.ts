/**
 * Backend readiness polling and startup failure reporting.
 *
 * The behaviour under test replaced a fixed 4500ms sleep followed by a request
 * to a non-existent endpoint. Both axios and the venv provisioning are mocked,
 * so nothing here spawns a process or touches the network.
 */
import { EventEmitter } from 'events';

jest.mock('axios');
jest.mock('../src/pythonEnv', () => {
    const actual = jest.requireActual('../src/pythonEnv');
    return {
        ...actual,
        ensureBackendEnv: jest.fn(),
    };
});
// Keep the real execFile (pythonEnv promisifies it at import time) and mock
// only spawn, which is what the controller uses.
jest.mock('child_process', () => ({
    ...jest.requireActual('child_process'),
    spawn: jest.fn(),
}));
jest.mock('../src/toolResolver', () => ({
    resolveOllama: jest.fn(async () => 'ollama'),
    pathPrefixFor: () => null,
    ToolResolutionError: class extends Error {},
}));

import axios from 'axios';
import { spawn } from 'child_process';
import { BackendController, BackendStatusEvent } from '../src/backendProcess';
import { ensureBackendEnv, BackendSetupError } from '../src/pythonEnv';
import { recorded, resetRecorded } from './vscodeStub';

const mockedAxios = axios as jest.Mocked<typeof axios>;
const mockedSpawn = spawn as unknown as jest.Mock;
const mockedEnsureEnv = ensureBackendEnv as jest.Mock;

/** A fake ChildProcess that behaves like a spawned uvicorn. */
class FakeChild extends EventEmitter {
    public stdout = new EventEmitter();
    public stderr = new EventEmitter();
    public exitCode: number | null = null;
    public killed = false;
    public signals: string[] = [];
    /** When false, kill() is ignored so escalation can be observed. */
    public diesOnSignal = true;

    kill(signal: string) {
        this.signals.push(signal);
        if (this.diesOnSignal) {
            this.killed = true;
            this.exitCode = 0;
            setImmediate(() => this.emit('exit', 0, signal));
        }
        return true;
    }
}

function makeContext() {
    return {
        extensionPath: '/ext',
        globalStorageUri: { fsPath: '/storage' },
        subscriptions: [],
    } as any;
}

describe('BackendController', () => {
    let child: FakeChild;

    beforeEach(() => {
        jest.clearAllMocks();
        resetRecorded();
        recorded.settings['codelens.startupTimeoutSeconds'] = 2;
        recorded.settings['codelens.port'] = 8000;

        child = new FakeChild();
        mockedSpawn.mockReturnValue(child);
        mockedEnsureEnv.mockResolvedValue({ pythonPath: '/storage/venv/bin/python', extensionRoot: '/ext' });
    });

    it('becomes ready as soon as /health answers, without waiting a fixed delay', async () => {
        let calls = 0;
        mockedAxios.get.mockImplementation(async () => {
            calls++;
            if (calls <= 3) { throw new Error('ECONNREFUSED'); }
            return { data: { ollama: true, index: true } } as any;
        });

        const controller = new BackendController(makeContext());
        const started = Date.now();
        const ok = await controller.start();

        expect(ok).toBe(true);
        expect(controller.getState()).toBe('ready');
        // Polling at ~250ms: four probes is roughly 750ms, nowhere near 4500ms.
        expect(Date.now() - started).toBeLessThan(3000);
        expect(calls).toBeGreaterThan(1);
    });

    it('only ever probes /health, never the bogus /健康 endpoint', async () => {
        mockedAxios.get.mockResolvedValue({ data: {} } as any);

        const controller = new BackendController(makeContext());
        await controller.start();

        const urls = mockedAxios.get.mock.calls.map((c) => String(c[0]));
        expect(urls.every((u) => u.endsWith('/health'))).toBe(true);
        expect(urls.some((u) => /[^\x00-\x7F]/.test(u))).toBe(false);
    });

    it('adopts a backend that is already serving the port', async () => {
        mockedAxios.get.mockResolvedValue({ data: { ollama: true } } as any);

        const controller = new BackendController(makeContext());
        await controller.start();

        expect(controller.getState()).toBe('ready');
        expect(mockedSpawn).not.toHaveBeenCalled();
    });

    it('reports a specific error when readiness times out', async () => {
        recorded.settings['codelens.startupTimeoutSeconds'] = 1;
        mockedAxios.get.mockRejectedValue(new Error('ECONNREFUSED'));

        const controller = new BackendController(makeContext());
        const events: BackendStatusEvent[] = [];
        controller.onDidChangeState((e) => events.push(e));

        const ok = await controller.start();

        expect(ok).toBe(false);
        expect(controller.getState()).toBe('failed');
        const failure = events.find((e) => e.state === 'failed');
        expect(failure?.detail).toMatch(/did not respond/i);
        expect(failure?.detail).toMatch(/health/);
        expect(recorded.errorMessages.join(' ')).toMatch(/startupTimeoutSeconds/);
    });

    it('surfaces a setup failure without spawning anything', async () => {
        mockedEnsureEnv.mockRejectedValue(
            new BackendSetupError('Python 3.10+ was not found.', 'Set "codelens.pythonPath".'),
        );

        const controller = new BackendController(makeContext());
        const ok = await controller.start();

        expect(ok).toBe(false);
        expect(controller.getState()).toBe('failed');
        expect(mockedSpawn).not.toHaveBeenCalled();
        expect(recorded.errorMessages[0]).toContain('Python 3.10+ was not found.');
        expect(recorded.errorMessages[0]).toContain('codelens.pythonPath');
    });

    it('reports a backend that exits during startup, quoting its stderr', async () => {
        mockedAxios.get.mockRejectedValue(new Error('ECONNREFUSED'));

        const controller = new BackendController(makeContext());
        const promise = controller.start();

        setTimeout(() => {
            child.stderr.emit('data', 'ModuleNotFoundError: No module named uvicorn\n');
            child.emit('exit', 1, null);
        }, 120);

        const ok = await promise;

        expect(ok).toBe(false);
        expect(controller.getState()).toBe('failed');
        const message = recorded.errorMessages.join(' ');
        expect(message).toMatch(/exited during startup/i);
        expect(message).toMatch(/exit code 1/);
        expect(message).toMatch(/ModuleNotFoundError/);
    });

    it('reports a crash after it was already ready', async () => {
        mockedAxios.get.mockResolvedValue({ data: {} } as any);
        mockedSpawn.mockReturnValue(child);

        // Force a real spawn by failing the initial adopt probe once.
        let first = true;
        mockedAxios.get.mockImplementation(async () => {
            if (first) { first = false; throw new Error('ECONNREFUSED'); }
            return { data: {} } as any;
        });

        const controller = new BackendController(makeContext());
        const events: BackendStatusEvent[] = [];
        controller.onDidChangeState((e) => events.push(e));
        await controller.start();
        expect(controller.getState()).toBe('ready');

        child.emit('exit', 139, null);
        await new Promise((r) => setImmediate(r));

        expect(controller.getState()).toBe('failed');
        expect(recorded.errorMessages.join(' ')).toMatch(/stopped unexpectedly/i);
    });

    it('does not run two start() calls concurrently', async () => {
        let first = true;
        mockedAxios.get.mockImplementation(async () => {
            if (first) { first = false; throw new Error('ECONNREFUSED'); }
            return { data: {} } as any;
        });

        const controller = new BackendController(makeContext());
        const [a, b] = await Promise.all([controller.start(), controller.start()]);

        expect(a).toBe(true);
        expect(b).toBe(true);
        expect(mockedSpawn).toHaveBeenCalledTimes(1);
    });

    it('stops with SIGINT when the child exits cleanly', async () => {
        let first = true;
        mockedAxios.get.mockImplementation(async () => {
            if (first) { first = false; throw new Error('ECONNREFUSED'); }
            return { data: {} } as any;
        });

        const controller = new BackendController(makeContext());
        await controller.start();
        await controller.stop();

        expect(child.signals).toEqual(['SIGINT']);
        expect(controller.getState()).toBe('stopped');
    });

    it('escalates to SIGTERM and SIGKILL when the child ignores signals', async () => {
        jest.setTimeout(20000);
        let first = true;
        mockedAxios.get.mockImplementation(async () => {
            if (first) { first = false; throw new Error('ECONNREFUSED'); }
            return { data: {} } as any;
        });

        const controller = new BackendController(makeContext());
        await controller.start();

        child.diesOnSignal = false;
        await controller.stop();

        expect(child.signals).toEqual(['SIGINT', 'SIGTERM', 'SIGKILL']);
        expect(controller.getState()).toBe('stopped');
    }, 20000);
});
