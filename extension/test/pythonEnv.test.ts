/**
 * Venv provisioning: when it rebuilds, how, and when it records success.
 *
 * execFile is replaced with a scripted fake, so nothing here runs Python. The
 * filesystem is real (a temp directory), because the stamp file and the venv
 * interpreter's presence are exactly what is under test.
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

type Call = { exe: string; args: string[]; cwd?: string };
const calls: Call[] = [];
/** Return a stderr string to make a matching call fail. */
let failWhen: (call: Call) => string | null = () => null;

jest.mock('child_process', () => {
    const actual = jest.requireActual('child_process');
    const { promisify: p } = jest.requireActual('util');
    const execFile: any = jest.fn();
    // pythonEnv promisifies execFile at import time; honour the same custom
    // promisify contract the real one has, resolving { stdout, stderr }.
    execFile[p.custom] = async (exe: string, args: string[], options: { cwd?: string } = {}) => {
        const call = { exe, args, cwd: options.cwd };
        calls.push(call);
        const stderr = failWhen(call);
        if (stderr) {
            throw Object.assign(new Error('command failed'), { stderr });
        }
        if (args.includes('-c') && args.some((a) => a.includes('sys.version_info'))) {
            return { stdout: '3.13\n', stderr: '' };
        }
        const venvAt = args.indexOf('venv');
        if (venvAt !== -1) {
            // Create the venv interpreter the way `python -m venv` would.
            const venvDir = args[args.length - 1];
            const exePath = process.platform === 'win32'
                ? path.join(venvDir, 'Scripts', 'python.exe')
                : path.join(venvDir, 'bin', 'python');
            fs.mkdirSync(path.dirname(exePath), { recursive: true });
            fs.writeFileSync(exePath, '');
        }
        return { stdout: '', stderr: '' };
    };
    return { ...actual, execFile };
});

import { ensureBackendEnv, venvPython } from '../src/pythonEnv';
import { resetRecorded } from './vscodeStub';

function setup() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'codelens-env-'));
    const extensionRoot = path.join(root, 'ext');
    const storage = path.join(root, 'storage');
    fs.mkdirSync(path.join(extensionRoot, 'backend'), { recursive: true });
    fs.writeFileSync(path.join(extensionRoot, 'backend', 'main.py'), '');
    fs.writeFileSync(path.join(extensionRoot, 'requirements.txt'), 'fastapi\n');
    const context = { extensionPath: extensionRoot, globalStorageUri: { fsPath: storage } } as any;
    return { context, extensionRoot, venvDir: path.join(storage, 'venv') };
}

describe('ensureBackendEnv', () => {
    beforeEach(() => {
        calls.length = 0;
        failWhen = () => null;
        resetRecorded();
    });

    it('builds the venv with --clear so a rebuild never inherits old packages', async () => {
        const { context, venvDir } = setup();
        await ensureBackendEnv(context);

        const venvCall = calls.find((c) => c.args.includes('venv'));
        expect(venvCall?.args).toEqual(expect.arrayContaining(['-m', 'venv', '--clear', venvDir]));
    });

    it('verifies the backend imports before recording the venv as good', async () => {
        const { context, extensionRoot, venvDir } = setup();
        await ensureBackendEnv(context);

        const verify = calls.find((c) => c.args.join(' ') === '-c import backend.main');
        expect(verify?.exe).toBe(venvPython(venvDir));
        expect(verify?.cwd).toBe(extensionRoot);
        expect(fs.existsSync(path.join(venvDir, '.codelens-stamp'))).toBe(true);
    });

    it('does not write the stamp when the backend fails to import, so the next start rebuilds', async () => {
        const { context, venvDir } = setup();
        failWhen = (c) => (c.args.includes('import backend.main')
            ? "ImportError: cannot import name 'ArgsKwargs' from 'pydantic_core._pydantic_core'"
            : null);

        await expect(ensureBackendEnv(context)).rejects.toThrow(/could not be imported/);
        expect(fs.existsSync(path.join(venvDir, '.codelens-stamp'))).toBe(false);

        // Next activation: the import works again, and the venv is rebuilt.
        failWhen = () => null;
        calls.length = 0;
        await ensureBackendEnv(context);
        expect(calls.some((c) => c.args.includes('--clear'))).toBe(true);
        expect(fs.existsSync(path.join(venvDir, '.codelens-stamp'))).toBe(true);
    });

    it('reuses an up-to-date venv without rebuilding it', async () => {
        const { context } = setup();
        await ensureBackendEnv(context);
        calls.length = 0;

        await ensureBackendEnv(context);
        expect(calls.some((c) => c.args.includes('venv'))).toBe(false);
    });
});
