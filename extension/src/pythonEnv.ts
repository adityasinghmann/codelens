/**
 * Locates a Python interpreter and provisions the private virtual environment
 * that the bundled backend runs in.
 *
 * The backend ships inside the .vsix. Its dependencies do not - wheels for
 * tree-sitter and numpy are platform- and Python-version-specific, so they are
 * installed on first activation into a venv under the extension's global
 * storage directory. That directory is owned by the extension, survives
 * upgrades, and is removed when the extension is uninstalled.
 *
 * Every failure path here throws BackendSetupError with a `remedy` that names
 * the concrete fix. Nothing in this module fails silently or hangs.
 */
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as crypto from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getPythonSetting } from './config';

const execFileAsync = promisify(execFile);

/** An error the user can act on. `remedy` is shown alongside the message. */
export class BackendSetupError extends Error {
    constructor(message: string, public readonly remedy: string) {
        super(message);
        this.name = 'BackendSetupError';
    }
}

export interface BackendEnv {
    /** Interpreter inside the provisioned venv - the one that runs uvicorn. */
    pythonPath: string;
    /** Directory containing the `backend` package; also the process cwd. */
    extensionRoot: string;
}

const MIN_PYTHON: readonly [number, number] = [3, 10];

/**
 * An interpreter candidate is an argv prefix, not just a path, so the Windows
 * `py -3` launcher can be a first-class option alongside plain executables.
 */
export type PythonCommand = string[];

/** Interpreter names to look for on PATH, in preference order. */
function pathCandidates(): PythonCommand[] {
    if (process.platform === 'win32') {
        return [['py', '-3'], ['python'], ['python3']];
    }
    return [['python3.12'], ['python3.11'], ['python3.10'], ['python3'], ['python']];
}

/** Platform-specific default install locations, tried after PATH. */
function defaultCandidates(): PythonCommand[] {
    const home = os.homedir();
    switch (process.platform) {
        case 'darwin':
            return [
                ['/opt/homebrew/bin/python3'],
                ['/usr/local/bin/python3.12'],
                ['/usr/local/bin/python3.11'],
                ['/usr/local/bin/python3.10'],
                ['/usr/local/bin/python3'],
                ['/usr/bin/python3'],
            ];
        case 'win32': {
            const roots = [
                path.join(process.env.LOCALAPPDATA ?? path.join(home, 'AppData', 'Local'), 'Programs', 'Python'),
                'C:\\',
                path.join(process.env.ProgramFiles ?? 'C:\\Program Files'),
            ];
            const found: PythonCommand[] = [];
            for (const root of roots) {
                let entries: string[];
                try {
                    entries = fs.readdirSync(root);
                } catch {
                    continue;
                }
                // Newest first, so Python313 beats Python310.
                for (const entry of entries.filter((e) => /^Python3\d+$/i.test(e)).sort().reverse()) {
                    found.push([path.join(root, entry, 'python.exe')]);
                }
            }
            return found;
        }
        default:
            return [['/usr/local/bin/python3'], ['/usr/bin/python3']];
    }
}

/**
 * Return the "major.minor" version reported by a command, or null if it cannot
 * be run or is older than the minimum. Uses a real subprocess probe rather than
 * trusting the name on disk.
 */
async function probeInterpreter(command: PythonCommand): Promise<string | null> {
    const [exe, ...prefix] = command;
    if (path.isAbsolute(exe) && !fs.existsSync(exe)) {
        return null;
    }
    try {
        const { stdout } = await execFileAsync(
            exe,
            [...prefix, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'],
            { timeout: 10000 },
        );
        const version = stdout.trim();
        const [major, minor] = version.split('.').map((n) => parseInt(n, 10));
        if (major > MIN_PYTHON[0] || (major === MIN_PYTHON[0] && minor >= MIN_PYTHON[1])) {
            return version;
        }
        return null;
    } catch {
        return null;
    }
}

/**
 * Find a Python >= 3.10 to build the venv from, in the documented order:
 * the `codelens.pythonPath` setting, then PATH, then platform defaults.
 *
 * An explicit setting that does not work is reported as such rather than
 * silently falling through to a different interpreter.
 */
export async function resolveBasePython(explicit: string): Promise<PythonCommand> {
    const min = `${MIN_PYTHON[0]}.${MIN_PYTHON[1]}`;

    if (explicit) {
        if (await probeInterpreter([explicit])) {
            return [explicit];
        }
        throw new BackendSetupError(
            `The interpreter configured in "codelens.pythonPath" is not usable: ${explicit} (it must exist and be Python ${min} or newer).`,
            'Correct "codelens.pythonPath", or clear it to let CodeLens search PATH.',
        );
    }

    const tried: string[] = [];
    for (const candidate of [...pathCandidates(), ...defaultCandidates()]) {
        tried.push(candidate.join(' '));
        if (await probeInterpreter(candidate)) {
            return candidate;
        }
    }

    throw new BackendSetupError(
        `CodeLens could not find Python ${min} or newer on ${process.platform} (tried: ${tried.join(', ')}).`,
        'Install Python 3.10+ and make sure it is on your PATH, or set "codelens.pythonPath" to the interpreter.',
    );
}

/** Path to the interpreter inside a venv directory, per platform. */
export function venvPython(venvDir: string): string {
    return process.platform === 'win32'
        ? path.join(venvDir, 'Scripts', 'python.exe')
        : path.join(venvDir, 'bin', 'python');
}

/**
 * Identity of a provisioned venv: the requirements content plus the base
 * interpreter version and platform. If any of those change, the venv is rebuilt
 * rather than silently reused with the wrong wheels.
 */
function envStamp(requirementsText: string, pythonVersion: string): string {
    return crypto
        .createHash('sha256')
        .update(requirementsText)
        .update(' ')
        .update(pythonVersion)
        .update(' ')
        .update(process.platform)
        .digest('hex');
}

/**
 * Ensure the bundled backend and a matching venv are present, provisioning the
 * venv if it is missing or stale. Resolves to the interpreter that should run
 * uvicorn.
 */
export async function ensureBackendEnv(context: vscode.ExtensionContext): Promise<BackendEnv> {
    const extensionRoot = context.extensionPath;
    const backendEntry = path.join(extensionRoot, 'backend', 'main.py');

    if (!fs.existsSync(backendEntry)) {
        throw new BackendSetupError(
            `The CodeLens backend is missing from the installed extension (expected ${backendEntry}).`,
            'Reinstall the CodeLens extension - the .vsix appears to be incomplete.',
        );
    }

    const requirementsPath = path.join(extensionRoot, 'requirements.txt');
    if (!fs.existsSync(requirementsPath)) {
        throw new BackendSetupError(
            `The CodeLens dependency manifest is missing (expected ${requirementsPath}).`,
            'Reinstall the CodeLens extension - the .vsix appears to be incomplete.',
        );
    }
    const requirementsText = fs.readFileSync(requirementsPath, 'utf-8');

    const basePython = await resolveBasePython(getPythonSetting());
    const pythonVersion = (await probeInterpreter(basePython)) ?? 'unknown';
    const [baseExe, ...baseArgs] = basePython;
    const baseLabel = basePython.join(' ');

    const storageDir = context.globalStorageUri.fsPath;
    fs.mkdirSync(storageDir, { recursive: true });

    const venvDir = path.join(storageDir, 'venv');
    const interpreter = venvPython(venvDir);
    const stampFile = path.join(venvDir, '.codelens-stamp');
    const wantStamp = envStamp(requirementsText, pythonVersion);

    const upToDate =
        fs.existsSync(interpreter) &&
        fs.existsSync(stampFile) &&
        fs.readFileSync(stampFile, 'utf-8').trim() === wantStamp;

    if (upToDate) {
        return { pythonPath: interpreter, extensionRoot };
    }

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'CodeLens: preparing the offline search backend (one time, may take a few minutes)',
            cancellable: false,
        },
        async (progress) => {
            progress.report({ message: 'creating virtual environment' });
            try {
                await execFileAsync(baseExe, [...baseArgs, '-m', 'venv', venvDir], { timeout: 300000 });
            } catch (err) {
                throw new BackendSetupError(
                    `Failed to create a Python virtual environment at ${venvDir}: ${describe(err)}`,
                    `Check that "${baseLabel} -m venv" works, or set "codelens.pythonPath" to a different interpreter. On Debian/Ubuntu the python3-venv package may be missing.`,
                );
            }

            if (!fs.existsSync(interpreter)) {
                throw new BackendSetupError(
                    `The virtual environment at ${venvDir} has no interpreter at ${interpreter}.`,
                    'Delete that directory and reload the window to retry provisioning.',
                );
            }

            progress.report({ message: 'installing dependencies with pip' });
            try {
                await execFileAsync(
                    interpreter,
                    ['-m', 'pip', 'install', '--disable-pip-version-check', '-r', requirementsPath],
                    { timeout: 1800000, maxBuffer: 32 * 1024 * 1024 },
                );
            } catch (err) {
                throw new BackendSetupError(
                    `Installing the CodeLens backend dependencies failed: ${describe(err)}`,
                    `Run "${interpreter} -m pip install -r ${requirementsPath}" in a terminal to see the full pip output. A missing C toolchain, or no available wheel for Python ${pythonVersion}, is the usual cause.`,
                );
            }

            fs.writeFileSync(stampFile, wantStamp, 'utf-8');
        },
    );

    return { pythonPath: interpreter, extensionRoot };
}

/** Best-effort readable description of a child_process failure. */
function describe(err: unknown): string {
    if (err && typeof err === 'object') {
        const e = err as { stderr?: string; message?: string };
        const stderr = (e.stderr ?? '').trim();
        if (stderr) {
            return stderr.split('\n').slice(-6).join('\n');
        }
        if (e.message) {
            return e.message;
        }
    }
    return String(err);
}
