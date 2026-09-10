/**
 * Cross-platform discovery of the two external executables CodeLens depends on.
 *
 * The same resolution order applies to both, and it is the order a user would
 * expect:
 *   1. the explicit VS Code setting, if set  (always wins, never second-guessed)
 *   2. the executable on PATH
 *   3. a platform-specific default install location
 *
 * If nothing resolves, the caller gets a ToolResolutionError naming the setting
 * to configure. Nothing here throws asynchronously, blocks indefinitely, or
 * falls back to a macOS path on Linux and Windows the way the previous
 * hardcoded constants did.
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

/** Probe timeout for "does this executable run?" checks. */
const PROBE_TIMEOUT_MS = 10000;

export class ToolResolutionError extends Error {
    constructor(message: string, public readonly remedy: string) {
        super(message);
        this.name = 'ToolResolutionError';
    }
}

/** Platform-specific default install locations for the Ollama CLI. */
function ollamaDefaults(): string[] {
    const home = os.homedir();
    switch (process.platform) {
        case 'darwin':
            return [
                '/Applications/Ollama.app/Contents/Resources/ollama',
                '/usr/local/bin/ollama',
                '/opt/homebrew/bin/ollama',
                path.join(home, '.ollama', 'bin', 'ollama'),
            ];
        case 'win32':
            return [
                path.join(process.env.LOCALAPPDATA ?? path.join(home, 'AppData', 'Local'), 'Programs', 'Ollama', 'ollama.exe'),
                path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Ollama', 'ollama.exe'),
            ];
        default:
            return [
                '/usr/local/bin/ollama',
                '/usr/bin/ollama',
                path.join(home, '.local', 'bin', 'ollama'),
            ];
    }
}

/** Names to try on PATH for the Ollama CLI. */
function ollamaOnPath(): string[] {
    return process.platform === 'win32' ? ['ollama.exe', 'ollama'] : ['ollama'];
}

/**
 * Can `exe` be executed? Runs it with `args` and treats any successful exit as
 * proof. An absolute path is additionally required to exist, so a typo in a
 * setting is reported as a bad path rather than a PATH lookup miss.
 */
async function isRunnable(exe: string, args: string[]): Promise<boolean> {
    if (path.isAbsolute(exe) && !fs.existsSync(exe)) {
        return false;
    }
    try {
        await execFileAsync(exe, args, { timeout: PROBE_TIMEOUT_MS });
        return true;
    } catch (err: any) {
        // A non-zero exit still proves the binary exists and ran; only a
        // spawn error (ENOENT/EACCES) means it is not usable.
        return err?.code !== 'ENOENT' && err?.code !== 'EACCES' && err?.errno !== -4058;
    }
}

/**
 * Resolve an executable using setting -> PATH -> platform defaults.
 *
 * `explicit` is the user's setting ('' when unset). It is never skipped: if it
 * is set but unusable, that is reported directly rather than silently falling
 * through to a different binary, which would hide the user's mistake.
 */
export async function resolveTool(options: {
    explicit: string;
    pathNames: string[];
    defaults: string[];
    probeArgs: string[];
    displayName: string;
    settingName: string;
    installHint: string;
}): Promise<string> {
    const { explicit, pathNames, defaults, probeArgs, displayName, settingName, installHint } = options;

    if (explicit) {
        if (await isRunnable(explicit, probeArgs)) {
            return explicit;
        }
        throw new ToolResolutionError(
            `CodeLens could not run ${displayName} configured in "${settingName}": ${explicit}`,
            `Set "${settingName}" to a working path, or clear it to let CodeLens search PATH.`,
        );
    }

    for (const name of pathNames) {
        if (await isRunnable(name, probeArgs)) {
            return name;
        }
    }

    for (const candidate of defaults) {
        if (await isRunnable(candidate, probeArgs)) {
            return candidate;
        }
    }

    throw new ToolResolutionError(
        `CodeLens could not find ${displayName} on PATH or in the usual install locations for ${process.platform}.`,
        `${installHint} Then set "${settingName}" to its full path if it is still not found.`,
    );
}

/** Locate the Ollama CLI. */
export async function resolveOllama(explicit: string): Promise<string> {
    return resolveTool({
        explicit,
        pathNames: ollamaOnPath(),
        defaults: ollamaDefaults(),
        probeArgs: ['--version'],
        displayName: 'the Ollama executable',
        settingName: 'codelens.ollamaPath',
        installHint: 'Install Ollama from https://ollama.com.',
    });
}

/**
 * Directory to prepend to PATH for the backend process, so a GUI-launched
 * VS Code (which does not inherit a login shell PATH) can still find Ollama.
 * Returns null when the resolved name came from PATH already.
 */
export function pathPrefixFor(resolved: string): string | null {
    return path.isAbsolute(resolved) ? path.dirname(resolved) : null;
}
