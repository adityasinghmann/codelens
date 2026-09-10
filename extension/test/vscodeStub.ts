/**
 * Minimal stand-in for the `vscode` module.
 *
 * Only the surface the tested modules actually touch is implemented. Calls are
 * recorded on `recorded` so tests can assert that, for example, a startup
 * failure produced a visible error message.
 */

export const recorded = {
    errorMessages: [] as string[],
    warningMessages: [] as string[],
    settings: {} as Record<string, unknown>,
};

export function resetRecorded() {
    recorded.errorMessages = [];
    recorded.warningMessages = [];
    recorded.settings = {};
}

export const window = {
    showErrorMessage: (message: string, ..._items: string[]) => {
        recorded.errorMessages.push(message);
        return Promise.resolve(undefined);
    },
    showWarningMessage: (message: string, ..._items: string[]) => {
        recorded.warningMessages.push(message);
        return Promise.resolve(undefined);
    },
    withProgress: async (_options: unknown, task: (progress: unknown) => Promise<unknown>) =>
        task({ report: () => undefined }),
    createStatusBarItem: () => ({
        text: '', tooltip: '', command: '', show: () => undefined, hide: () => undefined,
        dispose: () => undefined,
    }),
};

export const workspace = {
    workspaceFolders: undefined as unknown,
    getConfiguration: (_section?: string) => ({
        get: (key: string) => recorded.settings[`codelens.${key}`] ?? recorded.settings[key],
    }),
};

/** Mirrors vscode.EventEmitter closely enough for the lifecycle tests. */
export class EventEmitter<T> {
    private listeners: Array<(value: T) => void> = [];

    public event = (listener: (value: T) => void) => {
        this.listeners.push(listener);
        return { dispose: () => {
            this.listeners = this.listeners.filter((l) => l !== listener);
        } };
    };

    public fire(value: T) {
        for (const listener of [...this.listeners]) {
            listener(value);
        }
    }

    public dispose() {
        this.listeners = [];
    }
}

export const ProgressLocation = { Notification: 15 };

export const Uri = {
    file: (p: string) => ({ fsPath: p, scheme: 'file' }),
    parse: (s: string) => ({ toString: () => s }),
    joinPath: (base: { fsPath: string }, ...parts: string[]) => ({
        fsPath: [base.fsPath, ...parts].join('/'),
    }),
};

export const env = { openExternal: () => Promise.resolve(true) };

export const commands = {
    registerCommand: () => ({ dispose: () => undefined }),
    executeCommand: () => Promise.resolve(undefined),
};

export const StatusBarAlignment = { Left: 1, Right: 2 };
export const TextEditorRevealType = { InCenter: 2 };

export class Range {
    constructor(
        public startLine: number, public startChar: number,
        public endLine: number, public endChar: number,
    ) {}
    public get start() { return { line: this.startLine, character: this.startChar }; }
    public get end() { return { line: this.endLine, character: this.endChar }; }
}

export class Selection extends Range {}
