/**
 * Jest configuration for the extension's unit tests.
 *
 * The `vscode` module only exists inside a running VS Code host, so it is
 * mapped to a hand-written stub. Everything tested here is deliberately logic
 * that does not need a real editor: SSE parsing, path containment, and the
 * backend readiness/lifecycle state machine.
 */
module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'node',
    roots: ['<rootDir>/extension/test'],
    moduleNameMapper: {
        '^vscode$': '<rootDir>/extension/test/vscodeStub.ts',
    },
    transform: {
        '^.+\.ts$': ['ts-jest', { tsconfig: { module: 'commonjs', target: 'ES2022', esModuleInterop: true, strict: false } }],
    },
    testMatch: ['**/*.test.ts'],
};
