import * as path from 'path';
import { mergeFolderResults, selectWorkspaceRoot } from '../src/multiRoot';
import { QueryResult } from '../src/apiTypes';

function hit(name: string, score: number): QueryResult {
    return {
        symbol_name: name, qualified_name: name, symbol_type: 'function',
        file_path: `${name}.py`, start_line: 1, end_line: 2, language: 'python',
        chunk_text: `def ${name}(): pass`, score,
    };
}

const ROOT_A = path.resolve('/work/app');
const ROOT_B = path.resolve('/work/lib');

describe('mergeFolderResults', () => {
    it('ranks hits from every folder together by score', () => {
        const merged = mergeFolderResults([
            { root: ROOT_A, name: 'app', results: [hit('a1', 0.9), hit('a2', 0.4)] },
            { root: ROOT_B, name: 'lib', results: [hit('b1', 0.7)] },
        ], 10);

        expect(merged.map((h) => h.symbol_name)).toEqual(['a1', 'b1', 'a2']);
    });

    it('tags each hit with the folder it came from', () => {
        const merged = mergeFolderResults([
            { root: ROOT_A, name: 'app', results: [hit('a1', 0.9)] },
            { root: ROOT_B, name: 'lib', results: [hit('b1', 0.7)] },
        ], 10);

        expect(merged[0]).toMatchObject({ root: ROOT_A, folder: 'app' });
        expect(merged[1]).toMatchObject({ root: ROOT_B, folder: 'lib' });
    });

    it('does not label hits when there is only one folder', () => {
        const [only] = mergeFolderResults([{ root: ROOT_A, name: 'app', results: [hit('a1', 0.9)] }], 10);
        expect(only.root).toBe(ROOT_A);
        expect(only.folder).toBeUndefined();
    });

    it('keeps only the best `limit` hits overall', () => {
        const merged = mergeFolderResults([
            { root: ROOT_A, name: 'app', results: [hit('a1', 0.9), hit('a2', 0.2)] },
            { root: ROOT_B, name: 'lib', results: [hit('b1', 0.8), hit('b2', 0.1)] },
        ], 2);

        expect(merged.map((h) => h.symbol_name)).toEqual(['a1', 'b1']);
    });
});

describe('selectWorkspaceRoot', () => {
    const roots = [ROOT_A, ROOT_B];

    it('accepts any open workspace folder, not just the first', () => {
        expect(selectWorkspaceRoot(roots, ROOT_B)).toBe(ROOT_B);
    });

    it('refuses a root that is not an open workspace folder', () => {
        expect(selectWorkspaceRoot(roots, path.resolve('/etc'))).toBeNull();
        expect(selectWorkspaceRoot(roots, path.join(ROOT_A, '..'))).toBeNull();
    });

    it('falls back to the first folder when no root is given', () => {
        expect(selectWorkspaceRoot(roots, undefined)).toBe(ROOT_A);
        expect(selectWorkspaceRoot(roots, '')).toBe(ROOT_A);
    });

    it('returns null when no folder is open', () => {
        expect(selectWorkspaceRoot([], undefined)).toBeNull();
    });
});
