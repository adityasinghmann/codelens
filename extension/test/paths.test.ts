/**
 * Path containment for jump-to-file.
 *
 * The previous implementation used path.join with no check, so a file_path
 * containing '..' could open any file on the machine.
 */
import * as path from 'path';
import { resolveInsideRoot } from '../src/paths';

const isWindows = process.platform === 'win32';
const ROOT = isWindows ? 'D:\\work\\repo' : '/work/repo';
const OUTSIDE_ABSOLUTE = isWindows ? 'C:\\Windows\\system32\\evil.dll' : '/etc/passwd';

describe('resolveInsideRoot', () => {
    it('allows an ordinary relative path', () => {
        expect(resolveInsideRoot(ROOT, 'src/auth.py')).toBe(path.resolve(ROOT, 'src/auth.py'));
    });

    it('allows a deeply nested path', () => {
        expect(resolveInsideRoot(ROOT, 'a/b/c/d.ts')).not.toBeNull();
    });

    it('allows a traversal that lands back inside the root', () => {
        expect(resolveInsideRoot(ROOT, 'src/../lib/ok.py')).toBe(path.resolve(ROOT, 'lib/ok.py'));
    });

    it('rejects parent traversal', () => {
        expect(resolveInsideRoot(ROOT, '../../../etc/passwd')).toBeNull();
    });

    it('rejects traversal embedded mid-path', () => {
        expect(resolveInsideRoot(ROOT, 'src/../../outside.py')).toBeNull();
    });

    it('rejects an absolute path outside the root', () => {
        expect(resolveInsideRoot(ROOT, OUTSIDE_ABSOLUTE)).toBeNull();
    });

    it('rejects the root itself', () => {
        expect(resolveInsideRoot(ROOT, '.')).toBeNull();
    });

    it('rejects an empty path', () => {
        expect(resolveInsideRoot(ROOT, '')).toBeNull();
    });

    it('rejects a sibling directory that merely shares a name prefix', () => {
        expect(resolveInsideRoot(ROOT, '../repo-evil/x.py')).toBeNull();
    });

    it('normalises the root before comparing', () => {
        expect(resolveInsideRoot(`${ROOT}${path.sep}.`, 'src/a.py')).not.toBeNull();
    });

    if (isWindows) {
        it('rejects a path on another drive', () => {
            expect(resolveInsideRoot(ROOT, 'E:\\elsewhere\\x.py')).toBeNull();
        });
    }
});
