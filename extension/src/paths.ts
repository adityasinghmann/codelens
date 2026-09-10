/**
 * Path safety helpers.
 *
 * Kept free of any `vscode` import so the logic can be unit tested directly -
 * containment is exactly the kind of rule that should not be verified only by
 * reading it.
 */
import * as path from 'path';

/**
 * Resolve `candidate` against `root` and return the absolute path only if the
 * result stays inside `root`.
 *
 * Returns null when the path escapes, is the root itself, or is empty.
 *
 * `path.join` alone is not a containment check: it happily resolves
 * "../../../etc/passwd" to somewhere entirely outside the workspace, which is
 * what the previous jump-to-file implementation did.
 */
export function resolveInsideRoot(root: string, candidate: string): string | null {
    if (!candidate) {
        return null;
    }

    const absoluteRoot = path.resolve(root);
    const resolved = path.resolve(absoluteRoot, candidate);
    const relative = path.relative(absoluteRoot, resolved);

    // '' means the candidate resolved to the root itself; a '..' prefix or an
    // absolute result means it escaped.
    if (relative === '' || relative.startsWith('..') || path.isAbsolute(relative)) {
        return null;
    }
    return resolved;
}
