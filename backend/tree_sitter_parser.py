"""
Tree-sitter multi-language AST parser  (tree-sitter >= 0.22 API)
-----------------------------------------------------------------
Extracts semantic code units (functions, classes, methods) as chunks
for embedding into the local vector store.

Target node types are split into CONTAINERS and LEAVES:

  * a LEAF (function, method, constructor, arrow function, type alias) is
    emitted whole and traversal stops there - its body belongs to it
  * a CONTAINER (class, interface, impl, struct) is emitted as a header-only
    chunk (declaration, docstring and field declarations, with nested leaf
    bodies removed) and traversal continues into it

That split is what stops a class being emitted once as a chunk containing all
its methods and then again as one chunk per method - the same code embedded
twice, competing with itself in the results.

Code that no symbol covers (module-level statements, handlers passed inline
to a call) is emitted as "code" chunks, so nothing in a file goes unindexed
just because the file also contains a function.

Traversal is iterative and holds a reference to every node it visits. That is
not a style preference: the previous recursive walk dropped references to
intermediate Node objects, and tree-sitter's Python binding frees the
underlying node memory, so reading a child's bytes could hit freed memory. It
reproducibly segfaulted the backend on files in this repository.
"""

import os
import re
import logging
from typing import Optional

from tree_sitter import Language, Parser

import tree_sitter_python
import tree_sitter_typescript
import tree_sitter_javascript
import tree_sitter_go
import tree_sitter_rust
import tree_sitter_java

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language registry
# tree-sitter 0.22+ API: Language takes ONE argument (the capsule pointer)
# We store (Language object, explicit name string) because Language.name
# returns None for some grammars in 0.25.x (typescript, java, etc.)
# ---------------------------------------------------------------------------
_LANG_DEFS: dict[str, tuple[Language, str]] = {
    ".py":   (Language(tree_sitter_python.language()),                    "python"),
    ".ts":   (Language(tree_sitter_typescript.language_typescript()),     "typescript"),
    ".tsx":  (Language(tree_sitter_typescript.language_tsx()),            "tsx"),
    ".js":   (Language(tree_sitter_javascript.language()),                "javascript"),
    ".jsx":  (Language(tree_sitter_javascript.language()),                "javascript"),
    ".go":   (Language(tree_sitter_go.language()),                        "go"),
    ".rs":   (Language(tree_sitter_rust.language()),                      "rust"),
    ".java": (Language(tree_sitter_java.language()),                      "java"),
}

# Optional C/C++
try:
    import tree_sitter_cpp
    _LANG_DEFS[".cpp"] = (Language(tree_sitter_cpp.language()), "cpp")
    _LANG_DEFS[".cc"] = (Language(tree_sitter_cpp.language()), "cpp")
    _LANG_DEFS[".h"] = (Language(tree_sitter_cpp.language()), "cpp")
except ImportError:
    pass

# Public: just the extension set (for the file walker)
LANGUAGES: dict[str, Language] = {ext: lang for ext, (lang, _) in _LANG_DEFS.items()}


# ---------------------------------------------------------------------------
# Semantic units, split by whether their body belongs to them
# ---------------------------------------------------------------------------

# Emitted whole; traversal does NOT descend past them.
LEAF_NODE_TYPES: dict[str, set[str]] = {
    "python":     {"function_definition", "decorated_definition"},
    "typescript": {"function_declaration", "method_definition", "arrow_function",
                   "type_alias_declaration", "function_signature"},
    "tsx":        {"function_declaration", "method_definition", "arrow_function",
                   "type_alias_declaration"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    # Go types are units of their own: one chunk per type_spec, so a grouped
    # `type ( A struct{}; B int )` yields A and B, each under its real name.
    "go":         {"function_declaration", "method_declaration", "type_spec", "type_alias"},
    "rust":       {"function_item"},
    "java":       {"method_declaration", "constructor_declaration"},
    "cpp":        {"function_definition"},
}

# Emitted header-only; traversal DOES descend into them.
CONTAINER_NODE_TYPES: dict[str, set[str]] = {
    "python":     {"class_definition"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx":        {"class_declaration", "interface_declaration"},
    "javascript": {"class_declaration"},
    "go":         set(),
    "rust":       {"impl_item", "struct_item", "mod_item", "trait_item"},
    "java":       {"class_declaration", "interface_declaration", "enum_declaration"},
    "cpp":        {"class_specifier", "struct_specifier"},
}

# Mapping from AST node type to the symbol_type we report.
_SYMBOL_TYPES: dict[str, str] = {
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "function_signature": "function",
    "arrow_function": "function",
    "decorated_definition": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "class_definition": "class",
    "class_declaration": "class",
    "class_specifier": "class",
    "interface_declaration": "interface",
    "trait_item": "interface",
    "type_alias_declaration": "type",
    "type_spec": "type",
    "type_alias": "type",
    "struct_item": "struct",
    "struct_specifier": "struct",
    "impl_item": "impl",
    "mod_item": "module",
    "enum_declaration": "enum",
}

_NAME_NODE_TYPES = {"identifier", "name", "type_identifier", "property_identifier",
                    "field_identifier", "constant"}


def _node_text(source_bytes: bytes, node) -> str:
    """Decode a node's source range. Byte offsets, so slice the bytes."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_symbol_name(node, source_bytes: bytes) -> str:
    """
    Extract the declared name from a node.

    Handles Python's decorated_definition, whose direct children are the
    decorators plus the wrapped def/class - there is no identifier among them,
    so the previous implementation fell through and returned the literal string
    "decorated_definition" as the symbol name for every decorated function.
    """
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in {"function_definition", "class_definition"}:
                return _get_symbol_name(child, source_bytes)

    # Rust `impl Display for Money` belongs to Money, not to Display: name it
    # by its `type` field. The generic identifier scan below would find the
    # trait first. Generic arguments are dropped, so `impl<T> Wrapper<T>` is
    # scoped as Wrapper.
    if node.type == "impl_item":
        try:
            type_node = node.child_by_field_name("type")
            if type_node is not None and type_node.type == "generic_type":
                base = type_node.child_by_field_name("type")
                if base is not None:
                    type_node = base
        except Exception:
            type_node = None
        if type_node is not None:
            return _node_text(source_bytes, type_node)

    # A `name:` field is the most reliable source when the grammar has one.
    try:
        named = node.child_by_field_name("name")
    except Exception:
        named = None
    if named is not None:
        return _node_text(source_bytes, named)

    for child in node.children:
        if child.type in _NAME_NODE_TYPES:
            return _node_text(source_bytes, child)

    # Arrow function assigned to a variable takes the variable's name.
    if node.type == "arrow_function":
        parent = node.parent
        if parent is not None and parent.type == "variable_declarator":
            for child in parent.children:
                if child.type == "identifier":
                    return _node_text(source_bytes, child)

    # Rust `impl Foo for Bar` has a type, not a name.
    try:
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            return _node_text(source_bytes, type_node)
    except Exception:
        pass

    return node.type


def _effective_node(node):
    """The declaration a decorated_definition actually wraps."""
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in {"function_definition", "class_definition"}:
                return child
    return node


def _symbol_type_for(node, parent_symbol: Optional[str]) -> str:
    inner = _effective_node(node)
    symbol_type = _SYMBOL_TYPES.get(inner.type, inner.type)
    # A Go type_spec says what it declares through its `type` field.
    if inner.type == "type_spec":
        try:
            declared = inner.child_by_field_name("type")
        except Exception:
            declared = None
        if declared is not None:
            symbol_type = {"struct_type": "struct",
                           "interface_type": "interface"}.get(declared.type, symbol_type)
    # A function declared inside a class is a method, whatever the grammar
    # happens to call the node.
    if symbol_type == "function" and parent_symbol:
        return "method"
    return symbol_type


def _container_header(node, source_bytes: bytes, emitted_types: set[str]) -> str:
    """
    Header-only text for a container: the declaration line, its docstring and
    its field declarations, with the bodies of nested symbols removed.

    Keeping the signature and docstring preserves what makes the class findable
    semantically; dropping the bodies of everything emitted as its own chunk -
    methods, and nested classes too - is what stops the container chunk
    duplicating the chunks beneath it.
    """
    inner = _effective_node(node)
    pieces: list[str] = []
    cursor = inner.start_byte

    def is_leaf(n) -> bool:
        return _effective_node(n).type in emitted_types

    # Walk the class body one level deep, skipping the bodies of nested leaves.
    body = None
    try:
        body = inner.child_by_field_name("body")
    except Exception:
        body = None

    if body is None:
        return _node_text(source_bytes, inner)

    for child in list(body.children):
        if is_leaf(child):
            # Keep the signature line, drop the body.
            sig_end = child.start_byte
            inner_child = _effective_node(child)
            try:
                child_body = inner_child.child_by_field_name("body")
            except Exception:
                child_body = None
            if child_body is not None:
                sig_end = child_body.start_byte
            else:
                sig_end = child.end_byte
            pieces.append(source_bytes[cursor:sig_end].decode("utf-8", errors="replace"))
            cursor = child.end_byte

    pieces.append(source_bytes[cursor:inner.end_byte].decode("utf-8", errors="replace"))
    header = "".join(pieces)

    # Collapse the blank space left behind by removed bodies.
    lines = [ln.rstrip() for ln in header.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()) or _node_text(source_bytes, inner)


def extract_chunks(file_path: str, content: str) -> list[dict]:
    """
    Parse `content` with tree-sitter and return semantic chunk dicts.

    Falls back to a sliding window for unsupported languages, symbol-less
    files, and any parse failure - a malformed file must never raise.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _LANG_DEFS:
        return _sliding_window(file_path, content, "unknown")

    language, lang_name = _LANG_DEFS[ext]
    source_bytes = content.encode("utf-8")

    try:
        parser = Parser(language)
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.warning("Parse failed for %s (%s); using sliding window.", file_path, e)
        return _sliding_window(file_path, content, lang_name)

    leaf_types = LEAF_NODE_TYPES.get(lang_name, set())
    container_types = CONTAINER_NODE_TYPES.get(lang_name, set())
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    chunks: list[dict] = []
    # Keyed on (type, start_byte, end_byte). id(node) is NOT usable: the Python
    # binding constructs a fresh Node object on every access, so ids are neither
    # stable nor unique once earlier objects have been collected.
    visited: set[tuple] = set()

    # Explicit stack, and every node stays referenced by it for the duration of
    # the traversal. See the module docstring - this prevents a use-after-free.
    stack: list[tuple] = [(tree.root_node, None)]
    alive = [tree]  # keep the tree alive for the whole walk

    try:
        while stack:
            node, parent_symbol = stack.pop()
            alive.append(node)

            key = (node.type, node.start_byte, node.end_byte)
            if key in visited:
                continue
            visited.add(key)

            # Classify by what a decorator wraps, not by the wrapper: a
            # decorated class is a container. Classifying the wrapper made
            # `@dataclass class X` a leaf, so its methods were never visited
            # and their bodies were indexed nowhere.
            inner = _effective_node(node)
            alive.append(inner)
            is_leaf = inner.type in leaf_types
            is_container = inner.type in container_types

            # An arrow function only counts when bound to a variable, otherwise
            # every inline callback becomes a chunk.
            if is_leaf and inner.type == "arrow_function":
                parent = node.parent
                is_leaf = parent is not None and parent.type == "variable_declarator"

            if is_leaf or is_container:
                symbol_name = _get_symbol_name(node, source_bytes)
                symbol_type = _symbol_type_for(node, parent_symbol)
                qualified = ".".join(
                    p for p in (module_name, parent_symbol, symbol_name) if p
                )

                if is_container:
                    chunk_text = _container_header(node, source_bytes, leaf_types | container_types)
                else:
                    chunk_text = _node_text(source_bytes, node)

                chunks.append({
                    "file_path":      file_path,
                    "start_line":     node.start_point.row + 1,
                    "end_line":       node.end_point.row + 1,
                    "symbol_name":    symbol_name,
                    "qualified_name": qualified,
                    "symbol_type":    symbol_type,
                    "parent_symbol":  parent_symbol,
                    "language":       lang_name,
                    "chunk_text":     chunk_text,
                })

                if is_leaf:
                    # The body belongs to this chunk; do not emit it again.
                    continue

                # Container: descend, and nested symbols are qualified by it.
                # Descend from the unwrapped node, so a decorated class is not
                # visited a second time through its own class_definition.
                child_scope = ".".join(p for p in (parent_symbol, symbol_name) if p)
                for child in reversed(list(inner.children)):
                    stack.append((child, child_scope))
                continue

            for child in reversed(list(node.children)):
                stack.append((child, parent_symbol))
    except Exception as e:
        logger.warning("Chunking failed for %s (%s); using sliding window.", file_path, e)
        return _sliding_window(file_path, content, lang_name)

    if not chunks and content.strip():
        return _sliding_window(file_path, content, lang_name)

    chunks.extend(_module_level_chunks(file_path, content, chunks, lang_name, module_name))
    chunks.sort(key=lambda c: (c["start_line"], c["end_line"]))
    return chunks


# A line that carries no meaning of its own: imports, package/use lines and
# bare closing brackets. A block made only of these is not worth embedding.
_BOILERPLATE_LINE = re.compile(
    r"^\s*(?:"
    r"import\b|from\s+[\w.]+\s+import\b|package\s|use\s|#include\b|using\s|"
    r"(?:const|let|var)\s+[^=]+=\s*require\(|"
    r"(?:type|var|const|import)\s*\(\s*$|"   # Go group openers: `type (`
    r"[\])},;]*\s*$"
    r")"
)


def _module_level_chunks(file_path: str, content: str, chunks: list[dict],
                         lang_name: str, module_name: str,
                         window: int = 40, overlap: int = 10) -> list[dict]:
    """
    Chunks for code that no symbol covers.

    Symbol extraction alone drops everything outside a recognised function or
    class once a file has at least one: an Express handler passed inline to
    app.post(), a module-level config dict, a Go var block. That is often
    exactly what a question is about, so each contiguous run of uncovered lines
    becomes a "code" chunk (windowed if long). Runs of pure imports or closing
    brackets are skipped.
    """
    lines = content.splitlines()
    covered = [False] * (len(lines) + 2)
    for chunk in chunks:
        for line_no in range(chunk["start_line"], min(chunk["end_line"], len(lines)) + 1):
            covered[line_no] = True

    runs: list[tuple[int, int]] = []  # 1-based inclusive line ranges
    start = None
    for line_no in range(1, len(lines) + 1):
        if not covered[line_no]:
            if start is None:
                start = line_no
        elif start is not None:
            runs.append((start, line_no - 1))
            start = None
    if start is not None:
        runs.append((start, len(lines)))

    blocks: list[dict] = []
    step = max(1, window - overlap)
    for run_start, run_end in runs:
        # Trim blank lines at either end of the run.
        while run_start <= run_end and not lines[run_start - 1].strip():
            run_start += 1
        while run_end >= run_start and not lines[run_end - 1].strip():
            run_end -= 1
        if run_start > run_end:
            continue

        significant = [ln for ln in lines[run_start - 1:run_end] if ln.strip()]
        if all(_BOILERPLATE_LINE.match(ln) for ln in significant):
            continue

        for offset in range(0, run_end - run_start + 1, step):
            first = run_start + offset
            last = min(first + window - 1, run_end)
            text = "\n".join(lines[first - 1:last]).strip()
            if text:
                blocks.append({
                    "file_path":      file_path,
                    "start_line":     first,
                    "end_line":       last,
                    # Identity is (type, qualified name) plus an occurrence
                    # ordinal assigned in source order - not line numbers.
                    "symbol_name":    "<module>",
                    "qualified_name": f"{module_name}.<module>" if module_name else "<module>",
                    "symbol_type":    "code",
                    "parent_symbol":  None,
                    "language":       lang_name,
                    "chunk_text":     text,
                })
            if last == run_end:
                break
    return blocks


def _sliding_window(file_path: str, content: str, lang_name: str,
                    window: int = 40, overlap: int = 10) -> list[dict]:
    """Line-based sliding-window fallback."""
    lines = content.splitlines()
    step = max(1, window - overlap)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    chunks = []
    for i in range(0, len(lines), step):
        chunk_lines = lines[i: i + window]
        text = "\n".join(chunk_lines).strip()
        if not text:
            continue
        symbol = f"lines_{i+1}_{i+len(chunk_lines)}"
        chunks.append({
            "file_path":      file_path,
            "start_line":     i + 1,
            "end_line":       i + len(chunk_lines),
            "symbol_name":    symbol,
            "qualified_name": f"{module_name}.{symbol}" if module_name else symbol,
            "symbol_type":    "window",
            "parent_symbol":  None,
            "language":       lang_name,
            "chunk_text":     text,
        })
    return chunks
