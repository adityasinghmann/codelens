"""Tree-sitter chunking: what gets emitted, with what metadata."""

import pytest

from backend.tree_sitter_parser import extract_chunks
from tests.conftest import JAVA_CLASS, PY_CLASS, PY_DECORATED, PY_NESTED, TS_MIXED


def by_name(chunks):
    return {c["symbol_name"]: c for c in chunks}


# --- Python ---------------------------------------------------------------

def test_python_function_is_emitted_whole():
    chunks = extract_chunks("m.py", "def add(a, b):\n    return a + b\n")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["symbol_name"] == "add"
    assert chunk["symbol_type"] == "function"
    assert chunk["language"] == "python"
    assert "return a + b" in chunk["chunk_text"]
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 2


def test_class_with_three_methods_yields_one_container_and_three_leaves():
    chunks = extract_chunks("greeter.py", PY_CLASS)
    containers = [c for c in chunks if c["symbol_type"] == "class"]
    methods = [c for c in chunks if c["symbol_type"] == "method"]

    module_code = [c for c in chunks if c["symbol_type"] == "code"]

    assert len(containers) == 1
    assert len(methods) == 3
    # The module docstring sits outside every symbol and is its own chunk.
    assert [c["chunk_text"] for c in module_code] == ['"""Module docstring."""']
    assert len(chunks) == 5
    assert sorted(m["symbol_name"] for m in methods) == ["__init__", "greet", "shout"]


def test_container_chunk_does_not_contain_method_bodies():
    """The whole point of the container/leaf split: no duplicated source."""
    container = by_name(extract_chunks("greeter.py", PY_CLASS))["Greeter"]
    text = container["chunk_text"]

    # Bodies are gone...
    assert "self.name = name" not in text
    assert ".upper()" not in text
    # ...but everything that makes the class findable stays.
    assert "class Greeter" in text
    assert "Greets people" in text
    assert 'prefix = "Hello"' in text
    assert "def greet" in text


def test_method_chunk_keeps_its_own_body():
    shout = by_name(extract_chunks("greeter.py", PY_CLASS))["shout"]
    assert ".upper()" in shout["chunk_text"]


def test_decorated_functions_report_their_real_name():
    """Previously every decorated def was named "decorated_definition"."""
    chunks = extract_chunks("dec.py", PY_DECORATED)
    names = sorted(c["symbol_name"] for c in chunks)
    assert names == ["cached_lookup", "handler"]
    assert not any(c["symbol_name"] == "decorated_definition" for c in chunks)


def test_decorated_function_keeps_its_decorators_in_the_text():
    chunk = by_name(extract_chunks("dec.py", PY_DECORATED))["handler"]
    assert "@auth_required" in chunk["chunk_text"]


def test_nested_classes_qualify_through_every_enclosing_scope():
    chunks = extract_chunks("nest.py", PY_NESTED)
    qualified = sorted(c["qualified_name"] for c in chunks)
    assert qualified == [
        "nest.Outer",
        "nest.Outer.Inner",
        "nest.Outer.Inner.deep",
        "nest.Outer.shallow",
    ]


def test_metadata_shape_is_complete():
    chunk = by_name(extract_chunks("greeter.py", PY_CLASS))["greet"]
    for key in ("file_path", "start_line", "end_line", "symbol_name",
                "qualified_name", "symbol_type", "parent_symbol",
                "language", "chunk_text"):
        assert key in chunk, key
    assert chunk["qualified_name"] == "greeter.Greeter.greet"
    assert chunk["parent_symbol"] == "Greeter"
    assert chunk["start_line"] < chunk["end_line"]


def test_chunks_are_ordered_by_position():
    chunks = extract_chunks("greeter.py", PY_CLASS)
    lines = [c["start_line"] for c in chunks]
    assert lines == sorted(lines)


# --- Java -----------------------------------------------------------------

def test_java_methods_and_constructor():
    chunks = extract_chunks("Service.java", JAVA_CLASS)
    kinds = {c["symbol_name"]: c["symbol_type"] for c in chunks}
    assert kinds["Service"] in {"class", "constructor"}
    assert kinds["getCount"] == "method"
    assert kinds["reset"] == "method"
    assert any(c["symbol_type"] == "constructor" for c in chunks)
    assert any(c["symbol_type"] == "class" for c in chunks)


def test_java_container_excludes_method_bodies():
    container = [c for c in extract_chunks("Service.java", JAVA_CLASS)
                 if c["symbol_type"] == "class"][0]
    assert "count = 0" not in container["chunk_text"]
    assert "class Service" in container["chunk_text"]


def test_java_language_is_tagged():
    chunks = extract_chunks("Service.java", JAVA_CLASS)
    assert all(c["language"] == "java" for c in chunks)


# --- TypeScript -----------------------------------------------------------

def test_typescript_finds_interface_class_method_arrow_and_function():
    names = {c["symbol_name"] for c in extract_chunks("w.ts", TS_MIXED)}
    assert {"Opts", "Widget", "render", "helper", "topLevel"} <= names


def test_typescript_container_excludes_method_bodies():
    widget = by_name(extract_chunks("w.ts", TS_MIXED))["Widget"]
    assert "return this.id" not in widget["chunk_text"]
    assert "class Widget" in widget["chunk_text"]


def test_typescript_arrow_function_takes_the_variable_name():
    helper = by_name(extract_chunks("w.ts", TS_MIXED))["helper"]
    assert helper["symbol_type"] == "function"


def test_inline_arrow_callback_is_not_its_own_chunk():
    """Only arrow functions bound to a variable count as symbols."""
    src = "items.forEach((item) => { console.log(item); });\n"
    chunks = extract_chunks("cb.ts", src)
    # No named symbol here, so this falls back to the sliding window.
    assert all(c["symbol_type"] == "window" for c in chunks)


# --- Fallbacks ------------------------------------------------------------

def test_malformed_source_falls_back_instead_of_raising():
    chunks = extract_chunks("broken.py", "def broken(:\n  @@@ not python (((\n")
    assert chunks
    assert all("chunk_text" in c for c in chunks)


def test_unsupported_extension_falls_back_to_sliding_window():
    chunks = extract_chunks("notes.md", "# Title\n\nSome prose.\n")
    assert chunks
    assert chunks[0]["symbol_type"] == "window"
    assert chunks[0]["language"] == "unknown"


def test_sliding_window_carries_the_same_metadata_shape():
    chunk = extract_chunks("notes.md", "# Title\n\nSome prose.\n")[0]
    for key in ("file_path", "start_line", "end_line", "symbol_name",
                "qualified_name", "symbol_type", "parent_symbol",
                "language", "chunk_text"):
        assert key in chunk, key


def test_file_with_no_symbols_falls_back_to_sliding_window():
    chunks = extract_chunks("consts.py", "A = 1\nB = 2\nC = 3\n")
    assert chunks
    assert all(c["symbol_type"] == "window" for c in chunks)


def test_empty_source_yields_nothing():
    assert extract_chunks("empty.py", "") == []


@pytest.mark.parametrize("path", [
    "backend/db_client.py",
    "backend/indexer.py",
    "backend/main.py",
    "backend/tree_sitter_parser.py",
])
def test_this_repository_parses_without_crashing(path):
    """
    Regression: a recursive walk that dropped Node references caused a
    use-after-free that segfaulted on these exact files.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    source = (root / path).read_text(encoding="utf-8")
    chunks = extract_chunks(path, source)
    assert len(chunks) > 3


# --- Code outside any symbol ----------------------------------------------

def test_inline_route_handlers_are_indexed_alongside_named_functions():
    """Once a file had one symbol, everything outside symbols was dropped."""
    src = (
        'const app = require("express")();\n'
        "function helper() { return 1; }\n"
        'app.post("/login", (req, res) => {\n'
        "  if (!checkPassword(req.body.pw)) { return res.status(401).end(); }\n"
        "  issueJwtToken(res);\n"
        "});\n"
    )
    chunks = extract_chunks("server.js", src)
    code = [c for c in chunks if c["symbol_type"] == "code"]
    assert any("issueJwtToken" in c["chunk_text"] for c in code)
    assert by_name(chunks)["helper"]["symbol_type"] == "function"


def test_module_level_assignments_are_indexed():
    src = 'DATABASE_URL = "postgres://db/billing"\n\n\ndef get(name):\n    return name\n'
    code = [c for c in extract_chunks("settings.py", src) if c["symbol_type"] == "code"]
    assert len(code) == 1
    assert "DATABASE_URL" in code[0]["chunk_text"]
    assert (code[0]["start_line"], code[0]["end_line"]) == (1, 1)


def test_import_only_blocks_are_not_chunked():
    src = "import os\nfrom typing import Any\n\n\ndef f():\n    return os.sep\n"
    assert [c["symbol_type"] for c in extract_chunks("m.py", src)] == ["function"]


def test_module_code_never_duplicates_symbol_lines():
    src = "X = 1\n\ndef f():\n    return X\n\nY = 2\n"
    chunks = extract_chunks("m.py", src)
    code_lines = {ln for c in chunks if c["symbol_type"] == "code"
                  for ln in range(c["start_line"], c["end_line"] + 1)}
    f = by_name(chunks)["f"]
    assert code_lines == {1, 6}
    assert not code_lines & set(range(f["start_line"], f["end_line"] + 1))


# --- Decorated and nested classes -----------------------------------------

def test_methods_of_a_decorated_class_are_emitted():
    """@dataclass made the class a leaf, so its method bodies were lost."""
    src = (
        "@dataclass\n"
        "class Invoice:\n"
        "    total: int\n"
        "    def apply_discount(self, pct):\n"
        "        return compute_tax_rebate(self.total)\n"
    )
    chunks = by_name(extract_chunks("m.py", src))
    assert chunks["Invoice"]["symbol_type"] == "class"
    method = chunks["apply_discount"]
    assert method["symbol_type"] == "method"
    assert method["qualified_name"] == "m.Invoice.apply_discount"
    assert "compute_tax_rebate" in method["chunk_text"]
    assert "compute_tax_rebate" not in chunks["Invoice"]["chunk_text"]


def test_a_decorated_class_is_emitted_once():
    src = "@dataclass\nclass Point:\n    x: int\n"
    classes = [c for c in extract_chunks("m.py", src) if c["symbol_type"] == "class"]
    assert len(classes) == 1


def test_nested_class_bodies_are_not_duplicated_in_the_outer_header():
    src = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def deep(self):\n"
        "            return very_specific_body_text()\n"
        "    def top(self):\n"
        "        return 1\n"
    )
    chunks = extract_chunks("m.py", src)
    holders = [c["qualified_name"] for c in chunks if "very_specific_body_text" in c["chunk_text"]]
    assert holders == ["m.Outer.Inner.deep"]
    assert "class Inner" in by_name(chunks)["Outer"]["chunk_text"]


# --- Rust and Go naming ---------------------------------------------------

def test_rust_trait_impl_is_named_after_its_type_not_the_trait():
    src = (
        "struct Money(i64);\n"
        "impl Display for Money {\n"
        "    fn fmt(&self, f: &mut Formatter) -> Result { Ok(()) }\n"
        "}\n"
        "impl<T> Wrapper<T> {\n"
        "    fn new(v: T) -> Self { Wrapper(v) }\n"
        "}\n"
    )
    names = {c["qualified_name"] for c in extract_chunks("m.rs", src)}
    assert {"m.Money", "m.Money.fmt", "m.Wrapper", "m.Wrapper.new"} <= names
    assert not any("Display" in n for n in names)


def test_go_types_are_named_and_typed_individually():
    src = (
        "package main\n"
        "type User struct { Name string }\n"
        "type (\n"
        "    Store interface { Get(id int) User }\n"
        "    ID = int\n"
        ")\n"
    )
    chunks = by_name(extract_chunks("m.go", src))
    assert chunks["User"]["symbol_type"] == "struct"
    assert chunks["Store"]["symbol_type"] == "interface"
    assert chunks["ID"]["symbol_type"] == "type"
    assert "type_declaration" not in chunks
