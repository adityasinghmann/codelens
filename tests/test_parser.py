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

    assert len(containers) == 1
    assert len(methods) == 3
    assert len(chunks) == 4
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
