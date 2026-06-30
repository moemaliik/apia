"""Sandbox for synthesised capability code.

Synthesised code is LLM-generated, so it is statically validated with the `ast`
module before it is ever executed: no imports outside a small allowlist, no
dunder attribute access, no calls to open/exec/eval/compile/__import__/getattr
on arbitrary objects. The code runs with a curated builtins dict and a fixed set
of injected safe modules. This is defence-in-depth, not a security boundary for
hostile input — it stops an LLM from accidentally reaching the filesystem or net
outside the GitHub client.
"""
from __future__ import annotations

import ast
import builtins as _builtins
import collections
import datetime
import json
import math
import re
from typing import Callable

ALLOWED_IMPORTS = {"json", "re", "datetime", "collections", "math"}
# Pure-computation stdlib helpers that the allowed modules import lazily or
# internally (no filesystem/network). datetime.strptime, for example, does a
# lazy `import _strptime`, which in turn imports `time` and `calendar`; without
# these the obvious way to parse an ISO timestamp raises ImportError *inside*
# the sandbox even though the synthesised source only ever imported `datetime`.
_ALLOWED_INTERNAL = {"time", "_strptime", "calendar", "_datetime",
                     "itertools", "operator", "functools", "types",
                     "_collections", "_collections_abc", "keyword", "reprlib",
                     "heapq", "_json", "copyreg", "_locale"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals",
                   "locals", "vars", "input", "breakpoint", "memoryview"}

SAFE_MODULES = {"json": json, "re": re, "datetime": datetime,
                "collections": collections, "math": math}

_REAL_IMPORT = _builtins.__import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in ALLOWED_IMPORTS or root in _ALLOWED_INTERNAL:
        return _REAL_IMPORT(name, globals, locals, fromlist, level)
    raise ImportError(f"import not allowed: {name}")


SAFE_BUILTINS = {
    "len": len, "range": range, "sorted": sorted, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "str": str, "int": int, "float": float,
    "bool": bool, "min": min, "max": max, "sum": sum, "any": any, "all": all,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "reversed": reversed, "abs": abs, "round": round, "True": True,
    "False": False, "None": None, "isinstance": isinstance, "print": print,
    "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError,
    "TypeError": TypeError, "IndexError": IndexError, "AttributeError": AttributeError,
    "StopIteration": StopIteration, "frozenset": frozenset, "repr": repr,
    "next": next, "iter": iter, "divmod": divmod, "chr": chr, "ord": ord,
    "__import__": _safe_import,
    # the `class` statement compiles to a LOAD_BUILD_CLASS; without this any
    # synthesised code that defines a helper class fails with "__build_class__
    # not found". Escapes via class internals are still blocked by the AST
    # dunder-attribute check in validate().
    "__build_class__": _builtins.__build_class__,
}


class SandboxError(RuntimeError):
    pass


def _attr_chain(node: ast.AST) -> str | None:
    """Return the dotted source path for an attribute/name node, e.g.
    `ctx.gh.request`, or None if it isn't a plain attribute chain."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def validate(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names]
            bad = [n for n in names if n not in ALLOWED_IMPORTS]
            if bad:
                raise SandboxError(f"import not allowed: {bad}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(f"dunder attribute access not allowed: {node.attr}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxError(f"name not allowed: {node.id}")
        # Forbid stubbing the platform handle (e.g. `ctx.gh.request = mock`).
        # A selftest that monkeypatches ctx.gh proves nothing — it just confirms
        # the author's own assumptions about the data shape.
        targets = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for tgt in targets:
            chain = _attr_chain(tgt)
            if chain and (chain == "ctx.gh" or chain.startswith("ctx.gh.")):
                raise SandboxError(
                    f"reassigning the platform handle is not allowed: {chain} "
                    "(selftests must exercise the real ctx.gh, not a stub)")


def compile_capability(source: str, fn_name: str = "capability") -> Callable:
    """Validate then compile `source`, returning the callable `fn_name`."""
    validate(source)
    sandbox_globals = {"__builtins__": SAFE_BUILTINS, "__name__": "synthesised",
                       **SAFE_MODULES}
    try:
        exec(compile(source, "<synthesised>", "exec"), sandbox_globals)
    except Exception as e:
        raise SandboxError(f"failed to load: {e}")
    fn = sandbox_globals.get(fn_name)
    if not callable(fn):
        raise SandboxError(f"no callable `{fn_name}` defined")
    return fn
