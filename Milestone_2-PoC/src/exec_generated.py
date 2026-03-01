from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from src.baseline_solver import parse_orlib_uncap


@dataclass
class GeneratedRunResult:
    objective: float
    open_facilities: List[int]
    assignments: List[int]


def _safe_open(file, mode="r", *args, **kwargs):
    # Allow read-only access only
    if mode not in ("r", "rt"):
        raise RuntimeError("File writes are blocked in generated code.")
    return open(file, mode, *args, **kwargs)


def _to_int(x: Any) -> int:
    """
    Accept int/float/str like '7', '7.0', '7500.' and convert to int.
    If it's not safely convertible, raise.
    """
    if isinstance(x, bool):
        raise ValueError("bool is not a valid index")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return int(s)
        except Exception:
            return int(float(s))
    return int(float(x))


def run_generated_solver(code: str, instance_path: str) -> GeneratedRunResult:
    allowed_import_roots = {"pathlib", "typing", "ortools"}

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in allowed_import_roots:
            raise ImportError(f"Import blocked: {name}")
        return __import__(name, globals, locals, fromlist, level)

    allowed_builtins = {
        "__import__": _safe_import,
        "open": _safe_open,
        "Exception": Exception,
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "float": float,
        "int": int,
        "list": list,
        "dict": dict,
        "set": set,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "all": all,
        "any": any,
        "tuple": tuple,
        "next": next,
        "iter": iter,
    }

    env: Dict[str, Any] = {"__builtins__": allowed_builtins, "__name__": "__generated__"}

    # basic guardrails against nonsense and prompt injection
    lowered = code.lower()
    banned_snippets = [
        "import os",
        "import sys",
        "import subprocess",
        "from os",
        "from sys",
        "eval(",
        "exec(",
        "from src",
        "import src",
    ]
    for b in banned_snippets:
        if b in lowered:
            raise RuntimeError(f"Generated code contains banned snippet: {b}")

    # execute in sandbox
    exec(code, env, env)

    if "solve" not in env or not callable(env["solve"]):
        raise RuntimeError("Generated code does not define solve(instance_path: str) -> dict")

    out = env["solve"](instance_path)
    if not isinstance(out, dict):
        raise RuntimeError("solve() must return a dict")

    for k in ("objective", "open_facilities", "assignments"):
        if k not in out:
            raise RuntimeError(f"solve() output missing key: {k}")

    # --- Coerce outputs robustly ---
    obj = float(out["objective"])
    opens_raw = out["open_facilities"]
    assigns_raw = out["assignments"]

    if not isinstance(opens_raw, list) or not isinstance(assigns_raw, list):
        raise RuntimeError("open_facilities and assignments must be lists")

    opens = [_to_int(v) for v in opens_raw]
    assigns = [_to_int(v) for v in assigns_raw]

    # --- strict validation ---
    inst = parse_orlib_uncap(instance_path)
    m, n = inst.m, inst.n

    if not (obj >= 0.0):
        raise RuntimeError("Invalid objective < 0")

    if len(assigns) != n:
        raise RuntimeError(f"Assignments length mismatch: got {len(assigns)} expected {n}")

    opens_set: Set[int] = set(opens)
    if len(opens_set) != len(opens):
        raise RuntimeError("open_facilities contains duplicates")

    for i in opens:
        if i < 0 or i >= m:
            raise RuntimeError(f"Open facility index out of range: {i} (m={m})")

    for j, i in enumerate(assigns):
        if i < 0 or i >= m:
            raise RuntimeError(f"Assignment out of range: assignments[{j}]={i} (m={m})")
        # assignment implies open facility
        if i not in opens_set:
            raise RuntimeError(
                f"Invalid solution: assignments[{j}]={i} but facility not open"
            )

    return GeneratedRunResult(objective=obj, open_facilities=opens, assignments=assigns)