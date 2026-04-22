from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from m4_model_dev.data.benchmark import parse_orlib_uncap


@dataclass(frozen=True)
class GeneratedRunResult:
    objective: float
    open_facilities: list[int]
    assignments: list[int]


def _safe_open(file, mode="r", *args, **kwargs):
    if mode not in ("r", "rt"):
        raise RuntimeError("File writes are blocked in generated code.")
    return open(file, mode, *args, **kwargs)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    allowed_import_roots = {"pathlib", "typing", "ortools", "re"}
    if root not in allowed_import_roots:
        raise ImportError(f"Import blocked: {name}")
    return __import__(name, globals, locals, fromlist, level)


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid index")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except Exception:
            return int(float(stripped))
    return int(float(value))


def run_generated_solver(code: str, instance_path: str) -> GeneratedRunResult:
    lowered = code.lower()
    banned_snippets = [
        "import os",
        "import sys",
        "import subprocess",
        "from os",
        "from sys",
        "eval(",
        "exec(",
        "print(",
    ]
    for snippet in banned_snippets:
        if snippet in lowered:
            raise RuntimeError(f"Generated code contains banned snippet: {snippet}")

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

    env: dict[str, Any] = {"__builtins__": allowed_builtins, "__name__": "__generated__"}
    exec(code, env, env)

    if "solve" not in env or not callable(env["solve"]):
        raise RuntimeError("Generated code does not define solve(instance_path: str) -> dict")

    out = env["solve"](instance_path)
    if not isinstance(out, dict):
        raise RuntimeError("solve() must return a dict")
    for key in ("objective", "open_facilities", "assignments"):
        if key not in out:
            raise RuntimeError(f"solve() output missing key: {key}")

    objective = float(out["objective"])
    open_facilities = [_to_int(value) for value in out["open_facilities"]]
    assignments = [_to_int(value) for value in out["assignments"]]

    instance = parse_orlib_uncap(Path(instance_path))
    m = instance.facility_count_m
    n = instance.customer_count_n

    if objective < 0.0:
        raise RuntimeError("Invalid objective < 0")
    if len(assignments) != n:
        raise RuntimeError(f"Assignments length mismatch: got {len(assignments)} expected {n}")

    open_set = set(open_facilities)
    if len(open_set) != len(open_facilities):
        raise RuntimeError("open_facilities contains duplicates")

    for facility_id in open_facilities:
        if facility_id < 0 or facility_id >= m:
            raise RuntimeError(f"Open facility index out of range: {facility_id} (m={m})")

    for customer_idx, facility_id in enumerate(assignments):
        if facility_id < 0 or facility_id >= m:
            raise RuntimeError(f"Assignment out of range: assignments[{customer_idx}]={facility_id} (m={m})")
        if facility_id not in open_set:
            raise RuntimeError(
                f"Invalid solution: assignments[{customer_idx}]={facility_id} but the facility is not open."
            )

    return GeneratedRunResult(
        objective=objective,
        open_facilities=open_facilities,
        assignments=assignments,
    )
