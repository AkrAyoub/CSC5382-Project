from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _find_project_root(start: Path) -> Path:

    cur = start.resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "data" / "raw").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate project root containing src/ and data/raw/.")


def _pick_instances(data_raw: Path) -> List[Path]:

    preferred = ["cap71.txt", "cap74.txt", "cap131.txt"]
    picks: List[Path] = []
    for name in preferred:
        p = data_raw / name
        if p.exists():
            picks.append(p)

    if len(picks) >= 3:
        return picks[:3]

    # fallback: any cap*.txt
    caps = sorted(data_raw.glob("cap*.txt"))
    for p in caps:
        if p not in picks:
            picks.append(p)
        if len(picks) >= 3:
            break

    if len(picks) < 3:
        raise RuntimeError(f"Need at least 3 instances in {data_raw} (cap*.txt). Found {len(picks)}")
    return picks[:3]


def _assert_close(a: float, b: float, tol: float, label: str) -> None:
    denom = max(1.0, abs(b))
    rel = abs(a - b) / denom
    if rel > tol:
        raise AssertionError(f"{label}: not close. a={a} b={b} rel_err={rel} tol={tol}")


def test_mode_a_baseline(instances: List[Path], optfile: Optional[Path], tol: float) -> None:
    from src.baseline_solver import run_baseline

    print("\n[E2E] Mode A — Baseline (OR-Tools CBC) on 3 instances")
    for p in instances:
        res = run_baseline(str(p), str(optfile) if optfile else None)

        # If best-known exists, require near-zero gap
        if res.best_known is not None and res.gap_percent is not None:
            if abs(res.gap_percent) > (tol * 100.0):
                raise AssertionError(
                    f"Baseline gap too large for {p.name}: gap%={res.gap_percent} tol%={tol*100.0}"
                )

        print(
            f"  {p.name}: obj={res.objective:.3f} "
            f"best={res.best_known if res.best_known is not None else 'N/A'} "
            f"gap%={res.gap_percent if res.gap_percent is not None else 'N/A'} "
            f"rt={res.runtime_s:.3f}s"
        )

    print("[E2E] Mode A — OK")


def test_mode_b_llm(instance_path: Path, optfile: Optional[Path], tol: float) -> None:
    """
    Runs Mode B once (LLM codegen + sandbox execute + compare vs baseline).
    Requires GROQ_API_KEY to be set.
    """
    from src.compare import compare_baseline_vs_llm
    from src.pipeline_trace import PipelineTrace

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Mode B test requires GROQ_API_KEY env var. "
            "Set it (and optionally GROQ_MODEL) then rerun."
        )

    print("\n[E2E] Mode B — LLM generates solver + verify vs baseline (1 instance)")
    trace = PipelineTrace()
    result = compare_baseline_vs_llm(str(instance_path), str(optfile) if optfile else None, trace)

    if result.llm_status != "OK":
        # print trace summary to help debugging in CLI
        print("\n[E2E] Mode B failed. Trace steps:")
        for s in trace.steps:
            print(f" - {s.name}: {s.status} {f'err={s.error}' if s.error else ''}")
        raise AssertionError(f"Mode B failed: {result.llm_error}")

    if result.llm_objective is None:
        raise AssertionError("Mode B returned OK but llm_objective is None")

    _assert_close(result.llm_objective, result.baseline_objective, tol, "LLM vs baseline objective")
    if result.llm_gap_vs_baseline_pct is not None and abs(result.llm_gap_vs_baseline_pct) > (tol * 100.0):
        raise AssertionError(
            f"LLM gap vs baseline too large: {result.llm_gap_vs_baseline_pct}% tol%={tol*100.0}"
        )

    print(
        f"  {instance_path.name}: baseline={result.baseline_objective:.3f} "
        f"llm={result.llm_objective:.3f} "
        f"gap%={result.llm_gap_vs_baseline_pct:.6f} "
        f"model={result.llm_model}"
    )
    print("[E2E] Mode B — OK")


def main() -> None:

    this_file = Path(__file__).resolve()

    milestone_root = this_file.parent.parent

    project_root = milestone_root.parent

    # data/raw/
    data_raw = project_root / "data" / "raw"

    if not data_raw.exists():
        raise RuntimeError(f"data/raw not found at expected location: {data_raw}")

    optfile = data_raw / "uncapopt.txt"
    optfile_path = optfile if optfile.exists() else None

    sys.path.insert(0, str(milestone_root))

    tol = float(os.getenv("E2E_TOL", "1e-6"))

    instances = _pick_instances(data_raw)

    # Mode A
    test_mode_a_baseline(instances, optfile_path, tol)

    # Mode B
    mode_b_name = os.getenv("E2E_MODE_B_INSTANCE", instances[0].name)
    mode_b_inst = data_raw / mode_b_name
    if not mode_b_inst.exists():
        raise RuntimeError(f"E2E_MODE_B_INSTANCE={mode_b_name} not found in {data_raw}")

    test_mode_b_llm(mode_b_inst, optfile_path, tol)

    print("\n ALL E2E TESTS PASSED")


if __name__ == "__main__":
    main()