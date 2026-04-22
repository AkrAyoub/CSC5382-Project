from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def _pick_instances(data_raw: Path) -> List[Path]:
    preferred = ["cap71.txt", "cap74.txt", "cap131.txt"]
    picks: List[Path] = []
    for name in preferred:
        path = data_raw / name
        if path.exists():
            picks.append(path)

    if len(picks) >= 3:
        return picks[:3]

    for path in sorted(data_raw.glob("cap*.txt")):
        if path not in picks:
            picks.append(path)
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


def test_baseline_scenario(instances: List[Path], optfile: Optional[Path], tol: float) -> None:
    from src.baseline_solver import run_baseline

    print("\n[E2E] Baseline scenario on 3 instances")
    for path in instances:
        res = run_baseline(str(path), str(optfile) if optfile else None)
        if res.best_known is not None and res.gap_percent is not None:
            if abs(res.gap_percent) > (tol * 100.0):
                raise AssertionError(
                    f"Baseline gap too large for {path.name}: gap%={res.gap_percent} tol%={tol*100.0}"
                )

        print(
            f"  {path.name}: obj={res.objective:.3f} "
            f"best={res.best_known if res.best_known is not None else 'N/A'} "
            f"gap%={res.gap_percent if res.gap_percent is not None else 'N/A'} "
            f"rt={res.runtime_s:.3f}s"
        )

    print("[E2E] Baseline scenario - OK")


def test_optional_llm_verification(instance_path: Path, optfile: Optional[Path], tol: float) -> None:
    from src.pipeline_trace import PipelineTrace
    from src.poc_pipeline import run_poc_scenario

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Optional LLM verification requires GROQ_API_KEY. Set it (and optionally GROQ_MODEL) then rerun."
        )

    print("\n[E2E] Optional LLM verification scenario (1 instance)")
    trace = PipelineTrace()
    result = run_poc_scenario(
        str(instance_path),
        str(optfile) if optfile else None,
        enable_llm=True,
        trace=trace,
    )

    if result.llm.status != "OK":
        print("\n[E2E] Optional LLM verification failed. Trace steps:")
        for step in trace.steps:
            print(f" - {step.name}: {step.status} {f'err={step.error}' if step.error else ''}")
        raise AssertionError(f"Optional LLM verification failed: {result.llm.error}")

    if result.llm.objective is None:
        raise AssertionError("LLM verification returned OK but llm.objective is None")

    _assert_close(result.llm.objective, result.baseline.objective, tol, "LLM vs baseline objective")
    if result.llm.gap_vs_baseline_pct is not None and abs(result.llm.gap_vs_baseline_pct) > (tol * 100.0):
        raise AssertionError(
            f"LLM gap vs baseline too large: {result.llm.gap_vs_baseline_pct}% tol%={tol*100.0}"
        )

    print(
        f"  {instance_path.name}: baseline={result.baseline.objective:.3f} "
        f"llm={result.llm.objective:.3f} "
        f"gap%={result.llm.gap_vs_baseline_pct:.6f} "
        f"backend={result.llm.backend_name} "
        f"model={result.llm.model_name}"
    )
    print("[E2E] Optional LLM verification - OK")


def main() -> None:
    this_file = Path(__file__).resolve()
    milestone_root = this_file.parent.parent
    project_root = milestone_root.parent

    data_raw = project_root / "data" / "raw"
    if not data_raw.exists():
        raise RuntimeError(f"data/raw not found at expected location: {data_raw}")

    optfile = data_raw / "uncapopt.txt"
    optfile_path = optfile if optfile.exists() else None

    sys.path.insert(0, str(milestone_root))

    tol = float(os.getenv("E2E_TOL", "1e-6"))
    instances = _pick_instances(data_raw)

    test_baseline_scenario(instances, optfile_path, tol)

    enable_llm = os.getenv("E2E_ENABLE_LLM", "0").strip() == "1"
    if enable_llm:
        llm_instance_name = os.getenv("E2E_LLM_INSTANCE", instances[0].name)
        llm_instance = data_raw / llm_instance_name
        if not llm_instance.exists():
            raise RuntimeError(f"E2E_LLM_INSTANCE={llm_instance_name} not found in {data_raw}")
        test_optional_llm_verification(llm_instance, optfile_path, tol)
    else:
        print("\n[E2E] Optional LLM verification skipped. Set E2E_ENABLE_LLM=1 to run it.")

    print("\n ALL E2E TESTS PASSED")


if __name__ == "__main__":
    main()
