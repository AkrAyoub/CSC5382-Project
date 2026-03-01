from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from src.baseline_solver import run_baseline
from src.exec_generated import GeneratedRunResult, run_generated_solver
from src.llm_generate import SYSTEM, generate_solver_code

try:
    from src.pipeline_trace import PipelineTrace
except Exception:  # pragma: no cover
    PipelineTrace = None  # type: ignore


@dataclass
class CompareResult:
    baseline_objective: float
    best_known: Optional[float]
    baseline_gap_pct: Optional[float]

    llm_status: str  # "OK" or "FAIL"
    llm_error: Optional[str]
    llm_objective: Optional[float]
    llm_gap_vs_baseline_pct: Optional[float]
    llm_open_facilities: list[int]
    llm_assignments: list[int]
    llm_model: str


def _extract_tag(err_text: str) -> str:
    """
    Pulls the validator tag like 'MISSING_FIXED_COEF' from:
    RuntimeError: Groq call failed ...: MISSING_FIXED_COEF: ...
    """
    m = re.search(r"\b([A-Z0-9_]{6,})\b\s*:", err_text)
    return m.group(1) if m else "UNKNOWN"


def _signature_from_raw(raw_preview: str) -> str:
    """
    Cheap similarity signature: normalize whitespace, drop digits, keep first N chars.
    This catches 'same code pattern' even if tiny edits differ.
    """
    s = (raw_preview or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\d+", "0", s)
    return s[:500]


def _build_feedback(last_err: str, repeated: bool) -> str:
    e = (last_err or "")
    tag = _extract_tag(e).lower()
    fixes: list[str] = []

    # Escalation if repeating same mistake class
    if repeated:
        fixes.append(
            "CRITICAL: You repeated the same mistake. You MUST change approach."
        )

    if "missing_fixed_coef" in tag or "missing_assign_coef" in tag:
        fixes.append(
            "Objective MUST be built ONLY with SetCoefficient loops (NO objective.Add, NO solver.Sum)."
        )
        fixes.append(
            "Use EXACT template:\n"
            "obj = solver.Objective();\n"
            "for i in range(m): obj.SetCoefficient(y[i], fixed[i])\n"
            "for j in range(n):\n"
            "  for i in range(m): obj.SetCoefficient(x[j][i], cost[j][i])\n"
            "obj.SetMinimization()"
        )

    if "forbidden_solver_constructor" in tag or "missing_cbc_createsolver" in tag:
        fixes.append("- MUST use: solver = pywraplp.Solver.CreateSolver('CBC')")

    if "solver_sum_used" in tag:
        fixes.append("- NEVER use solver.Sum. Use Python sum(...).")

    if "bad_facility_parse" in tag:
        fixes.append("- Facility parse must read cap then f inside SAME for i in range(m) loop.")

    if "missing_demand_parse" in tag:
        fixes.append("- Must parse demand inside for j in range(n) loop then read m costs.")

    if "wrong_direction" in tag:
        fixes.append("- Objective must be MINIMIZATION only.")

    if not fixes:
        fixes.append("- Regenerate from scratch and follow the exact parsing + MILP template.")

    return (
        "Your previous attempt failed.\n"
        f"Error:\n{e}\n\n"
        "Apply ALL fixes below in the new code:\n"
        + "\n".join(f"- {x}" if not x.startswith("-") else x for x in fixes)
        + "\n"
    )


def compare_baseline_vs_llm(
    instance_path: str,
    optfile_path: Optional[str] = None,
    trace: Optional["PipelineTrace"] = None,
) -> CompareResult:
    s1 = trace.start("1) Baseline solve (deterministic OR-Tools MILP)") if trace else None
    base = run_baseline(instance_path, optfile_path)
    if s1:
        s1.artifacts["instance_path"] = instance_path
        s1.artifacts["baseline_objective"] = base.objective
        s1.artifacts["best_known"] = base.best_known
        s1.artifacts["baseline_gap_pct"] = base.gap_percent
        s1.artifacts["baseline_runtime_s"] = base.runtime_s
        s1.artifacts["baseline_open_facilities"] = base.open_facilities
        s1.artifacts["baseline_assignments_preview"] = base.assignments[:20]
        s1.end_ok()

    last_err: Optional[str] = None
    last_model: str = "(unknown)"

    # repetition tracking
    last_fail_key: Optional[Tuple[str, str]] = None
    repeat_count = 0

    attempt = 1
    while attempt <= 4:
        s2 = trace.start(f"2) LLM generate OR-Tools solver code (attempt {attempt})") if trace else None

        repeated = repeat_count >= 1
        extra = _build_feedback(last_err, repeated=repeated) if last_err else None

        try:
            gen = generate_solver_code(extra_instructions=extra, max_attempts=2)
            last_model = gen.model
            if s2:
                s2.artifacts["llm_model"] = gen.model
                s2.artifacts["system_prompt"] = SYSTEM
                s2.artifacts["user_prompt"] = gen.user_prompt
                s2.artifacts["raw_llm_output"] = gen.raw_text
                s2.artifacts["generated_code"] = gen.code
                s2.end_ok()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            last_err = msg
            if s2:
                s2.artifacts["error_text"] = msg
                s2.end_fail(e)

            low = msg.lower()
            if "rate limit" in low or "error code: 429" in low or "429" in low:
                continue

            # try to detect repetition: tag + signature from RAW_PREVIEW if present
            tag = _extract_tag(msg)
            raw_preview = ""
            m = re.search(r"RAW_PREVIEW:\s*(.*)$", msg, flags=re.DOTALL)
            if m:
                raw_preview = m.group(1)
            sig = _signature_from_raw(raw_preview)
            key = (tag, sig)

            if last_fail_key == key:
                repeat_count += 1
            else:
                repeat_count = 0
                last_fail_key = key

            attempt += 1
            continue

        s3 = trace.start(f"3) Execute generated solver in sandbox (attempt {attempt})") if trace else None
        try:
            llm_run: GeneratedRunResult = run_generated_solver(gen.code, instance_path)
            if s3:
                s3.artifacts["llm_objective"] = llm_run.objective
                s3.artifacts["llm_open_facilities"] = llm_run.open_facilities
                s3.artifacts["llm_assignments_preview"] = llm_run.assignments[:20]
                s3.end_ok()

            s4 = trace.start("4) Compare LLM vs baseline") if trace else None
            gap_vs_base = (llm_run.objective - base.objective) / max(1.0, abs(base.objective)) * 100.0
            if s4:
                s4.artifacts["llm_gap_vs_baseline_pct"] = gap_vs_base
                s4.end_ok()

            return CompareResult(
                baseline_objective=base.objective,
                best_known=base.best_known,
                baseline_gap_pct=base.gap_percent,
                llm_status="OK",
                llm_error=None,
                llm_objective=llm_run.objective,
                llm_gap_vs_baseline_pct=gap_vs_base,
                llm_open_facilities=llm_run.open_facilities,
                llm_assignments=llm_run.assignments,
                llm_model=gen.model,
            )

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if s3:
                s3.end_fail(e)
            attempt += 1
            continue

    return CompareResult(
        baseline_objective=base.objective,
        best_known=base.best_known,
        baseline_gap_pct=base.gap_percent,
        llm_status="FAIL",
        llm_error=last_err or "Unknown failure",
        llm_objective=None,
        llm_gap_vs_baseline_pct=None,
        llm_open_facilities=[],
        llm_assignments=[],
        llm_model=last_model,
    )