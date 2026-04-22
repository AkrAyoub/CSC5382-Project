from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.baseline_solver import BaselineResult, run_baseline
from src.exec_generated import GeneratedRunResult, run_generated_solver
from src.llm_generate import SYSTEM, generate_solver_code

try:
    from src.pipeline_trace import PipelineTrace
except Exception:  # pragma: no cover
    PipelineTrace = None  # type: ignore


@dataclass
class LLMRunOutcome:
    enabled: bool
    status: str  # OK | FAIL | SKIPPED
    error: Optional[str] = None
    objective: Optional[float] = None
    gap_vs_baseline_pct: Optional[float] = None
    open_facilities: list[int] = field(default_factory=list)
    assignments: list[int] = field(default_factory=list)
    backend_name: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class PoCScenarioResult:
    instance_path: str
    baseline: BaselineResult
    llm: LLMRunOutcome


def _extract_tag(err_text: str) -> str:
    match = re.search(r"\b([A-Z0-9_]{6,})\b\s*:", err_text)
    return match.group(1) if match else "UNKNOWN"


def _signature_from_raw(raw_preview: str) -> str:
    normalized = (raw_preview or "").lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\d+", "0", normalized)
    return normalized[:500]


def _build_feedback(last_err: str, repeated: bool) -> str:
    tag = _extract_tag(last_err).lower()
    fixes: list[str] = []

    if repeated:
        fixes.append("CRITICAL: You repeated the same mistake. You MUST change approach.")

    if "missing_fixed_coef" in tag or "missing_assign_coef" in tag:
        fixes.append("Objective MUST be built ONLY with SetCoefficient loops (NO objective.Add, NO solver.Sum).")
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
        f"Error:\n{last_err}\n\n"
        "Apply ALL fixes below in the new code:\n"
        + "\n".join(f"- {item}" if not item.startswith("-") else item for item in fixes)
        + "\n"
    )


def _run_llm_verification(
    instance_path: str,
    baseline: BaselineResult,
    trace: Optional["PipelineTrace"] = None,
) -> LLMRunOutcome:
    last_err: Optional[str] = None
    last_model = "(unknown)"
    last_backend = "(unknown)"
    last_fail_key: Optional[Tuple[str, str]] = None
    repeat_count = 0
    attempt = 1

    while attempt <= 4:
        generate_step = trace.start(f"2) Generate solver code with off-the-shelf LLM (attempt {attempt})") if trace else None
        repeated = repeat_count >= 1
        extra = _build_feedback(last_err, repeated=repeated) if last_err else None

        try:
            gen = generate_solver_code(extra_instructions=extra)
            last_model = gen.model
            last_backend = gen.backend_name
            if generate_step:
                generate_step.artifacts["llm_backend"] = gen.backend_name
                generate_step.artifacts["llm_model"] = gen.model
                generate_step.artifacts["system_prompt"] = SYSTEM
                generate_step.artifacts["user_prompt"] = gen.user_prompt
                generate_step.artifacts["raw_llm_output"] = gen.raw_text
                generate_step.artifacts["generated_code"] = gen.code
                generate_step.end_ok()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            last_err = message
            if generate_step:
                generate_step.artifacts["error_text"] = message
                generate_step.end_fail(exc)

            lowered = message.lower()
            if "rate limit" in lowered or "error code: 429" in lowered or "429" in lowered:
                continue

            tag = _extract_tag(message)
            raw_preview = ""
            match = re.search(r"RAW_PREVIEW:\s*(.*)$", message, flags=re.DOTALL)
            if match:
                raw_preview = match.group(1)
            key = (tag, _signature_from_raw(raw_preview))

            if last_fail_key == key:
                repeat_count += 1
            else:
                repeat_count = 0
                last_fail_key = key

            attempt += 1
            continue

        execute_step = trace.start(f"3) Execute generated solver in sandbox (attempt {attempt})") if trace else None
        try:
            llm_run: GeneratedRunResult = run_generated_solver(gen.code, instance_path)
            if execute_step:
                execute_step.artifacts["llm_objective"] = llm_run.objective
                execute_step.artifacts["llm_open_facilities"] = llm_run.open_facilities
                execute_step.artifacts["llm_assignments_preview"] = llm_run.assignments[:20]
                execute_step.end_ok()

            compare_step = trace.start("4) Compare generated solver output vs deterministic baseline") if trace else None
            gap_vs_base = (llm_run.objective - baseline.objective) / max(1.0, abs(baseline.objective)) * 100.0
            if compare_step:
                compare_step.artifacts["llm_gap_vs_baseline_pct"] = gap_vs_base
                compare_step.end_ok()

            return LLMRunOutcome(
                enabled=True,
                status="OK",
                objective=llm_run.objective,
                gap_vs_baseline_pct=gap_vs_base,
                open_facilities=llm_run.open_facilities,
                assignments=llm_run.assignments,
                backend_name=gen.backend_name,
                model_name=gen.model,
            )
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if execute_step:
                execute_step.end_fail(exc)
            attempt += 1

    return LLMRunOutcome(
        enabled=True,
        status="FAIL",
        error=last_err or "Unknown LLM verification failure",
        backend_name=last_backend,
        model_name=last_model,
    )


def run_poc_scenario(
    instance_path: str,
    optfile_path: Optional[str] = None,
    *,
    enable_llm: bool = True,
    trace: Optional["PipelineTrace"] = None,
) -> PoCScenarioResult:
    baseline_step = trace.start("1) Solve the instance with the deterministic CBC baseline") if trace else None
    baseline = run_baseline(instance_path, optfile_path)
    if baseline_step:
        baseline_step.artifacts["instance_path"] = instance_path
        baseline_step.artifacts["baseline_objective"] = baseline.objective
        baseline_step.artifacts["best_known"] = baseline.best_known
        baseline_step.artifacts["baseline_gap_pct"] = baseline.gap_percent
        baseline_step.artifacts["baseline_runtime_s"] = baseline.runtime_s
        baseline_step.artifacts["baseline_open_facilities"] = baseline.open_facilities
        baseline_step.artifacts["baseline_assignments_preview"] = baseline.assignments[:20]
        baseline_step.end_ok()

    if not enable_llm:
        return PoCScenarioResult(
            instance_path=instance_path,
            baseline=baseline,
            llm=LLMRunOutcome(
                enabled=False,
                status="SKIPPED",
                error="LLM verification disabled for this run.",
            ),
        )

    llm_result = _run_llm_verification(instance_path, baseline, trace=trace)
    return PoCScenarioResult(instance_path=instance_path, baseline=baseline, llm=llm_result)
