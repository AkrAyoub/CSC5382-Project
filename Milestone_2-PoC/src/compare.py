from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.poc_pipeline import run_poc_scenario

try:
    from src.pipeline_trace import PipelineTrace
except Exception:  # pragma: no cover
    PipelineTrace = None  # type: ignore


@dataclass
class CompareResult:
    baseline_objective: float
    best_known: Optional[float]
    baseline_gap_pct: Optional[float]
    llm_status: str
    llm_error: Optional[str]
    llm_objective: Optional[float]
    llm_gap_vs_baseline_pct: Optional[float]
    llm_open_facilities: list[int]
    llm_assignments: list[int]
    llm_backend: str
    llm_model: str


def compare_baseline_vs_llm(
    instance_path: str,
    optfile_path: Optional[str] = None,
    trace: Optional["PipelineTrace"] = None,
) -> CompareResult:
    scenario = run_poc_scenario(
        instance_path,
        optfile_path,
        enable_llm=True,
        trace=trace,
    )
    return CompareResult(
        baseline_objective=scenario.baseline.objective,
        best_known=scenario.baseline.best_known,
        baseline_gap_pct=scenario.baseline.gap_percent,
        llm_status=scenario.llm.status,
        llm_error=scenario.llm.error,
        llm_objective=scenario.llm.objective,
        llm_gap_vs_baseline_pct=scenario.llm.gap_vs_baseline_pct,
        llm_open_facilities=scenario.llm.open_facilities,
        llm_assignments=scenario.llm.assignments,
        llm_backend=scenario.llm.backend_name or "(unknown)",
        llm_model=scenario.llm.model_name or "(unknown)",
    )
