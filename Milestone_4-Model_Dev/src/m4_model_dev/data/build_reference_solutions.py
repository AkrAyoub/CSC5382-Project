from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from m4_model_dev.data.benchmark import (
    ReferenceSolveResult,
    discover_raw_instances,
    parse_orlib_uncap,
    parse_uncapopt,
    solve_reference_cbc,
)
from m4_model_dev.paths import DATA_RAW_DIR, M4_REFERENCE_DIR


OPTIMA_PATH = DATA_RAW_DIR / "uncapopt.txt"


@dataclass(frozen=True)
class ReferenceSolutionRow:
    instance_id: str
    instance_path: str
    facility_count_m: int
    customer_count_n: int
    objective: float
    best_known: float | None
    gap_percent: float | None
    runtime_s: float
    open_facility_count: int
    open_facilities: str


def _result_to_row(result: ReferenceSolveResult, instance_path: Path, m: int, n: int) -> ReferenceSolutionRow:
    return ReferenceSolutionRow(
        instance_id=result.instance_id,
        instance_path=str(instance_path),
        facility_count_m=m,
        customer_count_n=n,
        objective=result.objective,
        best_known=result.best_known,
        gap_percent=result.gap_percent,
        runtime_s=result.runtime_s,
        open_facility_count=len(result.open_facilities),
        open_facilities=" ".join(str(value) for value in result.open_facilities),
    )


def build_reference_solutions(output_path: Path | None = None) -> Path:
    output_path = output_path or (M4_REFERENCE_DIR / "reference_solutions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    optima = parse_uncapopt(OPTIMA_PATH) if OPTIMA_PATH.exists() else {}
    rows: list[ReferenceSolutionRow] = []

    for instance_path in discover_raw_instances(DATA_RAW_DIR):
        instance = parse_orlib_uncap(instance_path)
        result = solve_reference_cbc(instance, best_known=optima.get(instance.instance_id))
        rows.append(
            _result_to_row(
                result=result,
                instance_path=instance.instance_path,
                m=instance.facility_count_m,
                n=instance.customer_count_n,
            )
        )

    pd.DataFrame([asdict(row) for row in rows]).sort_values("instance_id").to_csv(output_path, index=False)
    return output_path
