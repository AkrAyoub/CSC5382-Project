from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from m4_model_dev.data.benchmark import discover_raw_instances, parse_orlib_uncap, parse_uncapopt
from m4_model_dev.data.build_reference_solutions import build_reference_solutions
from m4_model_dev.paths import DATA_RAW_DIR, M4_DATASETS_DIR, M4_REFERENCE_DIR


OPTIMA_PATH = DATA_RAW_DIR / "uncapopt.txt"


@dataclass(frozen=True)
class BenchmarkDatasetRow:
    instance_id: str
    instance_path: str
    facility_count_m: int
    customer_count_n: int
    best_known: float | None
    has_best_known: int


def build_benchmark_dataset(
    reference_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    reference_path = reference_path or (M4_REFERENCE_DIR / "reference_solutions.csv")
    output_path = output_path or (M4_DATASETS_DIR / "benchmark_instances.csv")

    if not reference_path.exists():
        reference_path = build_reference_solutions(reference_path)

    optima = parse_uncapopt(OPTIMA_PATH) if OPTIMA_PATH.exists() else {}
    rows: list[BenchmarkDatasetRow] = []

    for instance_path in discover_raw_instances(DATA_RAW_DIR):
        instance = parse_orlib_uncap(instance_path)
        best_known = optima.get(instance.instance_id)
        rows.append(
            BenchmarkDatasetRow(
                instance_id=instance.instance_id,
                instance_path=str(instance.instance_path),
                facility_count_m=instance.facility_count_m,
                customer_count_n=instance.customer_count_n,
                best_known=best_known,
                has_best_known=1 if best_known is not None else 0,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(row) for row in rows]).sort_values(
        ["facility_count_m", "customer_count_n", "instance_id"]
    ).to_csv(output_path, index=False)
    return output_path
