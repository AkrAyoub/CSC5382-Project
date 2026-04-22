from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import sys as _sys

try:
    from ..paths import FEATURES_DATA_DIR, INTERIM_DATA_DIR, M3_ROOT, PROCESSED_DATA_DIR, STATS_DIR, VALIDATION_DIR
except ImportError:
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import FEATURES_DATA_DIR, INTERIM_DATA_DIR, M3_ROOT, PROCESSED_DATA_DIR, STATS_DIR, VALIDATION_DIR


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script_path: Path


def run_step(step: PipelineStep) -> None:
    print(f"\n{'=' * 80}")
    print(f"RUNNING STEP: {step.name}")
    print(f"{'=' * 80}")

    result = subprocess.run(
        [sys.executable, str(step.script_path)],
        cwd=M3_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step.name} ({step.script_path})")

    print(f"COMPLETED: {step.name}")


def main() -> None:
    steps = [
        PipelineStep("Raw Data Ingestion", M3_ROOT / "pipeline" / "ingest_data.py"),
        PipelineStep("Data Preprocessing", M3_ROOT / "pipeline" / "preprocess_data.py"),
        PipelineStep("Feature Engineering", M3_ROOT / "pipeline" / "engineer_features.py"),
        PipelineStep("Data Validation", M3_ROOT / "pipeline" / "validate_data.py"),
    ]

    print("Starting Milestone 3 full data pipeline...")

    for step in steps:
        run_step(step)

    print(f"\n{'=' * 80}")
    print("MILESTONE 3 DATA PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")
    print("\nGenerated artifact locations:")
    print(f"- Interim data:    {INTERIM_DATA_DIR}")
    print(f"- Processed data:  {PROCESSED_DATA_DIR}")
    print(f"- Features:        {FEATURES_DATA_DIR}")
    print(f"- Validation:      {VALIDATION_DIR}")
    print(f"- Statistics:      {STATS_DIR}")


if __name__ == "__main__":
    main()
