from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step.name} ({step.script_path})")

    print(f"COMPLETED: {step.name}")


def main() -> None:
    steps = [
        PipelineStep("Raw Data Ingestion", PROJECT_ROOT / "pipeline" / "ingest_data.py"),
        PipelineStep("Data Preprocessing", PROJECT_ROOT / "pipeline" / "preprocess_data.py"),
        PipelineStep("Feature Engineering", PROJECT_ROOT / "pipeline" / "engineer_features.py"),
        PipelineStep("Data Validation", PROJECT_ROOT / "pipeline" / "validate_data.py"),
    ]

    print("Starting Milestone 3 full data pipeline...")

    for step in steps:
        run_step(step)

    print(f"\n{'=' * 80}")
    print("MILESTONE 3 DATA PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")
    print("\nGenerated artifact locations:")
    print(f"- Raw data:        {PROJECT_ROOT / 'data' / 'raw'}")
    print(f"- Interim data:    {PROJECT_ROOT / 'data' / 'interim'}")
    print(f"- Processed data:  {PROJECT_ROOT / 'data' / 'processed'}")
    print(f"- Features:        {PROJECT_ROOT / 'data' / 'features'}")
    print(f"- Validation:      {PROJECT_ROOT / 'reports' / 'validation'}")
    print(f"- Statistics:      {PROJECT_ROOT / 'reports' / 'stats'}")


if __name__ == "__main__":
    main()
