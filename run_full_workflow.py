from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(step_name: str, cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n{'=' * 80}")
    print(f"RUNNING STEP: {step_name}")
    print(f"{'=' * 80}")

    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name}")

    print(f"COMPLETED: {step_name}")


def main() -> None:
    print("Starting full Milestone 3 workflow...")

    run_step(
        "Milestone 3 Data Pipeline",
        [sys.executable, str(PROJECT_ROOT / "pipeline" / "run_data_pipeline.py")],
    )

    run_step(
        "Feast Repository Apply",
        [sys.executable, str(PROJECT_ROOT / "feature_repo" / "apply_repo.py")],
        cwd=PROJECT_ROOT / "feature_repo",
    )

    run_step(
        "Feast Feature Retrieval Demo",
        [sys.executable, str(PROJECT_ROOT / "feature_repo" / "run_feature_store_demo.py")],
        cwd=PROJECT_ROOT / "feature_repo",
    )

    print(f"\n{'=' * 80}")
    print("FULL MILESTONE 3 WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()