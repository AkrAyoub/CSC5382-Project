from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from paths import FEATURE_REPO_DIR, M3_ROOT
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import FEATURE_REPO_DIR, M3_ROOT


def run_step(step_name: str, cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n{'=' * 80}")
    print(f"RUNNING STEP: {step_name}")
    print(f"{'=' * 80}")

    result = subprocess.run(
        cmd,
        cwd=cwd or M3_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name}")

    print(f"COMPLETED: {step_name}")


def main() -> None:
    print("Starting full Milestone 3 workflow...")

    run_step(
        "Milestone 3 Data Pipeline",
        [sys.executable, str(M3_ROOT / "pipeline" / "run_data_pipeline.py")],
    )

    run_step(
        "Feast Repository Apply",
        [sys.executable, str(FEATURE_REPO_DIR / "apply_repo.py")],
        cwd=FEATURE_REPO_DIR,
    )

    run_step(
        "Feast Feature Retrieval Demo",
        [sys.executable, str(FEATURE_REPO_DIR / "run_feature_store_demo.py")],
        cwd=FEATURE_REPO_DIR,
    )

    print(f"\n{'=' * 80}")
    print("FULL MILESTONE 3 WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
