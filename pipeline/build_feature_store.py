from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_REPO_DIR = PROJECT_ROOT / "feature_repo"
APPLY_SCRIPT = FEATURE_REPO_DIR / "apply_repo.py"


def main() -> None:
    print("Applying Feast repository definitions...")

    result = subprocess.run(
        [sys.executable, str(APPLY_SCRIPT)],
        cwd=FEATURE_REPO_DIR,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Feature store build failed: {APPLY_SCRIPT}")

    print("Feature store build completed successfully.")


if __name__ == "__main__":
    main()
