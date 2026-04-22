from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import sys as _sys

try:
    from ..paths import FEATURE_REPO_DIR
except ImportError:
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import FEATURE_REPO_DIR

APPLY_SCRIPT = FEATURE_REPO_DIR / "apply_repo.py"


def run_feature_store_build() -> dict[str, str]:
    print("Applying Feast repository definitions...")

    result = subprocess.run(
        [sys.executable, str(APPLY_SCRIPT)],
        cwd=FEATURE_REPO_DIR,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Feature store build failed: {APPLY_SCRIPT}")

    print("Feature store build completed successfully.")

    return {
        "feature_repo_dir": str(FEATURE_REPO_DIR),
        "apply_script": str(APPLY_SCRIPT),
        "status": "success",
    }


def main() -> None:
    run_feature_store_build()


if __name__ == "__main__":
    main()
