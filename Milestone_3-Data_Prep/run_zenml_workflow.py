from __future__ import annotations

from pathlib import Path
import sys

try:
    from pipeline.zenml_pipeline import run_zenml_data_pipeline
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline.zenml_pipeline import run_zenml_data_pipeline


def main() -> None:
    status = run_zenml_data_pipeline()
    print("=== Milestone 3 ZenML Workflow Status ===")
    print(f"success: {status['success']}")
    print(f"details: {status['details']}")


if __name__ == "__main__":
    main()
