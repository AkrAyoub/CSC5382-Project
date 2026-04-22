from __future__ import annotations

from pathlib import Path


M4_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = M4_ROOT.parent
M3_ROOT = REPO_ROOT / "Milestone_3-Data_Prep"
M2_ROOT = REPO_ROOT / "Milestone_2-PoC"
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"

M4_DATA_DIR = M4_ROOT / "data"
M4_REFERENCE_DIR = M4_DATA_DIR / "reference"
M4_DATASETS_DIR = M4_DATA_DIR / "datasets"
M4_SPLITS_DIR = M4_DATA_DIR / "splits"
M4_SFT_DIR = M4_DATA_DIR / "sft"
M4_ARTIFACTS_DIR = M4_ROOT / "artifacts"
M4_REPORTS_DIR = M4_ROOT / "reports"
M4_GENERATED_CODE_DIR = M4_REPORTS_DIR / "generated_code"
M4_EVAL_RESULTS_DIR = M4_REPORTS_DIR / "evaluation"
M4_CONFIGS_DIR = M4_ROOT / "configs"
M4_MLFLOW_DIR = M4_ROOT / "mlruns"
M4_ZEN_CONFIG_DIR = M4_ROOT / ".zen"
M4_ZEN_LOCAL_STORE_DIR = M4_ROOT.parent.parent / ".zen_local"

M3_FEATURES_DIR = M3_ROOT / "data" / "features"
M3_INTERIM_DIR = M3_ROOT / "data" / "interim"
M3_PROCESSED_DIR = M3_ROOT / "data" / "processed"


def ensure_runtime_dirs() -> None:
    runtime_dirs = [
        M4_DATA_DIR,
        M4_REFERENCE_DIR,
        M4_DATASETS_DIR,
        M4_SPLITS_DIR,
        M4_SFT_DIR,
        M4_ARTIFACTS_DIR,
        M4_REPORTS_DIR,
        M4_GENERATED_CODE_DIR,
        M4_EVAL_RESULTS_DIR,
        M4_MLFLOW_DIR,
        M4_ZEN_CONFIG_DIR,
        M4_ZEN_LOCAL_STORE_DIR,
    ]
    for path in runtime_dirs:
        path.mkdir(parents=True, exist_ok=True)
