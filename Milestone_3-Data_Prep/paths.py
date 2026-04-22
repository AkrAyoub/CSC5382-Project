from __future__ import annotations

from pathlib import Path


M3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = M3_ROOT.parent

# Canonical shared raw input for the whole project.
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
UNCAPOPT_PATH = RAW_DATA_DIR / "uncapopt.txt"

# Milestone 3 owns the derived data layers and reporting outputs.
DATA_DIR = M3_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

SCHEMA_DIR = M3_ROOT / "schema"
REPORTS_DIR = M3_ROOT / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"
STATS_DIR = REPORTS_DIR / "stats"
ZENML_STATUS_JSON = REPORTS_DIR / "zenml_status.json"
ZENML_STATUS_TXT = REPORTS_DIR / "zenml_status.txt"

FEATURE_REPO_DIR = M3_ROOT / "feature_repo"
ZEN_CONFIG_DIR = M3_ROOT / ".zen"
ZEN_LOCAL_STORE_DIR = M3_ROOT.parent.parent / ".zen_local_m3"
