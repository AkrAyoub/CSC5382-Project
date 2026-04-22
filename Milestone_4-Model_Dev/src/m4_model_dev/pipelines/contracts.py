from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


class EvaluationInputs(TypedDict):
    reference_path: Path
    dataset_path: Path
    split_path: Path
    sft_paths: dict[str, Path]
    split_counts: dict[str, int]
    evaluation_instances: list[dict[str, Any]]


class CandidateRunResult(TypedDict):
    candidate_name: str
    candidate_kind: str
    metrics_df: pd.DataFrame
    raw_results_df: pd.DataFrame


class SingleCandidatePipelineResult(TypedDict):
    config: dict[str, Any]
    config_path: Path
    summary_path: Path
    metrics_path: Path
    raw_results_path: Path
    manifest_path: Path
    figure_paths: dict[str, Path]
    model_spec_path: Path
    sft_manifest_path: Path | None
    mlflow_logged: bool
    registered_model_name: str | None
    metrics_df: pd.DataFrame
    raw_results_df: pd.DataFrame


class ComparisonPipelineResult(TypedDict):
    results_df: pd.DataFrame
    raw_results_df: pd.DataFrame
    raw_results_path: Path
    metrics_path: Path
    summary_path: Path
    selection_path: Path
    figure_paths: dict[str, Path]
    mlflow_logged: bool
