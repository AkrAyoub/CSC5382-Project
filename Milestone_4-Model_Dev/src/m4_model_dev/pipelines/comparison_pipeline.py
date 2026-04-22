from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from m4_model_dev.models.model_registry import resolve_comparison_candidates
from m4_model_dev.paths import M4_CONFIGS_DIR, M4_REPORTS_DIR, ensure_runtime_dirs
from m4_model_dev.pipelines.contracts import ComparisonPipelineResult
from m4_model_dev.pipelines.training_pipeline import evaluate_candidate_bundle, prepare_training_inputs
from m4_model_dev.reporting.comparison_reports import write_comparison_reports
from m4_model_dev.reporting.figures import write_comparison_figures
from m4_model_dev.tracking.mlflow_utils import log_comparison_run
from m4_model_dev.utils.config import load_yaml_config


def run_model_comparison_pipeline(config_path: Path | None = None) -> ComparisonPipelineResult:
    ensure_runtime_dirs()
    config_path = config_path or (M4_CONFIGS_DIR / "compare_models.yaml")
    config = load_yaml_config(config_path)
    training_inputs = prepare_training_inputs(config)
    candidate_specs = resolve_comparison_candidates(config)

    metrics_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    candidate_spec_paths: dict[str, Path] = {}

    for candidate_spec in candidate_specs:
        candidate_result = evaluate_candidate_bundle(config=config, training_inputs=training_inputs, candidate=candidate_spec)
        metrics_frames.append(candidate_result["metrics_df"])
        raw_frames.append(candidate_result["raw_results_df"])
        candidate_spec_paths[candidate_spec.name] = candidate_result["model_spec_path"]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
        raw_results_df = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()

    metrics_path = M4_REPORTS_DIR / "model_comparison.csv"
    raw_results_path = M4_REPORTS_DIR / "model_comparison_raw_results.csv"
    summary_path = M4_REPORTS_DIR / "model_comparison_summary.txt"
    selection_path = M4_REPORTS_DIR / "model_selection.json"

    selection_payload = write_comparison_reports(
        results_df=metrics_df,
        raw_results_df=raw_results_df,
        config_path=config_path,
        candidate_spec_paths=candidate_spec_paths,
        metrics_path=metrics_path,
        raw_results_path=raw_results_path,
        summary_path=summary_path,
        selection_path=selection_path,
    )
    ordered_metrics_df = pd.read_csv(metrics_path)
    figure_paths = write_comparison_figures(ordered_metrics_df, M4_REPORTS_DIR)

    mlflow_logged = log_comparison_run(
        config=config,
        config_path=config_path,
        results_df=ordered_metrics_df,
        raw_results_df=raw_results_df,
        selection_payload=selection_payload,
        artifacts=[metrics_path, metrics_path.with_suffix(".json"), raw_results_path, raw_results_path.with_suffix(".json"), summary_path, selection_path, *figure_paths.values(), *candidate_spec_paths.values()],
    )

    return {
        "results_df": ordered_metrics_df,
        "raw_results_df": raw_results_df,
        "raw_results_path": raw_results_path,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "selection_path": selection_path,
        "figure_paths": figure_paths,
        "mlflow_logged": mlflow_logged,
    }
