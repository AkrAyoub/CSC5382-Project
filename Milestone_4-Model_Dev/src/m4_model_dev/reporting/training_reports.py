from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.models.model_registry import CandidateSpec
from m4_model_dev.utils.io import write_dataframe, write_json, write_text


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def _serialize_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.where(pd.notna(df), None).to_dict(orient="records")


def build_run_manifest(
    *,
    config_path: Path,
    dataset_path: Path,
    split_path: Path,
    reference_path: Path,
    candidate_spec_path: Path,
    sft_manifest_path: Path | None,
    split_counts: dict[str, int],
    metrics_df: pd.DataFrame,
    candidate_spec: CandidateSpec,
) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "reference_path": str(reference_path),
        "candidate_spec_path": str(candidate_spec_path),
        "sft_manifest_path": str(sft_manifest_path) if sft_manifest_path else "",
        "candidate_spec": asdict(candidate_spec),
        "split_counts": split_counts,
        "metrics": _serialize_dataframe(metrics_df),
        "mlflow_logged": False,
        "registered_model_name": "",
        "mlflow_artifact_log_errors": [],
    }


def write_training_reports(
    *,
    metrics_df: pd.DataFrame,
    raw_results_df: pd.DataFrame,
    run_manifest: dict[str, Any],
    metrics_path: Path,
    raw_results_path: Path,
    manifest_path: Path,
) -> None:
    write_dataframe(metrics_df, metrics_path)
    write_json({"metrics": _serialize_dataframe(metrics_df)}, metrics_path.with_suffix(".json"))
    write_dataframe(raw_results_df, raw_results_path)
    write_json({"results": _serialize_dataframe(raw_results_df)}, raw_results_path.with_suffix(".json"))
    write_json(run_manifest, manifest_path)


def _render_split_metrics(row: dict[str, Any]) -> str:
    return (
        f"{row['split']}: "
        f"success_rate={float(row['success_rate']):.4f}, "
        f"generation_success_rate={float(row['generation_success_rate']):.4f}, "
        f"execution_success_rate={float(row['execution_success_rate']):.4f}, "
        f"feasibility_rate={float(row['feasibility_rate']):.4f}, "
        f"exact_match_rate={float(row['exact_match_rate']):.4f}, "
        f"mean_gap_vs_baseline_pct={float(row['mean_gap_vs_baseline_pct']):.4f}, "
        f"mean_runtime_s={float(row['mean_total_runtime_s']):.4f}, "
        f"attempted={int(row['attempted_instances'])}, "
        f"ok={int(row['successful_instances'])}, "
        f"failed={int(row['failed_instances'])}, "
        f"skipped={int(row['skipped_instances'])}"
    )


def render_training_summary(
    *,
    config: dict[str, Any],
    candidate_spec: CandidateSpec,
    metrics_df: pd.DataFrame,
    split_counts: dict[str, int],
    model_spec_path: Path,
    emissions_path: Path | None,
    mlflow_logged: bool,
    registered_model_name: str | None,
) -> str:
    lines = [
        "Milestone 4 symbolic model evaluation summary",
        f"run_name: {config.get('run_name', 'm4-run')}",
        f"candidate_name: {candidate_spec.name}",
        f"candidate_kind: {candidate_spec.kind}",
        f"backend_name: {candidate_spec.backend or 'none'}",
        f"model_name: {candidate_spec.model_name or 'none'}",
        f"prompt_template: {candidate_spec.prompt_template or 'none'}",
        f"candidate_spec_path: {model_spec_path}",
        f"mlflow_logged: {mlflow_logged}",
        f"registered_model_name: {registered_model_name or 'none'}",
        f"codecarbon_emissions_logged: {bool(emissions_path and emissions_path.exists())}",
        "split_counts:",
    ]
    for split_name in ("train", "val", "test"):
        if split_name in split_counts:
            lines.append(f"  - {split_name}: {split_counts[split_name]}")

    lines.append("aggregated_metrics:")
    ordered_metrics_df = metrics_df.copy()
    ordered_metrics_df["split_rank"] = ordered_metrics_df["split"].map(SPLIT_ORDER).fillna(len(SPLIT_ORDER)).astype(int)
    ordered_metrics_df = ordered_metrics_df.sort_values("split_rank").drop(columns=["split_rank"])
    for row in _serialize_dataframe(ordered_metrics_df):
        lines.append(f"  - {_render_split_metrics(row)}")
    return "\n".join(lines) + "\n"


def write_summary(summary_text: str, summary_path: Path) -> None:
    write_text(summary_text, summary_path)
