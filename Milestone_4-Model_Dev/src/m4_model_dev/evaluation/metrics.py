from __future__ import annotations

from typing import Any

import pandas as pd


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean())


def safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.median())


def aggregate_candidate_split_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (candidate_name, split_name), group_df in results_df.groupby(["candidate_name", "split"], sort=True):
        attempted = len(group_df)
        success_df = group_df[group_df["status"] == "OK"].copy()
        skipped_df = group_df[group_df["status"] == "SKIPPED"].copy()
        representative = group_df.iloc[0].to_dict()

        rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_kind": representative.get("candidate_kind", ""),
                "backend_name": representative.get("backend_name", ""),
                "model_name": representative.get("model_name", ""),
                "prompt_template": representative.get("prompt_template", ""),
                "split": split_name,
                "attempted_instances": attempted,
                "successful_instances": int((group_df["status"] == "OK").sum()),
                "failed_instances": int((group_df["status"] == "FAIL").sum()),
                "skipped_instances": len(skipped_df),
                "success_rate": float((group_df["status"] == "OK").mean()) if attempted else 0.0,
                "generation_success_rate": safe_mean(group_df["generation_success"]),
                "execution_success_rate": safe_mean(group_df["execution_success"]),
                "feasibility_rate": safe_mean(group_df["feasible_solution"]),
                "exact_match_rate": safe_mean(group_df["exact_match_with_baseline"]),
                "mean_gap_vs_baseline_pct": safe_mean(success_df["gap_vs_baseline_pct"]) if not success_df.empty else 0.0,
                "median_gap_vs_baseline_pct": safe_median(success_df["gap_vs_baseline_pct"]) if not success_df.empty else 0.0,
                "mean_gap_vs_best_known_pct": safe_mean(success_df["gap_vs_best_known_pct"]) if not success_df.empty else 0.0,
                "mean_total_runtime_s": safe_mean(group_df["total_runtime_s"]),
                "mean_generation_runtime_s": safe_mean(group_df["generation_runtime_s"]),
                "mean_execution_runtime_s": safe_mean(group_df["execution_runtime_s"]),
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    metrics_df["split_rank"] = metrics_df["split"].map(SPLIT_ORDER).fillna(len(SPLIT_ORDER)).astype(int)
    return metrics_df.sort_values(["candidate_name", "split_rank"]).drop(columns=["split_rank"]).reset_index(drop=True)


def build_validation_selection_frame(metrics_df: pd.DataFrame, *, exclude_baseline: bool = True) -> pd.DataFrame:
    validation_df = metrics_df[metrics_df["split"] == "val"].copy()
    if validation_df.empty:
        fallback_df = metrics_df.copy()
        if exclude_baseline and "candidate_kind" in fallback_df.columns:
            non_baseline_df = fallback_df[fallback_df["candidate_kind"] != "deterministic_baseline"].copy()
            if not non_baseline_df.empty:
                fallback_df = non_baseline_df
        return fallback_df

    if exclude_baseline and "candidate_kind" in validation_df.columns:
        non_baseline_df = validation_df[validation_df["candidate_kind"] != "deterministic_baseline"].copy()
        if not non_baseline_df.empty:
            validation_df = non_baseline_df

    if {"skipped_instances", "attempted_instances"}.issubset(validation_df.columns):
        active_df = validation_df[validation_df["skipped_instances"] < validation_df["attempted_instances"]].copy()
        if not active_df.empty:
            validation_df = active_df

    ranked = validation_df.sort_values(
        ["success_rate", "exact_match_rate", "mean_gap_vs_baseline_pct", "candidate_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return ranked
