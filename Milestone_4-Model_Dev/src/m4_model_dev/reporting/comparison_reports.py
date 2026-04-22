from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.evaluation.metrics import build_validation_selection_frame
from m4_model_dev.utils.io import write_dataframe, write_json, write_text


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def _serialize_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _selection_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    return build_validation_selection_frame(results_df, exclude_baseline=True)


def order_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in results_df.columns:
        return _selection_frame(results_df)

    selection_df = _selection_frame(results_df)
    ranked_candidates = selection_df["candidate_name"].tolist()
    rank_map = {candidate_name: index for index, candidate_name in enumerate(ranked_candidates)}

    ordered = results_df.copy()
    ordered["candidate_rank"] = ordered["candidate_name"].map(rank_map).fillna(len(rank_map)).astype(int)
    ordered["split_rank"] = ordered["split"].map(SPLIT_ORDER).fillna(len(SPLIT_ORDER)).astype(int)
    ordered = ordered.sort_values(["candidate_rank", "candidate_name", "split_rank"]).drop(
        columns=["candidate_rank", "split_rank"]
    )
    return ordered.reset_index(drop=True)


def render_comparison_summary(results_df: pd.DataFrame) -> str:
    selection_df = _selection_frame(results_df)
    lines = [
        "Milestone 4 symbolic candidate comparison summary",
        "",
        "Ranking by validation success rate, then exact match rate, then validation mean gap vs baseline:",
    ]
    for _, row in selection_df.iterrows():
        lines.append(
            f"- {row['candidate_name']}: "
            f"success_rate={float(row['success_rate']):.4f}, "
            f"exact_match_rate={float(row['exact_match_rate']):.4f}, "
            f"mean_gap_vs_baseline_pct={float(row['mean_gap_vs_baseline_pct']):.4f}, "
            f"mean_runtime_s={float(row['mean_total_runtime_s']):.4f}"
        )

    if not selection_df.empty:
        best_row = selection_df.iloc[0]
        lines.extend(
            [
                "",
                f"Selected non-baseline candidate: {best_row['candidate_name']}",
                f"Validation success rate: {float(best_row['success_rate']):.4f}",
                f"Validation exact match rate: {float(best_row['exact_match_rate']):.4f}",
                f"Validation mean gap vs baseline: {float(best_row['mean_gap_vs_baseline_pct']):.4f}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_selection_payload(best_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_rule": (
            "highest validation success rate, tie-broken by validation exact match rate, "
            "then by lower validation mean gap vs baseline, then candidate_name"
        ),
        "selected_candidate_name": best_row["candidate_name"],
        "selected_candidate_kind": best_row.get("candidate_kind", ""),
        "selected_backend_name": best_row.get("backend_name", ""),
        "selected_model_name": best_row.get("model_name", ""),
        "selected_prompt_template": best_row.get("prompt_template", ""),
        "selected_validation_success_rate": best_row["success_rate"],
        "selected_validation_exact_match_rate": best_row["exact_match_rate"],
        "selected_validation_mean_gap_vs_baseline_pct": best_row["mean_gap_vs_baseline_pct"],
    }


def write_comparison_reports(
    *,
    results_df: pd.DataFrame,
    raw_results_df: pd.DataFrame,
    config_path: Path,
    candidate_spec_paths: dict[str, Path],
    metrics_path: Path,
    raw_results_path: Path,
    summary_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    ordered = order_results_df(results_df)
    selection_df = _selection_frame(ordered)
    best_row = selection_df.iloc[0].to_dict() if not selection_df.empty else ordered.iloc[0].to_dict()

    write_dataframe(ordered, metrics_path)
    write_json(
        {
            "config_path": str(config_path),
            "candidate_spec_paths": {name: str(path) for name, path in candidate_spec_paths.items()},
            "results": _serialize_dataframe(ordered),
        },
        metrics_path.with_suffix(".json"),
    )
    write_dataframe(raw_results_df, raw_results_path)
    write_json({"results": _serialize_dataframe(raw_results_df)}, raw_results_path.with_suffix(".json"))
    write_text(render_comparison_summary(ordered), summary_path)

    selection_payload = build_selection_payload(best_row)
    write_json(selection_payload, selection_path)
    return selection_payload
