from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m4_model_dev.reporting.comparison_reports import order_results_df


def _save_figure(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_training_figures(
    *,
    metrics_df: pd.DataFrame,
    raw_results_df: pd.DataFrame,
    candidate_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    ordered_metrics = metrics_df.copy()
    split_order = ["train", "val", "test"]
    ordered_metrics["split_rank"] = ordered_metrics["split"].map({name: idx for idx, name in enumerate(split_order)})
    ordered_metrics = ordered_metrics.sort_values("split_rank").drop(columns=["split_rank"])

    split_labels = ordered_metrics["split"].str.upper().tolist()
    x_positions = np.arange(len(split_labels))
    figure_paths: dict[str, Path] = {}

    metrics_path = output_dir / "training_metrics.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.24
    for index, metric_name in enumerate(["success_rate", "execution_success_rate", "exact_match_rate"]):
        ax.bar(
            x_positions + index * bar_width,
            ordered_metrics[metric_name].to_numpy(dtype=float),
            width=bar_width,
            label=metric_name.replace("_", " ").title(),
        )
    ax.set_xticks(x_positions + bar_width, split_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"{candidate_name} split success metrics")
    ax.legend(fontsize=8)
    figure_paths["training_metrics"] = _save_figure(fig, metrics_path)

    status_path = output_dir / "training_status.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    stacked_bottom = np.zeros(len(split_labels))
    status_columns = [
        ("successful_instances", "#2e8b57", "Successful"),
        ("failed_instances", "#c0392b", "Failed"),
        ("skipped_instances", "#95a5a6", "Skipped"),
    ]
    for column_name, color, label in status_columns:
        values = ordered_metrics[column_name].to_numpy(dtype=float)
        ax.bar(x_positions, values, bottom=stacked_bottom, color=color, label=label)
        stacked_bottom += values
    ax.set_xticks(x_positions, split_labels)
    ax.set_ylabel("Instances")
    ax.set_title(f"{candidate_name} evaluation status counts")
    ax.legend(fontsize=8)
    figure_paths["training_status"] = _save_figure(fig, status_path)

    dashboard_path = output_dir / "training_dashboard.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(
        split_labels,
        ordered_metrics["mean_gap_vs_baseline_pct"].to_numpy(dtype=float),
        marker="o",
        color="#1f77b4",
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Mean gap vs baseline (%)")

    axes[0, 1].plot(
        split_labels,
        ordered_metrics["mean_total_runtime_s"].to_numpy(dtype=float),
        marker="o",
        color="#ff7f0e",
    )
    axes[0, 1].set_title("Mean total runtime (s)")

    exact_values = ordered_metrics["exact_match_rate"].to_numpy(dtype=float)
    axes[1, 0].bar(split_labels, exact_values, color="#3c8dbc")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Exact match rate")

    status_counts = (
        raw_results_df.groupby("status")["instance_id"].count().reindex(["OK", "FAIL", "SKIPPED"]).fillna(0.0)
    )
    axes[1, 1].bar(status_counts.index.tolist(), status_counts.to_numpy(dtype=float), color=["#2e8b57", "#c0392b", "#95a5a6"])
    axes[1, 1].set_title("Overall status distribution")

    figure_paths["training_dashboard"] = _save_figure(fig, dashboard_path)
    return figure_paths


def write_comparison_figures(results_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    ordered = order_results_df(results_df)
    validation_df = ordered[ordered["split"] == "val"].copy()
    test_df = ordered[ordered["split"] == "test"].copy()

    validation_names = validation_df["candidate_name"].tolist()
    validation_positions = np.arange(len(validation_names))
    figure_paths: dict[str, Path] = {}

    validation_path = output_dir / "comparison_validation_metrics.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        validation_positions - 0.18,
        validation_df["success_rate"].to_numpy(dtype=float),
        height=0.35,
        label="Validation success rate",
    )
    ax.barh(
        validation_positions + 0.18,
        validation_df["exact_match_rate"].to_numpy(dtype=float),
        height=0.35,
        label="Validation exact match rate",
    )
    ax.set_yticks(validation_positions, validation_names)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Rate")
    ax.set_title("Validation candidate comparison")
    ax.legend()
    figure_paths["comparison_validation_metrics"] = _save_figure(fig, validation_path)

    test_path = output_dir / "comparison_test_metrics.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    test_positions = np.arange(len(test_df["candidate_name"].tolist()))
    ax.barh(
        test_positions - 0.18,
        test_df["success_rate"].to_numpy(dtype=float),
        height=0.35,
        label="Test success rate",
    )
    ax.barh(
        test_positions + 0.18,
        test_df["exact_match_rate"].to_numpy(dtype=float),
        height=0.35,
        label="Test exact match rate",
    )
    ax.set_yticks(test_positions, test_df["candidate_name"].tolist())
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Rate")
    ax.set_title("Held-out test candidate comparison")
    ax.legend()
    figure_paths["comparison_test_metrics"] = _save_figure(fig, test_path)

    dashboard_path = output_dir / "comparison_dashboard.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    axes[0].bar(
        validation_positions - 0.18,
        validation_df["mean_gap_vs_baseline_pct"].to_numpy(dtype=float),
        width=0.35,
        label="Validation gap vs baseline",
    )
    if not test_df.empty:
        axes[0].bar(
            validation_positions + 0.18,
            test_df.set_index("candidate_name").reindex(validation_names)["mean_gap_vs_baseline_pct"].fillna(0.0).to_numpy(dtype=float),
            width=0.35,
            label="Test gap vs baseline",
        )
    axes[0].set_xticks(validation_positions, validation_names, rotation=20, ha="right")
    axes[0].set_ylabel("Gap (%)")
    axes[0].set_title("Mean objective gap vs baseline")
    axes[0].legend()

    axes[1].bar(
        validation_positions - 0.18,
        validation_df["mean_total_runtime_s"].to_numpy(dtype=float),
        width=0.35,
        label="Validation runtime",
    )
    if not test_df.empty:
        axes[1].bar(
            validation_positions + 0.18,
            test_df.set_index("candidate_name").reindex(validation_names)["mean_total_runtime_s"].fillna(0.0).to_numpy(dtype=float),
            width=0.35,
            label="Test runtime",
        )
    axes[1].set_xticks(validation_positions, validation_names, rotation=20, ha="right")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Mean runtime by candidate")
    axes[1].legend()

    figure_paths["comparison_dashboard"] = _save_figure(fig, dashboard_path)
    return figure_paths
