from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.data.build_labels import build_facility_open_labels
from m4_model_dev.data.make_splits import build_grouped_splits
from m4_model_dev.data.prepare_dataset import build_training_dataset
from m4_model_dev.evaluation.metrics import binary_classification_report
from m4_model_dev.models.logistic_numpy import StandardScaler
from m4_model_dev.models.model_registry import (
    SUPPORTED_MODEL_NAMES,
    extract_feature_importance,
    fit_model_bundle,
    predict_scores,
    save_model_bundle,
)
from m4_model_dev.paths import (
    M4_ARTIFACTS_DIR,
    M4_CONFIGS_DIR,
    M4_LABELS_DIR,
    M4_MERGED_DIR,
    M4_REPORTS_DIR,
    ensure_runtime_dirs,
)
from m4_model_dev.tracking.mlflow_utils import log_training_run
from m4_model_dev.utils.config import load_yaml_config
from m4_model_dev.utils.io import write_dataframe, write_json, write_text


IDENTIFIER_COLUMNS = {
    "facility_key",
    "instance_id",
    "event_timestamp",
    "split",
}

LABEL_COLUMNS = {
    "is_open",
    "open_facility_count",
    "objective",
    "best_known",
    "gap_percent",
    "runtime_s",
}


def _maybe_start_codecarbon(enable: bool):
    if not enable:
        return None
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        return None
    return EmissionsTracker(
        project_name="m4-facility-opening",
        output_dir=str(M4_REPORTS_DIR),
        output_file="emissions.csv",
        save_to_file=True,
    )


def select_threshold(
    y_true: list[int],
    y_score: list[float],
    threshold_grid: list[float],
) -> tuple[float, dict[str, float | int]]:
    best_threshold = threshold_grid[0]
    best_report = binary_classification_report(y_true, y_score, best_threshold)

    for threshold in threshold_grid[1:]:
        report = binary_classification_report(y_true, y_score, threshold)
        if report["f1"] > best_report["f1"]:
            best_threshold = threshold
            best_report = report

    return best_threshold, best_report


def prepare_feature_columns(df: pd.DataFrame) -> list[str]:
    candidate_columns: list[str] = []
    for column in df.columns:
        if column in IDENTIFIER_COLUMNS or column in LABEL_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            candidate_columns.append(column)
    return candidate_columns


def build_split_dataset(dataset_path: Path, split_path: Path) -> pd.DataFrame:
    dataset_df = pd.read_csv(dataset_path)
    split_df = pd.read_csv(split_path)
    return dataset_df.merge(split_df[["instance_id", "split"]], on="instance_id", how="left")


def _build_runtime_dataset(config: dict[str, Any]) -> tuple[Path, Path, Path, pd.DataFrame, list[str]]:
    labels_path = (
        build_facility_open_labels()
        if config.get("regenerate_labels", True)
        else (M4_LABELS_DIR / "facility_open_labels.csv")
    )
    if not labels_path.exists():
        labels_path = build_facility_open_labels()

    dataset_path = build_training_dataset(labels_path=labels_path)
    split_path = build_grouped_splits(dataset_path=dataset_path)

    merged_df = build_split_dataset(dataset_path, split_path)
    merged_output = M4_MERGED_DIR / "facility_training_dataset_with_splits.csv"
    merged_df.to_csv(merged_output, index=False)
    feature_columns = prepare_feature_columns(merged_df)
    return labels_path, dataset_path, split_path, merged_df, feature_columns


def load_training_config(config_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    ensure_runtime_dirs()
    resolved_path = config_path or (M4_CONFIGS_DIR / "train_best_model.yaml")
    config = load_yaml_config(resolved_path)
    model_family = str(config.get("model_family", "logreg_numpy"))
    if model_family not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported model_family '{model_family}'. Supported: {sorted(SUPPORTED_MODEL_NAMES)}")
    return resolved_path, config


def prepare_training_inputs(config: dict[str, Any]) -> dict[str, Any]:
    labels_path, dataset_path, split_path, merged_df, feature_columns = _build_runtime_dataset(config)

    train_df = merged_df[merged_df["split"] == "train"].copy()
    val_df = merged_df[merged_df["split"] == "val"].copy()
    test_df = merged_df[merged_df["split"] == "test"].copy()

    x_train_raw = train_df[feature_columns].to_numpy(dtype=float)
    x_val_raw = val_df[feature_columns].to_numpy(dtype=float)
    x_test_raw = test_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["is_open"].to_numpy(dtype=int)
    y_val = val_df["is_open"].to_numpy(dtype=int)
    y_test = test_df["is_open"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(x_train_raw)

    return {
        "labels_path": labels_path,
        "dataset_path": dataset_path,
        "split_path": split_path,
        "merged_dataset_path": M4_MERGED_DIR / "facility_training_dataset_with_splits.csv",
        "feature_columns": feature_columns,
        "split_counts": merged_df["split"].value_counts().to_dict(),
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "x_train_scaled": scaler.transform(x_train_raw),
        "x_val_scaled": scaler.transform(x_val_raw),
        "x_test_scaled": scaler.transform(x_test_raw),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
    }


def train_model_bundle(config: dict[str, Any], training_inputs: dict[str, Any]) -> dict[str, Any]:
    model_family = str(config.get("model_family", "logreg_numpy"))
    random_seed = int(config.get("random_seed", 42))
    tracker = _maybe_start_codecarbon(config.get("tracking", {}).get("enable_codecarbon", False))
    if tracker is not None:
        tracker.start()

    try:
        bundle = fit_model_bundle(
            model_name=model_family,
            config=config,
            x_train_raw=training_inputs["x_train_raw"],
            x_train_scaled=training_inputs["x_train_scaled"],
            y_train=training_inputs["y_train"],
            scaler=training_inputs["scaler"],
            random_seed=random_seed,
        )
    finally:
        if tracker is not None:
            tracker.stop()

    emissions_path = M4_REPORTS_DIR / "emissions.csv"
    return {
        "bundle": bundle,
        "model_family": model_family,
        "emissions_path": emissions_path if emissions_path.exists() else None,
    }


def evaluate_model_bundle(
    config: dict[str, Any],
    training_inputs: dict[str, Any],
    trained_model: dict[str, Any],
) -> dict[str, Any]:
    bundle = trained_model["bundle"]
    train_scores = predict_scores(bundle, training_inputs["x_train_raw"], training_inputs["x_train_scaled"])
    val_scores = predict_scores(bundle, training_inputs["x_val_raw"], training_inputs["x_val_scaled"])
    test_scores = predict_scores(bundle, training_inputs["x_test_raw"], training_inputs["x_test_scaled"])

    threshold_grid = [float(value) for value in config["training"]["threshold_grid"]]
    best_threshold, val_report = select_threshold(training_inputs["y_val"].tolist(), val_scores, threshold_grid)
    metrics = {
        "train": binary_classification_report(training_inputs["y_train"].tolist(), train_scores, best_threshold),
        "val": val_report,
        "test": binary_classification_report(training_inputs["y_test"].tolist(), test_scores, best_threshold),
    }
    confusion_report = {
        split_name: {
            key: split_metrics[key]
            for key in ("tp", "fp", "tn", "fn", "threshold", "support")
        }
        for split_name, split_metrics in metrics.items()
    }
    feature_importance_rows = extract_feature_importance(bundle, training_inputs["feature_columns"])

    return {
        "threshold": best_threshold,
        "metrics": metrics,
        "confusion_report": confusion_report,
        "feature_importance_rows": feature_importance_rows,
    }


def persist_training_outputs(
    config: dict[str, Any],
    config_path: Path,
    training_inputs: dict[str, Any],
    trained_model: dict[str, Any],
    evaluation_results: dict[str, Any],
) -> dict[str, Any]:
    bundle = trained_model["bundle"]
    emissions_path = trained_model.get("emissions_path")
    threshold = float(evaluation_results["threshold"])
    metrics = evaluation_results["metrics"]
    feature_columns = training_inputs["feature_columns"]

    model_artifact_path = save_model_bundle(bundle, feature_columns, threshold, M4_ARTIFACTS_DIR)
    metrics_path = M4_REPORTS_DIR / "metrics.json"
    summary_path = M4_REPORTS_DIR / "summary.txt"
    manifest_path = M4_REPORTS_DIR / "run_manifest.json"
    confusion_path = M4_REPORTS_DIR / "confusion_report.json"
    feature_importance_path = M4_REPORTS_DIR / "feature_importance.csv"

    write_json(metrics, metrics_path)
    write_json(evaluation_results["confusion_report"], confusion_path)
    if evaluation_results["feature_importance_rows"]:
        write_dataframe(pd.DataFrame(evaluation_results["feature_importance_rows"]), feature_importance_path)

    run_manifest = {
        "config_path": str(config_path),
        "labels_path": str(training_inputs["labels_path"]),
        "dataset_path": str(training_inputs["dataset_path"]),
        "merged_dataset_path": str(training_inputs["merged_dataset_path"]),
        "split_path": str(training_inputs["split_path"]),
        "model_family": trained_model["model_family"],
        "model_artifact_path": str(model_artifact_path),
        "feature_columns": feature_columns,
        "selected_threshold": threshold,
        "metrics": metrics,
        "metadata": bundle.get("metadata", {}),
        "uses_scaled_features": bundle.get("uses_scaled_features", False),
        "mlflow_artifact_log_errors": [],
    }
    write_json(run_manifest, manifest_path)

    pre_mlflow_summary = render_summary(
        config=config,
        feature_columns=feature_columns,
        threshold=threshold,
        metrics=metrics,
        split_counts=training_inputs["split_counts"],
        model_artifact_path=model_artifact_path,
        emissions_path=emissions_path,
        mlflow_logged=False,
        registered_model_name=None,
        artifact_log_errors=[],
    )
    write_text(pre_mlflow_summary, summary_path)

    artifacts_for_mlflow = [metrics_path, summary_path]
    mlflow_result = log_training_run(
        config=config,
        metrics=metrics,
        artifacts=artifacts_for_mlflow,
        bundle=bundle,
        feature_columns=feature_columns,
        threshold=threshold,
        config_path=config_path,
        register_model=bool(config.get("tracking", {}).get("register_model", False)),
        registered_model_name=str(config.get("tracking", {}).get("registered_model_name", "")) or None,
    )
    run_manifest["mlflow_artifact_log_errors"] = mlflow_result.get("artifact_log_errors", [])
    write_json(run_manifest, manifest_path)

    summary_text = render_summary(
        config=config,
        feature_columns=feature_columns,
        threshold=threshold,
        metrics=metrics,
        split_counts=training_inputs["split_counts"],
        model_artifact_path=model_artifact_path,
        emissions_path=emissions_path,
        mlflow_logged=bool(mlflow_result["mlflow_logged"]),
        registered_model_name=mlflow_result.get("registered_model_name"),
        artifact_log_errors=mlflow_result.get("artifact_log_errors", []),
    )
    write_text(summary_text, summary_path)

    return {
        "config": config,
        "config_path": config_path,
        "dataset_path": training_inputs["dataset_path"],
        "split_path": training_inputs["split_path"],
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "confusion_path": confusion_path,
        "feature_importance_path": feature_importance_path if evaluation_results["feature_importance_rows"] else None,
        "emissions_path": emissions_path,
        "model_artifact_path": model_artifact_path,
        "mlflow_logged": bool(mlflow_result["mlflow_logged"]),
        "registered_model_name": mlflow_result.get("registered_model_name"),
        "metrics": metrics,
        "model_family": trained_model["model_family"],
        "model_bundle": bundle,
        "feature_columns": feature_columns,
        "threshold": threshold,
    }


def _render_split_metrics(split_name: str, split_metrics: dict[str, float | int]) -> str:
    return (
        f"{split_name}: "
        f"accuracy={split_metrics['accuracy']:.4f}, "
        f"precision={split_metrics['precision']:.4f}, "
        f"recall={split_metrics['recall']:.4f}, "
        f"f1={split_metrics['f1']:.4f}, "
        f"roc_auc={split_metrics['roc_auc']:.4f}, "
        f"log_loss={split_metrics['log_loss']:.4f}, "
        f"tp={split_metrics['tp']}, fp={split_metrics['fp']}, "
        f"tn={split_metrics['tn']}, fn={split_metrics['fn']}"
    )


def render_summary(
    config: dict[str, Any],
    feature_columns: list[str],
    threshold: float,
    metrics: dict[str, dict[str, float | int]],
    split_counts: dict[str, int],
    model_artifact_path: Path,
    emissions_path: Path | None,
    mlflow_logged: bool,
    registered_model_name: str | None,
    artifact_log_errors: list[str],
) -> str:
    model_family = str(config.get("model_family", "logreg_numpy"))
    lines = [
        "Milestone 4 training summary",
        f"run_name: {config.get('run_name', 'm4-run')}",
        f"model_family: {model_family}",
        f"feature_count: {len(feature_columns)}",
        f"selected_threshold: {threshold:.2f}",
        f"model_artifact_path: {model_artifact_path}",
        f"mlflow_logged: {mlflow_logged}",
        f"registered_model_name: {registered_model_name or 'none'}",
        f"codecarbon_emissions_logged: {bool(emissions_path and emissions_path.exists())}",
        f"mlflow_artifact_copy_warnings: {len(artifact_log_errors)}",
        "split_counts:",
    ]
    for split_name, count in split_counts.items():
        lines.append(f"  - {split_name}: {count}")
    lines.append("metrics:")
    for split_name, split_metrics in metrics.items():
        lines.append(f"  - {_render_split_metrics(split_name, split_metrics)}")
    if artifact_log_errors:
        lines.append("mlflow_artifact_copy_errors:")
        for error in artifact_log_errors:
            lines.append(f"  - {error}")
    return "\n".join(lines) + "\n"


def run_training_pipeline(config_path: Path | None = None) -> dict[str, Any]:
    config_path, config = load_training_config(config_path)
    training_inputs = prepare_training_inputs(config)
    trained_model = train_model_bundle(config, training_inputs)
    evaluation_results = evaluate_model_bundle(config, training_inputs, trained_model)
    return persist_training_outputs(
        config=config,
        config_path=config_path,
        training_inputs=training_inputs,
        trained_model=trained_model,
        evaluation_results=evaluation_results,
    )
