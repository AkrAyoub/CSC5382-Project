from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.data.build_reference_solutions import build_reference_solutions
from m4_model_dev.data.build_sft_dataset import build_sft_dataset
from m4_model_dev.data.build_benchmark_dataset import build_benchmark_dataset
from m4_model_dev.data.make_splits import build_grouped_splits
from m4_model_dev.evaluation.generated_exec import run_generated_solver
from m4_model_dev.evaluation.metrics import aggregate_candidate_split_metrics
from m4_model_dev.models.model_registry import (
    CandidateSpec,
    resolve_single_candidate_config,
    serialize_candidate_spec,
)
from m4_model_dev.models.symbolic_generator import generate_solver_code
from m4_model_dev.paths import (
    M4_ARTIFACTS_DIR,
    M4_CONFIGS_DIR,
    M4_DATASETS_DIR,
    M4_EVAL_RESULTS_DIR,
    M4_GENERATED_CODE_DIR,
    M4_REFERENCE_DIR,
    M4_REPORTS_DIR,
    M4_SFT_DIR,
    ensure_runtime_dirs,
)
from m4_model_dev.pipelines.contracts import EvaluationInputs, SingleCandidatePipelineResult
from m4_model_dev.reporting.figures import write_training_figures
from m4_model_dev.reporting.training_reports import (
    build_run_manifest,
    render_training_summary,
    write_summary,
    write_training_reports,
)
from m4_model_dev.tracking.codecarbon_utils import maybe_start_codecarbon
from m4_model_dev.tracking.mlflow_utils import log_training_run
from m4_model_dev.utils.config import load_yaml_config


DEFAULT_EVAL_SPLITS = ["train", "val", "test"]


def load_training_config(config_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    ensure_runtime_dirs()
    resolved_path = config_path or (M4_CONFIGS_DIR / "train_best_model.yaml")
    return resolved_path, load_yaml_config(resolved_path)


def _read_reference_df(reference_path: Path) -> pd.DataFrame:
    df = pd.read_csv(reference_path)
    return df.sort_values("instance_id").reset_index(drop=True)


def _read_split_df(split_path: Path) -> pd.DataFrame:
    df = pd.read_csv(split_path)
    return df.sort_values(["split", "instance_id"]).reset_index(drop=True)


def _build_runtime_inputs(config: dict[str, Any]) -> EvaluationInputs:
    reference_path = build_reference_solutions(M4_REFERENCE_DIR / "reference_solutions.csv")
    dataset_path = build_benchmark_dataset(reference_path=reference_path, output_path=M4_DATASETS_DIR / "benchmark_instances.csv")
    split_path = build_grouped_splits(dataset_path=dataset_path)
    sft_paths = build_sft_dataset(split_path=split_path, output_dir=M4_SFT_DIR)

    dataset_df = pd.read_csv(dataset_path)
    split_df = _read_split_df(split_path)
    merged_df = dataset_df.merge(split_df[["instance_id", "split"]], on="instance_id", how="left")
    split_counts = merged_df["split"].value_counts().to_dict()

    selected_splits = [str(value) for value in config.get("evaluation", {}).get("splits", DEFAULT_EVAL_SPLITS)]
    evaluation_instances = (
        merged_df[merged_df["split"].isin(selected_splits)]
        .sort_values(["split", "facility_count_m", "instance_id"])
        .to_dict(orient="records")
    )

    return {
        "reference_path": reference_path,
        "dataset_path": dataset_path,
        "split_path": split_path,
        "sft_paths": sft_paths,
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "evaluation_instances": evaluation_instances,
    }


def prepare_training_inputs(config: dict[str, Any]) -> EvaluationInputs:
    return _build_runtime_inputs(config)


def _reference_index(reference_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["instance_id"]): row.to_dict() for _, row in reference_df.iterrows()}


def _code_output_path(candidate_name: str) -> Path:
    return M4_GENERATED_CODE_DIR / candidate_name / f"{candidate_name}.py"


def _candidate_result_row(
    *,
    candidate: CandidateSpec,
    split_name: str,
    instance_id: str,
    instance_path: str,
    facility_count_m: int,
    customer_count_n: int,
    baseline_row: dict[str, Any],
    status: str,
    error: str = "",
    generation_success: int = 0,
    execution_success: int = 0,
    feasible_solution: int = 0,
    exact_match_with_baseline: int = 0,
    candidate_objective: float | None = None,
    generation_runtime_s: float = 0.0,
    execution_runtime_s: float = 0.0,
    total_runtime_s: float = 0.0,
    generated_code_path: str = "",
) -> dict[str, Any]:
    baseline_objective = float(baseline_row["objective"])
    best_known = baseline_row.get("best_known")
    gap_vs_baseline_pct = None
    gap_vs_best_known_pct = None
    if candidate_objective is not None:
        gap_vs_baseline_pct = (float(candidate_objective) - baseline_objective) / max(1.0, abs(baseline_objective)) * 100.0
        if pd.notna(best_known):
            gap_vs_best_known_pct = (float(candidate_objective) - float(best_known)) / max(1.0, abs(float(best_known))) * 100.0

    return {
        "candidate_name": candidate.name,
        "candidate_kind": candidate.kind,
        "backend_name": candidate.backend or "",
        "model_name": candidate.model_name or "",
        "prompt_template": candidate.prompt_template or "",
        "split": split_name,
        "instance_id": instance_id,
        "instance_path": instance_path,
        "facility_count_m": facility_count_m,
        "customer_count_n": customer_count_n,
        "status": status,
        "error": error,
        "generation_success": generation_success,
        "execution_success": execution_success,
        "feasible_solution": feasible_solution,
        "exact_match_with_baseline": exact_match_with_baseline,
        "baseline_objective": baseline_objective,
        "baseline_best_known": best_known if pd.notna(best_known) else None,
        "baseline_gap_percent": baseline_row.get("gap_percent"),
        "baseline_runtime_s": float(baseline_row["runtime_s"]),
        "candidate_objective": candidate_objective,
        "gap_vs_baseline_pct": gap_vs_baseline_pct,
        "gap_vs_best_known_pct": gap_vs_best_known_pct,
        "generation_runtime_s": generation_runtime_s,
        "execution_runtime_s": execution_runtime_s,
        "total_runtime_s": total_runtime_s,
        "generated_code_path": generated_code_path,
    }


def evaluate_candidate_bundle(
    *,
    config: dict[str, Any],
    training_inputs: EvaluationInputs,
    candidate: CandidateSpec,
) -> dict[str, Any]:
    reference_df = _read_reference_df(training_inputs["reference_path"])
    baseline_by_instance = _reference_index(reference_df)
    tolerance_pct = float(config.get("evaluation", {}).get("exact_match_tolerance_pct", 1e-6))
    evaluation_instances = list(training_inputs["evaluation_instances"])
    shared_generation_runtime_s = 0.0
    generated_code_path = ""
    generated_code = ""
    generation_error: Exception | None = None
    per_instance_generation_runtime_s = 0.0

    tracker = maybe_start_codecarbon(
        config.get("tracking", {}).get("enable_codecarbon", False),
        output_dir=M4_REPORTS_DIR,
        output_file="emissions.csv",
    )
    if tracker is not None:
        tracker.start()

    results: list[dict[str, Any]] = []
    try:
        if candidate.kind == "llm" and candidate.enabled:
            llm_t0 = time.time()
            try:
                generated = generate_solver_code(candidate)
                shared_generation_runtime_s = time.time() - llm_t0
                code_path = _code_output_path(candidate.name)
                code_path.parent.mkdir(parents=True, exist_ok=True)
                code_path.write_text(generated.code, encoding="utf-8")
                generated_code_path = str(code_path)
                generated_code = generated.code
            except Exception as exc:
                shared_generation_runtime_s = time.time() - llm_t0
                generation_error = exc

            if evaluation_instances:
                per_instance_generation_runtime_s = shared_generation_runtime_s / float(len(evaluation_instances))

        for row in evaluation_instances:
            instance_id = str(row["instance_id"])
            baseline_row = baseline_by_instance[instance_id]
            instance_path = str(row["instance_path"])
            split_name = str(row["split"])

            if candidate.kind == "deterministic_baseline":
                results.append(
                    _candidate_result_row(
                        candidate=candidate,
                        split_name=split_name,
                        instance_id=instance_id,
                        instance_path=instance_path,
                        facility_count_m=int(row["facility_count_m"]),
                        customer_count_n=int(row["customer_count_n"]),
                        baseline_row=baseline_row,
                        status="OK",
                        generation_success=1,
                        execution_success=1,
                        feasible_solution=1,
                        exact_match_with_baseline=1,
                        candidate_objective=float(baseline_row["objective"]),
                        total_runtime_s=float(baseline_row["runtime_s"]),
                    )
                )
                continue

            if not candidate.enabled:
                results.append(
                    _candidate_result_row(
                        candidate=candidate,
                        split_name=split_name,
                        instance_id=instance_id,
                        instance_path=instance_path,
                        facility_count_m=int(row["facility_count_m"]),
                        customer_count_n=int(row["customer_count_n"]),
                        baseline_row=baseline_row,
                        status="SKIPPED",
                        error="Candidate is disabled in configuration.",
                        generation_runtime_s=0.0,
                    )
                )
                continue

            if generation_error is not None:
                results.append(
                    _candidate_result_row(
                        candidate=candidate,
                        split_name=split_name,
                        instance_id=instance_id,
                        instance_path=instance_path,
                        facility_count_m=int(row["facility_count_m"]),
                        customer_count_n=int(row["customer_count_n"]),
                        baseline_row=baseline_row,
                        status="SKIPPED" if "Missing GROQ_API_KEY" in str(generation_error) else "FAIL",
                        error=f"{type(generation_error).__name__}: {generation_error}",
                        generation_runtime_s=per_instance_generation_runtime_s,
                        total_runtime_s=per_instance_generation_runtime_s,
                    )
                )
                continue

            exec_t0 = time.time()
            try:
                executed = run_generated_solver(generated_code, instance_path)
                execution_runtime_s = time.time() - exec_t0
                gap_vs_baseline_pct = (
                    (executed.objective - float(baseline_row["objective"]))
                    / max(1.0, abs(float(baseline_row["objective"])))
                    * 100.0
                )

                results.append(
                    _candidate_result_row(
                        candidate=candidate,
                        split_name=split_name,
                        instance_id=instance_id,
                        instance_path=instance_path,
                        facility_count_m=int(row["facility_count_m"]),
                        customer_count_n=int(row["customer_count_n"]),
                        baseline_row=baseline_row,
                        status="OK",
                        generation_success=1,
                        execution_success=1,
                        feasible_solution=1,
                        exact_match_with_baseline=1 if abs(gap_vs_baseline_pct) <= tolerance_pct else 0,
                        candidate_objective=executed.objective,
                        generation_runtime_s=per_instance_generation_runtime_s,
                        execution_runtime_s=execution_runtime_s,
                        total_runtime_s=per_instance_generation_runtime_s + execution_runtime_s,
                        generated_code_path=generated_code_path,
                    )
                )
            except Exception as exc:
                execution_runtime_s = time.time() - exec_t0
                results.append(
                    _candidate_result_row(
                        candidate=candidate,
                        split_name=split_name,
                        instance_id=instance_id,
                        instance_path=instance_path,
                        facility_count_m=int(row["facility_count_m"]),
                        customer_count_n=int(row["customer_count_n"]),
                        baseline_row=baseline_row,
                        status="FAIL",
                        error=f"{type(exc).__name__}: {exc}",
                        generation_runtime_s=per_instance_generation_runtime_s,
                        execution_runtime_s=execution_runtime_s,
                        total_runtime_s=per_instance_generation_runtime_s + execution_runtime_s,
                        generated_code_path=generated_code_path,
                    )
                )
    finally:
        if tracker is not None:
            tracker.stop()

    raw_results_df = pd.DataFrame(results)
    metrics_df = aggregate_candidate_split_metrics(raw_results_df)
    model_spec_path = serialize_candidate_spec(candidate, M4_ARTIFACTS_DIR / f"{candidate.name}_spec.json")
    emissions_path = M4_REPORTS_DIR / "emissions.csv"

    return {
        "candidate": candidate,
        "raw_results_df": raw_results_df,
        "metrics_df": metrics_df,
        "model_spec_path": model_spec_path,
        "emissions_path": emissions_path if emissions_path.exists() else None,
    }


def train_model_bundle(config: dict[str, Any], training_inputs: EvaluationInputs) -> dict[str, Any]:
    candidate = resolve_single_candidate_config(config)
    return evaluate_candidate_bundle(config=config, training_inputs=training_inputs, candidate=candidate)


def evaluate_model_bundle(
    config: dict[str, Any],
    training_inputs: EvaluationInputs,
    trained_model: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metrics_df": trained_model["metrics_df"],
        "raw_results_df": trained_model["raw_results_df"],
    }


def persist_training_outputs(
    config: dict[str, Any],
    config_path: Path,
    training_inputs: EvaluationInputs,
    trained_model: dict[str, Any],
    evaluation_results: dict[str, Any],
) -> SingleCandidatePipelineResult:
    candidate: CandidateSpec = trained_model["candidate"]
    metrics_df: pd.DataFrame = evaluation_results["metrics_df"]
    raw_results_df: pd.DataFrame = evaluation_results["raw_results_df"]

    metrics_path = M4_EVAL_RESULTS_DIR / "single_candidate_metrics.csv"
    raw_results_path = M4_EVAL_RESULTS_DIR / "single_candidate_raw_results.csv"
    summary_path = M4_REPORTS_DIR / "summary.txt"
    manifest_path = M4_REPORTS_DIR / "run_manifest.json"
    sft_manifest_path = training_inputs["sft_paths"].get("manifest")

    figure_paths = write_training_figures(
        metrics_df=metrics_df,
        raw_results_df=raw_results_df,
        candidate_name=candidate.name,
        output_dir=M4_REPORTS_DIR,
    )

    run_manifest = build_run_manifest(
        config_path=config_path,
        dataset_path=training_inputs["dataset_path"],
        split_path=training_inputs["split_path"],
        reference_path=training_inputs["reference_path"],
        candidate_spec_path=trained_model["model_spec_path"],
        sft_manifest_path=sft_manifest_path,
        split_counts=training_inputs["split_counts"],
        metrics_df=metrics_df,
        candidate_spec=candidate,
    )
    run_manifest["figure_paths"] = {name: str(path) for name, path in figure_paths.items()}

    write_training_reports(
        metrics_df=metrics_df,
        raw_results_df=raw_results_df,
        run_manifest=run_manifest,
        metrics_path=metrics_path,
        raw_results_path=raw_results_path,
        manifest_path=manifest_path,
    )

    mlflow_result = log_training_run(
        config=config,
        candidate=candidate,
        metrics_df=metrics_df,
        raw_results_df=raw_results_df,
        artifacts=[
            metrics_path,
            metrics_path.with_suffix(".json"),
            raw_results_path,
            raw_results_path.with_suffix(".json"),
            manifest_path,
            *figure_paths.values(),
            trained_model["model_spec_path"],
        ],
        config_path=config_path,
        candidate_spec_path=trained_model["model_spec_path"],
        register_model=bool(config.get("tracking", {}).get("register_model", False)),
        registered_model_name=str(config.get("tracking", {}).get("registered_model_name", "")) or None,
    )

    run_manifest["mlflow_logged"] = bool(mlflow_result["mlflow_logged"])
    run_manifest["registered_model_name"] = mlflow_result.get("registered_model_name") or ""
    run_manifest["mlflow_artifact_log_errors"] = list(mlflow_result.get("artifact_log_errors", []))
    write_training_reports(
        metrics_df=metrics_df,
        raw_results_df=raw_results_df,
        run_manifest=run_manifest,
        metrics_path=metrics_path,
        raw_results_path=raw_results_path,
        manifest_path=manifest_path,
    )

    summary_text = render_training_summary(
        config=config,
        candidate_spec=candidate,
        metrics_df=metrics_df,
        split_counts=training_inputs["split_counts"],
        model_spec_path=trained_model["model_spec_path"],
        emissions_path=trained_model.get("emissions_path"),
        mlflow_logged=bool(mlflow_result["mlflow_logged"]),
        registered_model_name=mlflow_result.get("registered_model_name"),
    )
    write_summary(summary_text, summary_path)

    return {
        "config": config,
        "config_path": config_path,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "raw_results_path": raw_results_path,
        "manifest_path": manifest_path,
        "figure_paths": figure_paths,
        "model_spec_path": trained_model["model_spec_path"],
        "sft_manifest_path": sft_manifest_path,
        "mlflow_logged": bool(mlflow_result["mlflow_logged"]),
        "registered_model_name": mlflow_result.get("registered_model_name"),
        "metrics_df": metrics_df,
        "raw_results_df": raw_results_df,
    }


def run_training_pipeline(config_path: Path | None = None) -> SingleCandidatePipelineResult:
    resolved_config_path, config = load_training_config(config_path)
    training_inputs = prepare_training_inputs(config)
    trained_model = train_model_bundle(config, training_inputs)
    evaluation_results = evaluate_model_bundle(config, training_inputs, trained_model)
    return persist_training_outputs(
        config=config,
        config_path=resolved_config_path,
        training_inputs=training_inputs,
        trained_model=trained_model,
        evaluation_results=evaluation_results,
    )
