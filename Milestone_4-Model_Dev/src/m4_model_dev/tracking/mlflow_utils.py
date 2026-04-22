from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from m4_model_dev.data.benchmark import parse_orlib_uncap, solve_reference_cbc
from m4_model_dev.evaluation.generated_exec import run_generated_solver
from m4_model_dev.models.model_registry import CandidateSpec
from m4_model_dev.paths import M4_MLFLOW_DIR


MLFLOW_DB_PATH = M4_MLFLOW_DIR / "mlflow.db"


def get_tracking_uri() -> str:
    return f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"


def get_registry_uri() -> str:
    return get_tracking_uri()


def _default_experiment_artifact_uri(experiment_name: str) -> str:
    sanitized = experiment_name.replace(" ", "-").lower()
    artifact_root = M4_MLFLOW_DIR / sanitized
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root.resolve().as_uri()


def configure_mlflow():
    import mlflow
    from mlflow.tracking import MlflowClient

    M4_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_registry_uri(get_registry_uri())
    return mlflow, MlflowClient()


def get_or_create_experiment(client, experiment_name: str) -> str:
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(
        name=experiment_name,
        artifact_location=_default_experiment_artifact_uri(experiment_name),
    )


def _artifact_uri_to_path(artifact_uri: str) -> Path | None:
    if not artifact_uri.startswith("file:"):
        return None
    parsed = urlparse(artifact_uri)
    return Path(unquote(parsed.path.lstrip("/")))


def _ensure_run_artifact_dir(mlflow) -> None:
    active_run = mlflow.active_run()
    if active_run is None:
        return
    artifact_path = _artifact_uri_to_path(active_run.info.artifact_uri)
    if artifact_path is None:
        return
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "supporting_artifacts").mkdir(parents=True, exist_ok=True)


def _tracking_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("tracking", {}).get("enable_mlflow", False))


def _log_supporting_artifacts(mlflow, artifacts: list[Path]) -> list[str]:
    artifact_log_errors: list[str] = []
    for artifact in artifacts:
        if not artifact.exists():
            continue
        try:
            mlflow.log_artifact(str(artifact), artifact_path="supporting_artifacts")
        except Exception:
            try:
                mlflow.log_artifact(str(artifact))
            except Exception as fallback_exc:
                artifact_log_errors.append(f"{artifact.name}: {fallback_exc}")
    return artifact_log_errors


class _CandidatePyfuncModelBase:
    def load_context(self, context) -> None:
        payload = json.loads(Path(context.artifacts["candidate_spec"]).read_text(encoding="utf-8"))
        self.candidate_spec = CandidateSpec(**payload)

    def predict(self, context, model_input):
        input_df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        if "instance_path" not in input_df.columns:
            return pd.DataFrame(
                [
                    {
                        "candidate_name": self.candidate_spec.name,
                        "candidate_kind": self.candidate_spec.kind,
                        "backend_name": self.candidate_spec.backend or "",
                        "model_name": self.candidate_spec.model_name or "",
                        "prompt_template": self.candidate_spec.prompt_template or "",
                    }
                ]
            )

        rows: list[dict[str, Any]] = []
        for instance_path_str in input_df["instance_path"].astype(str):
            try:
                if self.candidate_spec.kind == "deterministic_baseline":
                    instance = parse_orlib_uncap(Path(instance_path_str))
                    solved = solve_reference_cbc(instance)
                    rows.append(
                        {
                            "instance_path": instance_path_str,
                            "status": "OK",
                            "objective": solved.objective,
                            "candidate_name": self.candidate_spec.name,
                        }
                    )
                else:
                    from m4_model_dev.models.symbolic_generator import generate_solver_code

                    generated = generate_solver_code(self.candidate_spec)
                    executed = run_generated_solver(generated.code, instance_path_str)
                    rows.append(
                        {
                            "instance_path": instance_path_str,
                            "status": "OK",
                            "objective": executed.objective,
                            "candidate_name": self.candidate_spec.name,
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "instance_path": instance_path_str,
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                        "candidate_name": self.candidate_spec.name,
                    }
                )
        return pd.DataFrame(rows)


def _log_candidate_params(mlflow, config: dict[str, Any], config_path: Path, candidate: CandidateSpec) -> None:
    mlflow.log_param("config_path", str(config_path))
    mlflow.log_param("candidate_name", candidate.name)
    mlflow.log_param("candidate_kind", candidate.kind)
    mlflow.log_param("backend_name", candidate.backend or "")
    mlflow.log_param("model_name", candidate.model_name or "")
    mlflow.log_param("prompt_template", candidate.prompt_template or "")
    mlflow.log_param("candidate_enabled", candidate.enabled)
    mlflow.log_param("temperature", candidate.temperature)
    mlflow.log_param("max_tokens", candidate.max_tokens)
    for key, value in config.get("evaluation", {}).items():
        if isinstance(value, list):
            mlflow.log_param(f"evaluation_{key}", ",".join(str(item) for item in value))
        else:
            mlflow.log_param(f"evaluation_{key}", value)


def _log_metrics_table(mlflow, metrics_df: pd.DataFrame) -> None:
    for row in metrics_df.to_dict(orient="records"):
        split_name = str(row["split"])
        for metric_name, metric_value in row.items():
            if metric_name in {"candidate_name", "candidate_kind", "backend_name", "model_name", "prompt_template", "split"}:
                continue
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"{split_name}_{metric_name}", float(metric_value))


def _log_candidate_model(mlflow, candidate: CandidateSpec, candidate_spec_path: Path, register_model: bool, registered_model_name: str | None) -> str | None:
    import mlflow.pyfunc

    class CandidatePyfuncModel(mlflow.pyfunc.PythonModel, _CandidatePyfuncModelBase):
        pass

    active_registered_model_name = registered_model_name
    if register_model and not active_registered_model_name:
        active_registered_model_name = f"m4-{candidate.name}"

    kwargs = {
        "python_model": CandidatePyfuncModel(),
        "artifacts": {"candidate_spec": str(candidate_spec_path)},
    }
    if register_model and active_registered_model_name:
        kwargs["registered_model_name"] = active_registered_model_name

    try:
        mlflow.pyfunc.log_model(name="model", **kwargs)
    except TypeError:
        mlflow.pyfunc.log_model(artifact_path="model", **kwargs)

    return active_registered_model_name


def log_training_run(
    *,
    config: dict[str, Any],
    candidate: CandidateSpec,
    metrics_df: pd.DataFrame,
    raw_results_df: pd.DataFrame,
    artifacts: list[Path],
    config_path: Path,
    candidate_spec_path: Path,
    register_model: bool,
    registered_model_name: str | None,
) -> dict[str, Any]:
    if not _tracking_enabled(config):
        return {"mlflow_logged": False, "registered_model_name": None, "artifact_log_errors": []}

    try:
        mlflow, client = configure_mlflow()
    except ImportError:
        return {"mlflow_logged": False, "registered_model_name": None, "artifact_log_errors": []}

    experiment_name = str(config.get("tracking", {}).get("experiment_name", "milestone4-symbolic-evaluation"))
    experiment_id = get_or_create_experiment(client, experiment_name)
    run_name = str(config.get("run_name", candidate.name))

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        _ensure_run_artifact_dir(mlflow)
        _log_candidate_params(mlflow, config, config_path, candidate)
        _log_metrics_table(mlflow, metrics_df)
        mlflow.log_metric("evaluated_instances", float(len(raw_results_df)))
        active_registered_model_name = _log_candidate_model(
            mlflow=mlflow,
            candidate=candidate,
            candidate_spec_path=candidate_spec_path,
            register_model=register_model,
            registered_model_name=registered_model_name,
        )
        artifact_log_errors = _log_supporting_artifacts(mlflow, artifacts)

    return {
        "mlflow_logged": True,
        "registered_model_name": active_registered_model_name,
        "artifact_log_errors": artifact_log_errors,
    }


def log_comparison_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    results_df: pd.DataFrame,
    raw_results_df: pd.DataFrame,
    selection_payload: dict[str, Any],
    artifacts: list[Path],
) -> bool:
    if not _tracking_enabled(config):
        return False

    try:
        mlflow, client = configure_mlflow()
    except ImportError:
        return False

    experiment_name = str(config.get("tracking", {}).get("experiment_name", "milestone4-symbolic-comparison"))
    experiment_id = get_or_create_experiment(client, experiment_name)

    with mlflow.start_run(run_name=config.get("run_name", "m4-compare"), experiment_id=experiment_id):
        _ensure_run_artifact_dir(mlflow)
        mlflow.log_param("config_path", str(config_path))
        mlflow.log_param("comparison_candidates", ",".join(sorted(results_df["candidate_name"].unique().tolist())))
        for key, value in selection_payload.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"selection_{key}", float(value))
            else:
                mlflow.log_param(f"selection_{key}", value)

        for candidate_name, candidate_df in results_df.groupby("candidate_name", sort=True):
            with mlflow.start_run(run_name=str(candidate_name), nested=True):
                row0 = candidate_df.iloc[0]
                mlflow.log_param("candidate_name", str(candidate_name))
                mlflow.log_param("candidate_kind", str(row0.get("candidate_kind", "")))
                mlflow.log_param("backend_name", str(row0.get("backend_name", "")))
                mlflow.log_param("model_name", str(row0.get("model_name", "")))
                mlflow.log_param("prompt_template", str(row0.get("prompt_template", "")))
                _log_metrics_table(mlflow, candidate_df)

        mlflow.log_metric("evaluated_instances", float(len(raw_results_df)))
        _log_supporting_artifacts(mlflow, artifacts)
    return True
