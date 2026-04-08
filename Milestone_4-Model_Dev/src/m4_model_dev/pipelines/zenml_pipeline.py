import os
from pathlib import Path
from typing import Any
from uuid import UUID

from m4_model_dev.paths import M4_REPORTS_DIR, M4_ROOT
from m4_model_dev.pipelines.training_pipeline import (
    evaluate_model_bundle,
    load_training_config,
    persist_training_outputs,
    prepare_training_inputs,
    train_model_bundle,
)
from m4_model_dev.utils.io import write_json, write_text


def _patch_sqlalchemy_uuid_binding() -> None:
    # ZenML's local Windows stack can hand string UUIDs to SQLAlchemy UUID bind
    # processors during step execution. Coercing those values keeps the local
    # ZenML requirement working in this environment without changing the actual
    # training pipeline logic.
    try:
        import sqlalchemy.sql.sqltypes as sqltypes
    except Exception:
        return

    def _patch_class(class_name: str) -> None:
        target = getattr(sqltypes, class_name, None)
        if target is None or getattr(target, "_m4_uuid_patch_applied", False):
            return

        original = getattr(target, "bind_processor", None)
        if original is None:
            return

        def bind_processor(self, dialect):
            processor = original(self, dialect)
            if processor is None:
                return None

            def wrapped(value):
                if isinstance(value, str):
                    try:
                        value = UUID(value)
                    except ValueError:
                        pass
                return processor(value)

            return wrapped

        target.bind_processor = bind_processor
        target._m4_uuid_patch_applied = True

    _patch_class("Uuid")
    _patch_class("UUID")


_patch_sqlalchemy_uuid_binding()

try:
    from zenml import pipeline, step
except ImportError:
    pipeline = None
    step = None


if step is not None:
    @step(enable_cache=False)
    def load_config_step(config_path_str: str | None = None) -> dict[str, Any]:
        path = Path(config_path_str) if config_path_str else None
        _, config = load_training_config(path)
        return config

    @step(enable_cache=False)
    def prepare_data_step(config: dict[str, Any]) -> dict[str, Any]:
        return prepare_training_inputs(config)

    @step(enable_cache=False)
    def train_model_step(config: dict[str, Any], training_inputs: dict[str, Any]) -> dict[str, Any]:
        return train_model_bundle(config, training_inputs)

    @step(enable_cache=False)
    def evaluate_model_step(
        config: dict[str, Any],
        training_inputs: dict[str, Any],
        trained_model: dict[str, Any],
    ) -> dict[str, Any]:
        return evaluate_model_bundle(config, training_inputs, trained_model)

    @step(enable_cache=False)
    def persist_outputs_step(
        config_path_str: str | None,
        config: dict[str, Any],
        training_inputs: dict[str, Any],
        trained_model: dict[str, Any],
        evaluation_results: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_config_path, _ = load_training_config(Path(config_path_str) if config_path_str else None)
        return persist_training_outputs(
            config=config,
            config_path=resolved_config_path,
            training_inputs=training_inputs,
            trained_model=trained_model,
            evaluation_results=evaluation_results,
        )


if pipeline is not None:
    @pipeline(enable_cache=False)
    def milestone4_pipeline(config_path_str=None):
        config = load_config_step(config_path_str)
        training_inputs = prepare_data_step(config)
        trained_model = train_model_step(config, training_inputs)
        evaluation_results = evaluate_model_step(config, training_inputs, trained_model)
        persist_outputs_step(
            config_path_str,
            config,
            training_inputs,
            trained_model,
            evaluation_results,
        )


def _coerce_uuid_config_values() -> None:
    try:
        from zenml.client import Client
        from zenml.config.global_config import GlobalConfiguration
    except Exception:
        return

    global_config = GlobalConfiguration()
    for attr in ("active_project_id", "active_stack_id", "user_id"):
        value = getattr(global_config, attr, None)
        if isinstance(value, str):
            try:
                setattr(global_config, attr, UUID(value))
            except ValueError:
                pass

    try:
        client = Client(root=M4_ROOT)
        if client._config is not None:
            for attr in ("active_project_id", "active_stack_id"):
                value = getattr(client._config, attr, None)
                if isinstance(value, str):
                    try:
                        setattr(client._config, attr, UUID(value))
                    except ValueError:
                        pass
    except Exception:
        return


def _write_zenml_status(success: bool, config_path: Path | None, details: str) -> dict[str, str | bool]:
    payload = {
        "success": success,
        "config_path": str(config_path) if config_path else "",
        "details": details,
    }
    write_json(payload, M4_REPORTS_DIR / "zenml_status.json")
    write_text(
        "\n".join(
            [
                "Milestone 4 ZenML status",
                f"success: {success}",
                f"config_path: {config_path}" if config_path else "config_path: ",
                f"details: {details}",
            ]
        )
        + "\n",
        M4_REPORTS_DIR / "zenml_status.txt",
    )
    return payload


def run_zenml_training_pipeline(config_path: Path | None = None):
    zen_root = M4_ROOT / ".zen"
    zen_root.mkdir(parents=True, exist_ok=True)
    # Keep the local store path short on Windows to avoid step artifact paths
    # exceeding filesystem limits during multi-step materialization.
    local_store_root = M4_ROOT.parent.parent / ".zen_local"
    local_store_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ZENML_CONFIG_PATH", str(zen_root))
    os.environ.setdefault("ZENML_LOCAL_STORES_PATH", str(local_store_root))
    os.environ.setdefault("ZENML_DEFAULT_USER_NAME", "default")
    os.environ.setdefault("ZENML_DEFAULT_USER_PASSWORD", "default")
    os.environ.setdefault("ZENML_DISABLE_INTERACTIVE_INPUT", "true")
    os.environ.setdefault("ZENML_DISABLE_PIPELINE_LOGS_STORAGE", "true")
    os.environ.setdefault("ZENML_DISABLE_STEP_LOGS_STORAGE", "true")

    if pipeline is None or step is None:
        message = "ZenML is not installed. Install the milestone 4 requirements to run the pipeline."
        _write_zenml_status(False, config_path, message)
        raise RuntimeError(message)

    _coerce_uuid_config_values()

    try:
        milestone4_pipeline(config_path_str=str(config_path) if config_path else None)
    except Exception as exc:
        return _write_zenml_status(
            False,
            config_path,
            (
                "ZenML pipeline initialization succeeded, but local Windows execution failed "
                f"inside ZenML with: {exc}"
            ),
        )

    return _write_zenml_status(
        True,
        config_path,
        "ZenML pipeline executed successfully.",
    )
