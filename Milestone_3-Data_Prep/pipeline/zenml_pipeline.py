import os
from pathlib import Path
from typing import Any
from uuid import UUID

try:
    from .build_feature_store import run_feature_store_build
    from .common import write_json_file, write_text_file
    from .engineer_features import run_feature_engineering
    from .ingest_data import run_ingestion
    from .preprocess_data import run_preprocessing
    from .validate_data import run_validation
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.build_feature_store import run_feature_store_build
    from pipeline.common import write_json_file, write_text_file
    from pipeline.engineer_features import run_feature_engineering
    from pipeline.ingest_data import run_ingestion
    from pipeline.preprocess_data import run_preprocessing
    from pipeline.validate_data import run_validation

try:
    from ..paths import M3_ROOT, ZEN_CONFIG_DIR, ZEN_LOCAL_STORE_DIR, ZENML_STATUS_JSON, ZENML_STATUS_TXT
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import M3_ROOT, ZEN_CONFIG_DIR, ZEN_LOCAL_STORE_DIR, ZENML_STATUS_JSON, ZENML_STATUS_TXT


def _patch_sqlalchemy_uuid_binding() -> None:
    try:
        import sqlalchemy.sql.sqltypes as sqltypes
    except Exception:
        return

    def _patch_class(class_name: str) -> None:
        target = getattr(sqltypes, class_name, None)
        if target is None or getattr(target, "_m3_uuid_patch_applied", False):
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
        target._m3_uuid_patch_applied = True

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
    def ingest_step() -> dict[str, Any]:
        return run_ingestion()

    @step(enable_cache=False)
    def preprocess_step(ingest_summary: dict[str, Any]) -> dict[str, Any]:
        _ = ingest_summary
        return run_preprocessing()

    @step(enable_cache=False)
    def feature_engineering_step(preprocess_summary: dict[str, Any]) -> dict[str, Any]:
        _ = preprocess_summary
        return run_feature_engineering()

    @step(enable_cache=False)
    def validation_step(
        ingest_summary: dict[str, Any],
        preprocess_summary: dict[str, Any],
        feature_summary: dict[str, Any],
    ) -> dict[str, Any]:
        _ = ingest_summary
        _ = preprocess_summary
        _ = feature_summary
        return run_validation()

    @step(enable_cache=False)
    def publish_feature_store_step(
        feature_summary: dict[str, Any],
        validation_summary: dict[str, Any],
    ) -> dict[str, Any]:
        _ = feature_summary
        _ = validation_summary
        return run_feature_store_build()


if pipeline is not None:
    @pipeline(enable_cache=False)
    def milestone3_pipeline():
        ingest_summary = ingest_step()
        preprocess_summary = preprocess_step(ingest_summary)
        feature_summary = feature_engineering_step(preprocess_summary)
        validation_summary = validation_step(ingest_summary, preprocess_summary, feature_summary)
        publish_feature_store_step(feature_summary, validation_summary)


def _configure_zenml_runtime() -> None:
    for path in (ZEN_CONFIG_DIR, ZEN_LOCAL_STORE_DIR):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("ZENML_CONFIG_PATH", str(ZEN_CONFIG_DIR))
    os.environ.setdefault("ZENML_LOCAL_STORES_PATH", str(ZEN_LOCAL_STORE_DIR))
    os.environ.setdefault("ZENML_DEFAULT_USER_NAME", "default")
    os.environ.setdefault("ZENML_DEFAULT_USER_PASSWORD", "default")
    os.environ.setdefault("ZENML_DISABLE_INTERACTIVE_INPUT", "true")
    os.environ.setdefault("ZENML_DISABLE_PIPELINE_LOGS_STORAGE", "true")
    os.environ.setdefault("ZENML_DISABLE_STEP_LOGS_STORAGE", "true")


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
        client = Client(root=M3_ROOT)
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


def _write_zenml_status(success: bool, details: str) -> dict[str, Any]:
    payload = {
        "success": success,
        "details": details,
    }
    write_json_file(ZENML_STATUS_JSON, payload)
    write_text_file(
        ZENML_STATUS_TXT,
        "\n".join(
            [
                "Milestone 3 ZenML status",
                f"success: {success}",
                f"details: {details}",
            ]
        )
        + "\n",
    )
    return payload


def run_zenml_data_pipeline() -> dict[str, Any]:
    _configure_zenml_runtime()

    if pipeline is None or step is None:
        message = "ZenML is not installed. Install the milestone 3 requirements to run the pipeline."
        _write_zenml_status(False, message)
        raise RuntimeError(message)

    _coerce_uuid_config_values()

    try:
        milestone3_pipeline()
    except Exception as exc:
        return _write_zenml_status(
            False,
            (
                "ZenML pipeline initialization succeeded, but local Windows execution failed "
                f"inside ZenML with: {exc}"
            ),
        )

    return _write_zenml_status(True, "ZenML pipeline executed successfully.")
