from __future__ import annotations

from pathlib import Path


def maybe_start_codecarbon(enable: bool, output_dir: Path, output_file: str):
    if not enable:
        return None
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        return None
    return EmissionsTracker(
        project_name="m4-symbolic-generator-eval",
        output_dir=str(output_dir),
        output_file=output_file,
        save_to_file=True,
    )
