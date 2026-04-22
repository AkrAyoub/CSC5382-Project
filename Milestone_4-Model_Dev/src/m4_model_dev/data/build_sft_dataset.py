from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from m4_model_dev.data.benchmark import build_reference_solver_module_source
from m4_model_dev.paths import M4_SFT_DIR


ROBUST_PROMPT_TEMPLATE = """
Write ONE self-contained Python module that solves the OR-Library UNCAPACITATED Facility Location Problem (UFLP) with OR-Tools CBC.

Requirements:
- define solve(instance_path: str) -> dict
- parse both OR-Library variants using line-based parsing and numeric-token extraction
- use solver = pywraplp.Solver.CreateSolver("CBC")
- return keys: objective, open_facilities, assignments
- no prints, no filesystem writes, no unsafe imports
""".strip()


def build_sft_dataset(
    split_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or M4_SFT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(split_path)
    teacher_response = build_reference_solver_module_source()

    outputs: dict[str, Path] = {}
    for split_name, group_df in split_df.groupby("split", sort=True):
        output_path = output_dir / f"sft_{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for _, row in group_df.iterrows():
                payload = {
                    "instance_id": row["instance_id"],
                    "instance_path": row["instance_path"],
                    "split": split_name,
                    "prompt_template": "robust_v1",
                    "prompt": ROBUST_PROMPT_TEMPLATE,
                    "response": teacher_response,
                }
                handle.write(json.dumps(payload) + "\n")
        outputs[split_name] = output_path

    manifest_path = output_dir / "sft_manifest.json"
    manifest_path.write_text(
        json.dumps({split_name: str(path) for split_name, path in outputs.items()}, indent=2),
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path
    return outputs
