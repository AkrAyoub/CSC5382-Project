from __future__ import annotations

from pathlib import Path

import pandas as pd

from m4_model_dev.paths import M4_DATASETS_DIR, M4_SPLITS_DIR


SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"


def assign_split_for_group(instance_ids: list[str]) -> dict[str, str]:
    instance_ids = sorted(instance_ids)
    size = len(instance_ids)
    assignments: dict[str, str] = {}

    if size == 1:
        assignments[instance_ids[0]] = SPLIT_TRAIN
        return assignments
    if size == 2:
        assignments[instance_ids[0]] = SPLIT_TRAIN
        assignments[instance_ids[1]] = SPLIT_TEST
        return assignments
    if size == 3:
        assignments[instance_ids[0]] = SPLIT_TRAIN
        assignments[instance_ids[1]] = SPLIT_VAL
        assignments[instance_ids[2]] = SPLIT_TEST
        return assignments

    for idx, instance_id in enumerate(instance_ids):
        if idx < 2:
            assignments[instance_id] = SPLIT_TRAIN
        elif idx == 2:
            assignments[instance_id] = SPLIT_VAL
        else:
            assignments[instance_id] = SPLIT_TEST
    return assignments


def _build_split_rows(instances: pd.DataFrame) -> list[dict[str, str | int]]:
    split_rows: list[dict[str, str | int]] = []
    for (_, group_df) in instances.groupby(["facility_count_m", "customer_count_n"], sort=True):
        assignments = assign_split_for_group(group_df["instance_id"].tolist())
        for _, row in group_df.iterrows():
            split_rows.append(
                {
                    "instance_id": row["instance_id"],
                    "instance_path": row["instance_path"],
                    "facility_count_m": int(row["facility_count_m"]),
                    "customer_count_n": int(row["customer_count_n"]),
                    "split": assignments[row["instance_id"]],
                }
            )
    return split_rows


def build_grouped_splits(
    dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    dataset_path = dataset_path or (M4_DATASETS_DIR / "benchmark_instances.csv")
    output_path = output_path or (M4_SPLITS_DIR / "instance_splits.csv")

    df = pd.read_csv(dataset_path)
    split_rows = _build_split_rows(
        df[["instance_id", "instance_path", "facility_count_m", "customer_count_n"]]
        .drop_duplicates()
        .sort_values(["facility_count_m", "customer_count_n", "instance_id"])
    )

    split_df = pd.DataFrame(split_rows).sort_values(["split", "facility_count_m", "instance_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    return output_path
