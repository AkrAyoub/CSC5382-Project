from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

try:
    from .common import read_csv_rows, safe_mean, safe_std, write_dataclass_rows_json
except ImportError:
    from common import read_csv_rows, safe_mean, safe_std, write_dataclass_rows_json

try:
    from ..paths import FEATURES_DATA_DIR, PROCESSED_DATA_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import FEATURES_DATA_DIR, PROCESSED_DATA_DIR

# A fixed timestamp keeps Feast joins deterministic for this milestone dataset.
DEFAULT_EVENT_TIMESTAMP = datetime(2026, 3, 21, 0, 0, 0, tzinfo=timezone.utc).isoformat()


def make_facility_key(instance_id: str, facility_id: int) -> str:
    return f"{instance_id}__{facility_id}"


def make_customer_key(instance_id: str, customer_id: int) -> str:
    return f"{instance_id}__{customer_id}"


@dataclass
class InstanceFeatureRow:
    instance_id: str
    event_timestamp: str
    facility_count_m: int
    customer_count_n: int
    facility_customer_ratio: float
    total_fixed_cost: float
    avg_fixed_cost: float
    min_fixed_cost: float
    max_fixed_cost: float
    std_fixed_cost: float
    total_assignment_cost_entries: int
    avg_assignment_cost: float
    min_assignment_cost: float
    max_assignment_cost: float
    std_assignment_cost: float
    fixed_cost_range: float
    assignment_cost_range: float


@dataclass
class FacilityFeatureRow:
    facility_key: str
    instance_id: str
    facility_id: int
    event_timestamp: str
    fixed_cost: float
    normalized_fixed_cost_minmax: float
    fixed_cost_zscore: float
    fixed_cost_rank_ascending: int
    avg_assignment_cost_from_facility: float
    min_assignment_cost_from_facility: float
    max_assignment_cost_from_facility: float
    std_assignment_cost_from_facility: float


@dataclass
class CustomerFeatureRow:
    customer_key: str
    instance_id: str
    customer_id: int
    event_timestamp: str
    min_assignment_cost: float
    max_assignment_cost: float
    avg_assignment_cost: float
    std_assignment_cost: float
    assignment_cost_range: float
    nearest_facility_id: int
    nearest_facility_cost: float


def write_json(rows: list[object], output_path: Path) -> None:
    write_dataclass_rows_json(rows, output_path)


def write_csv_and_parquet(rows: list[object], csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        pd.DataFrame().to_csv(csv_path, index=False)
        pd.DataFrame().to_parquet(parquet_path, index=False)
        return

    df = pd.DataFrame([asdict(row) for row in rows])

    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def group_by_instance(rows: list[dict[str, str]], key: str = "instance_id") -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def safe_minmax_normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def safe_zscore(value: float, mean_value: float, std_value: float) -> float:
    if std_value == 0:
        return 0.0
    return (value - mean_value) / std_value


def build_instance_features(
    instances_rows: list[dict[str, str]],
    assignment_rows: list[dict[str, str]],
) -> list[InstanceFeatureRow]:
    assignment_by_instance = group_by_instance(assignment_rows)
    results: list[InstanceFeatureRow] = []

    for row in instances_rows:
        instance_id = row["instance_id"]
        assignment_values = [
            float(r["assignment_cost"])
            for r in assignment_by_instance.get(instance_id, [])
        ]

        total_fixed_cost = float(row["total_fixed_cost"])
        avg_fixed_cost = float(row["avg_fixed_cost"])
        min_fixed_cost = float(row["min_fixed_cost"])
        max_fixed_cost = float(row["max_fixed_cost"])

        m = int(row["facility_count_m"])
        n = int(row["customer_count_n"])

        avg_assignment_cost = safe_mean(assignment_values)
        min_assignment_cost = min(assignment_values) if assignment_values else 0.0
        max_assignment_cost = max(assignment_values) if assignment_values else 0.0
        std_assignment_cost = safe_std(assignment_values)

        results.append(
            InstanceFeatureRow(
                instance_id=instance_id,
                event_timestamp=DEFAULT_EVENT_TIMESTAMP,
                facility_count_m=m,
                customer_count_n=n,
                facility_customer_ratio=(m / n) if n != 0 else 0.0,
                total_fixed_cost=total_fixed_cost,
                avg_fixed_cost=avg_fixed_cost,
                min_fixed_cost=min_fixed_cost,
                max_fixed_cost=max_fixed_cost,
                std_fixed_cost=0.0,
                total_assignment_cost_entries=int(row["total_assignment_cost_entries"]),
                avg_assignment_cost=avg_assignment_cost,
                min_assignment_cost=min_assignment_cost,
                max_assignment_cost=max_assignment_cost,
                std_assignment_cost=std_assignment_cost,
                fixed_cost_range=max_fixed_cost - min_fixed_cost,
                assignment_cost_range=max_assignment_cost - min_assignment_cost,
            )
        )

    return results


def build_facility_features(
    facility_rows: list[dict[str, str]],
    assignment_rows: list[dict[str, str]],
) -> list[FacilityFeatureRow]:
    facilities_by_instance = group_by_instance(facility_rows)

    assignment_map: dict[tuple[str, int], list[float]] = {}
    for row in assignment_rows:
        instance_id = row["instance_id"]
        facility_id = int(row["facility_id"])
        assignment_map.setdefault((instance_id, facility_id), []).append(float(row["assignment_cost"]))

    results: list[FacilityFeatureRow] = []

    for instance_id, rows in facilities_by_instance.items():
        fixed_costs = [float(r["fixed_cost"]) for r in rows]
        mean_fixed = safe_mean(fixed_costs)
        std_fixed = safe_std(fixed_costs)
        min_fixed = min(fixed_costs) if fixed_costs else 0.0
        max_fixed = max(fixed_costs) if fixed_costs else 0.0

        ranked_facilities = sorted(
            [(int(r["facility_id"]), float(r["fixed_cost"])) for r in rows],
            key=lambda x: x[1]
        )
        rank_map = {facility_id: rank + 1 for rank, (facility_id, _) in enumerate(ranked_facilities)}

        for r in rows:
            facility_id = int(r["facility_id"])
            fixed_cost = float(r["fixed_cost"])
            assignment_values = assignment_map.get((instance_id, facility_id), [])

            results.append(
                FacilityFeatureRow(
                    facility_key=make_facility_key(instance_id, facility_id),
                    instance_id=instance_id,
                    facility_id=facility_id,
                    event_timestamp=DEFAULT_EVENT_TIMESTAMP,
                    fixed_cost=fixed_cost,
                    normalized_fixed_cost_minmax=safe_minmax_normalize(fixed_cost, min_fixed, max_fixed),
                    fixed_cost_zscore=safe_zscore(fixed_cost, mean_fixed, std_fixed),
                    fixed_cost_rank_ascending=rank_map[facility_id],
                    avg_assignment_cost_from_facility=safe_mean(assignment_values),
                    min_assignment_cost_from_facility=min(assignment_values) if assignment_values else 0.0,
                    max_assignment_cost_from_facility=max(assignment_values) if assignment_values else 0.0,
                    std_assignment_cost_from_facility=safe_std(assignment_values),
                )
            )

    return results


def build_customer_features(
    customer_rows: list[dict[str, str]],
    assignment_rows: list[dict[str, str]],
) -> list[CustomerFeatureRow]:
    assignment_map: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in assignment_rows:
        instance_id = row["instance_id"]
        customer_id = int(row["customer_id"])
        facility_id = int(row["facility_id"])
        cost = float(row["assignment_cost"])
        assignment_map.setdefault((instance_id, customer_id), []).append((facility_id, cost))

    results: list[CustomerFeatureRow] = []

    for row in customer_rows:
        instance_id = row["instance_id"]
        customer_id = int(row["customer_id"])
        values = assignment_map.get((instance_id, customer_id), [])

        costs = [cost for _, cost in values]
        nearest_facility_id = min(values, key=lambda x: x[1])[0] if values else -1
        nearest_facility_cost = min(costs) if costs else 0.0

        results.append(
            CustomerFeatureRow(
                customer_key=make_customer_key(instance_id, customer_id),
                instance_id=instance_id,
                customer_id=customer_id,
                event_timestamp=DEFAULT_EVENT_TIMESTAMP,
                min_assignment_cost=float(row["min_assignment_cost"]),
                max_assignment_cost=float(row["max_assignment_cost"]),
                avg_assignment_cost=float(row["avg_assignment_cost"]),
                std_assignment_cost=safe_std(costs),
                assignment_cost_range=float(row["max_assignment_cost"]) - float(row["min_assignment_cost"]),
                nearest_facility_id=nearest_facility_id,
                nearest_facility_cost=nearest_facility_cost,
            )
        )

    return results


def enrich_instance_fixed_cost_std(
    instance_features: list[InstanceFeatureRow],
    facility_features: list[FacilityFeatureRow],
) -> list[InstanceFeatureRow]:
    fixed_costs_by_instance: dict[str, list[float]] = {}
    for row in facility_features:
        fixed_costs_by_instance.setdefault(row.instance_id, []).append(row.fixed_cost)

    enriched: list[InstanceFeatureRow] = []
    for row in instance_features:
        fixed_costs = fixed_costs_by_instance.get(row.instance_id, [])
        row.std_fixed_cost = safe_std(fixed_costs)
        enriched.append(row)

    return enriched


def run_feature_engineering() -> dict[str, object]:
    instances_rows = read_csv_rows(PROCESSED_DATA_DIR / "instances.csv")
    facility_rows = read_csv_rows(PROCESSED_DATA_DIR / "facilities.csv")
    customer_rows = read_csv_rows(PROCESSED_DATA_DIR / "customers.csv")
    assignment_rows = read_csv_rows(PROCESSED_DATA_DIR / "assignment_costs.csv")

    instance_features = build_instance_features(instances_rows, assignment_rows)
    facility_features = build_facility_features(facility_rows, assignment_rows)
    customer_features = build_customer_features(customer_rows, assignment_rows)
    instance_features = enrich_instance_fixed_cost_std(instance_features, facility_features)

    write_csv_and_parquet(
        instance_features,
        FEATURES_DATA_DIR / "instance_features.csv",
        FEATURES_DATA_DIR / "instance_features.parquet",
    )
    write_csv_and_parquet(
        facility_features,
        FEATURES_DATA_DIR / "facility_features.csv",
        FEATURES_DATA_DIR / "facility_features.parquet",
    )
    write_csv_and_parquet(
        customer_features,
        FEATURES_DATA_DIR / "customer_features.csv",
        FEATURES_DATA_DIR / "customer_features.parquet",
    )

    write_json(instance_features, FEATURES_DATA_DIR / "instance_features.json")
    write_json(facility_features, FEATURES_DATA_DIR / "facility_features.json")
    write_json(customer_features, FEATURES_DATA_DIR / "customer_features.json")

    print("=== Milestone 3: Feature Engineering Summary ===")
    print(f"Instance feature rows written: {len(instance_features)}")
    print(f"Facility feature rows written: {len(facility_features)}")
    print(f"Customer feature rows written: {len(customer_features)}")
    print("\nFeature files written to:")
    print(f"- {FEATURES_DATA_DIR / 'instance_features.csv'}")
    print(f"- {FEATURES_DATA_DIR / 'instance_features.parquet'}")
    print(f"- {FEATURES_DATA_DIR / 'facility_features.csv'}")
    print(f"- {FEATURES_DATA_DIR / 'facility_features.parquet'}")
    print(f"- {FEATURES_DATA_DIR / 'customer_features.csv'}")
    print(f"- {FEATURES_DATA_DIR / 'customer_features.parquet'}")

    return {
        "instance_feature_rows": len(instance_features),
        "facility_feature_rows": len(facility_features),
        "customer_feature_rows": len(customer_features),
        "instance_features_csv": str(FEATURES_DATA_DIR / "instance_features.csv"),
        "facility_features_csv": str(FEATURES_DATA_DIR / "facility_features.csv"),
        "customer_features_csv": str(FEATURES_DATA_DIR / "customer_features.csv"),
        "instance_features_parquet": str(FEATURES_DATA_DIR / "instance_features.parquet"),
        "facility_features_parquet": str(FEATURES_DATA_DIR / "facility_features.parquet"),
        "customer_features_parquet": str(FEATURES_DATA_DIR / "customer_features.parquet"),
    }


def main() -> None:
    run_feature_engineering()


if __name__ == "__main__":
    main()
