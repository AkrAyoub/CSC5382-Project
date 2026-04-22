from __future__ import annotations

from pathlib import Path
import sys

try:
    from .common import (
        group_count,
        is_number,
        read_csv_rows,
        read_json_file,
        safe_mean,
        safe_std,
        write_json_file,
        write_text_file,
    )
except ImportError:
    from common import (
        group_count,
        is_number,
        read_csv_rows,
        read_json_file,
        safe_mean,
        safe_std,
        write_json_file,
        write_text_file,
    )

try:
    from ..paths import (
        FEATURES_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        SCHEMA_DIR,
        STATS_DIR,
        VALIDATION_DIR,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import (
        FEATURES_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        SCHEMA_DIR,
        STATS_DIR,
        VALIDATION_DIR,
    )


def summarize_numeric_column(rows: list[dict[str, str]], column: str) -> dict:
    values = [float(row[column]) for row in rows if row.get(column, "") != "" and is_number(row[column])]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": safe_mean(values),
        "std": safe_std(values)
    }


def validate_raw_layer(raw_schema: dict) -> tuple[list[str], dict]:
    anomalies: list[str] = []
    stats: dict = {}

    expected_instance_files = set(raw_schema["expected_instance_files"])
    required_companion_files = set(raw_schema["required_companion_files"])
    expected_instance_count = int(raw_schema["expected_instance_count"])
    manifest_files = raw_schema["manifest_files"]

    present_raw_files = {p.name for p in RAW_DATA_DIR.glob("*") if p.is_file()}
    present_instance_files = {name for name in present_raw_files if name in expected_instance_files}

    missing_instances = sorted(expected_instance_files - present_instance_files)
    missing_companions = sorted(required_companion_files - present_raw_files)

    if missing_instances:
        anomalies.append(f"raw layer: missing expected instance files: {missing_instances}")

    if missing_companions:
        anomalies.append(f"raw layer: missing required companion files: {missing_companions}")

    if len(present_instance_files) != expected_instance_count:
        anomalies.append(
            f"raw layer: instance file count is {len(present_instance_files)}, expected {expected_instance_count}"
        )

    for manifest_name in manifest_files:
        manifest_path = INTERIM_DATA_DIR / manifest_name
        if not manifest_path.exists():
            anomalies.append(f"raw layer: missing ingestion manifest file {manifest_name}")

    stats["present_raw_files"] = sorted(present_raw_files)
    stats["present_instance_files"] = sorted(present_instance_files)
    stats["instance_count"] = len(present_instance_files)
    stats["missing_instances"] = missing_instances
    stats["missing_companion_files"] = missing_companions

    return anomalies, stats


def validate_file_against_schema(
    file_name: str,
    rows: list[dict[str, str]],
    schema: dict
) -> tuple[list[str], dict]:
    anomalies: list[str] = []
    stats: dict = {
        "row_count": len(rows),
        "column_stats": {}
    }

    file_schema = schema[file_name]
    required_columns = file_schema["required_columns"]
    numeric_columns = file_schema["numeric_columns"]
    non_negative_columns = file_schema["non_negative_columns"]
    id_columns = file_schema["id_columns"]

    actual_columns = list(rows[0].keys()) if rows else []

    for col in required_columns:
        if col not in actual_columns:
            anomalies.append(f"{file_name}: missing required column '{col}'")

    for col in required_columns:
        if col in actual_columns:
            missing_rows = [
                idx for idx, row in enumerate(rows, start=1)
                if row.get(col, "") == ""
            ]
            if missing_rows:
                anomalies.append(
                    f"{file_name}: column '{col}' has empty values in rows {missing_rows[:10]}"
                )

    for col in numeric_columns:
        if col in actual_columns:
            bad_rows = [
                idx for idx, row in enumerate(rows, start=1)
                if row.get(col, "") == "" or not is_number(row[col])
            ]
            if bad_rows:
                anomalies.append(
                    f"{file_name}: column '{col}' has non-numeric values in rows {bad_rows[:10]}"
                )
            else:
                stats["column_stats"][col] = summarize_numeric_column(rows, col)

    for col in non_negative_columns:
        if col in actual_columns:
            bad_rows = [
                idx for idx, row in enumerate(rows, start=1)
                if is_number(row[col]) and float(row[col]) < 0
            ]
            if bad_rows:
                anomalies.append(
                    f"{file_name}: column '{col}' has negative values in rows {bad_rows[:10]}"
                )

    if id_columns and all(col in actual_columns for col in id_columns):
        seen = set()
        duplicates = []
        for idx, row in enumerate(rows, start=1):
            key = tuple(row[col] for col in id_columns)
            if key in seen:
                duplicates.append({"row": idx, "key": key})
            else:
                seen.add(key)
        if duplicates:
            anomalies.append(
                f"{file_name}: duplicate keys found for id columns {id_columns}: {duplicates[:10]}"
            )

    return anomalies, stats


def validate_processed_consistency(
    instances: list[dict[str, str]],
    facilities: list[dict[str, str]],
    customers: list[dict[str, str]],
    assignments: list[dict[str, str]],
) -> list[str]:
    anomalies: list[str] = []

    facilities_by_instance = group_count(facilities, "instance_id")
    customers_by_instance = group_count(customers, "instance_id")
    assignments_by_instance = group_count(assignments, "instance_id")

    for row in instances:
        instance_id = row["instance_id"]
        m = int(float(row["facility_count_m"]))
        n = int(float(row["customer_count_n"]))
        expected_assignments = m * n

        actual_m = facilities_by_instance.get(instance_id, 0)
        actual_n = customers_by_instance.get(instance_id, 0)
        actual_assignments = assignments_by_instance.get(instance_id, 0)

        if actual_m != m:
            anomalies.append(
                f"processed consistency: {instance_id} facilities count {actual_m} != expected {m}"
            )
        if actual_n != n:
            anomalies.append(
                f"processed consistency: {instance_id} customers count {actual_n} != expected {n}"
            )
        if actual_assignments != expected_assignments:
            anomalies.append(
                f"processed consistency: {instance_id} assignment row count {actual_assignments} != expected {expected_assignments}"
            )

    return anomalies


def validate_feature_consistency(
    instance_features: list[dict[str, str]],
    facility_features: list[dict[str, str]],
    customer_features: list[dict[str, str]],
    instances: list[dict[str, str]],
    facilities: list[dict[str, str]],
    customers: list[dict[str, str]],
) -> list[str]:
    anomalies: list[str] = []

    if len(instance_features) != len(instances):
        anomalies.append(
            f"feature consistency: instance_features rows {len(instance_features)} != instances rows {len(instances)}"
        )
    if len(facility_features) != len(facilities):
        anomalies.append(
            f"feature consistency: facility_features rows {len(facility_features)} != facilities rows {len(facilities)}"
        )
    if len(customer_features) != len(customers):
        anomalies.append(
            f"feature consistency: customer_features rows {len(customer_features)} != customers rows {len(customers)}"
        )

    for row in facility_features:
        val = float(row["normalized_fixed_cost_minmax"])
        if not (0.0 <= val <= 1.0):
            anomalies.append(
                f"facility_features.csv: normalized_fixed_cost_minmax out of [0,1] for instance {row['instance_id']} facility {row['facility_id']}"
            )

    for row in customer_features:
        nearest_id = int(float(row["nearest_facility_id"]))
        if nearest_id < 0:
            anomalies.append(
                f"customer_features.csv: nearest_facility_id < 0 for instance {row['instance_id']} customer {row['customer_id']}"
            )

    return anomalies


def run_validation() -> dict[str, object]:
    raw_schema = read_json_file(SCHEMA_DIR / "raw_schema.json")
    processed_schema = read_json_file(SCHEMA_DIR / "processed_schema.json")
    feature_schema = read_json_file(SCHEMA_DIR / "feature_schema.json")

    raw_anomalies, raw_stats = validate_raw_layer(raw_schema)

    instances = read_csv_rows(PROCESSED_DATA_DIR / "instances.csv")
    facilities = read_csv_rows(PROCESSED_DATA_DIR / "facilities.csv")
    customers = read_csv_rows(PROCESSED_DATA_DIR / "customers.csv")
    assignments = read_csv_rows(PROCESSED_DATA_DIR / "assignment_costs.csv")

    instance_features = read_csv_rows(FEATURES_DATA_DIR / "instance_features.csv")
    facility_features = read_csv_rows(FEATURES_DATA_DIR / "facility_features.csv")
    customer_features = read_csv_rows(FEATURES_DATA_DIR / "customer_features.csv")

    processed_files = {
        "instances.csv": instances,
        "facilities.csv": facilities,
        "customers.csv": customers,
        "assignment_costs.csv": assignments
    }

    feature_files = {
        "instance_features.csv": instance_features,
        "facility_features.csv": facility_features,
        "customer_features.csv": customer_features
    }

    processed_anomalies: list[str] = []
    feature_anomalies: list[str] = []
    processed_stats: dict = {}
    feature_stats: dict = {}

    for file_name, rows in processed_files.items():
        anomalies, stats = validate_file_against_schema(file_name, rows, processed_schema)
        processed_anomalies.extend(anomalies)
        processed_stats[file_name] = stats

    for file_name, rows in feature_files.items():
        anomalies, stats = validate_file_against_schema(file_name, rows, feature_schema)
        feature_anomalies.extend(anomalies)
        feature_stats[file_name] = stats

    processed_consistency_anomalies = validate_processed_consistency(
        instances, facilities, customers, assignments
    )

    feature_consistency_anomalies = validate_feature_consistency(
        instance_features,
        facility_features,
        customer_features,
        instances,
        facilities,
        customers
    )

    all_anomalies = (
        raw_anomalies
        + processed_anomalies
        + feature_anomalies
        + processed_consistency_anomalies
        + feature_consistency_anomalies
    )

    summary = {
        "status": "passed" if not all_anomalies else "warnings_found",
        "anomaly_count": len(all_anomalies),
        "raw_layer": {
            "instance_count": raw_stats["instance_count"]
        },
        "processed_layer": {
            "instances": len(instances),
            "facilities": len(facilities),
            "customers": len(customers),
            "assignment_costs": len(assignments)
        },
        "feature_layer": {
            "instance_features": len(instance_features),
            "facility_features": len(facility_features),
            "customer_features": len(customer_features)
        }
    }

    write_json_file(VALIDATION_DIR / "validation_summary.json", summary)
    write_json_file(VALIDATION_DIR / "anomalies.json", {"anomalies": all_anomalies})
    write_text_file(
        VALIDATION_DIR / "anomalies.txt",
        "No anomalies detected." if not all_anomalies else "\n".join(f"- {a}" for a in all_anomalies)
    )

    write_json_file(STATS_DIR / "raw_statistics.json", raw_stats)
    write_json_file(STATS_DIR / "processed_statistics.json", processed_stats)
    write_json_file(STATS_DIR / "feature_statistics.json", feature_stats)

    print("=== Milestone 3: Full Data Validation Summary ===")
    print(f"Raw instance count: {raw_stats['instance_count']}")
    print(f"Processed rows: instances={len(instances)}, facilities={len(facilities)}, customers={len(customers)}, assignment_costs={len(assignments)}")
    print(f"Feature rows: instance_features={len(instance_features)}, facility_features={len(facility_features)}, customer_features={len(customer_features)}")
    print(f"Anomalies found: {len(all_anomalies)}")
    print(f"Validation status: {summary['status']}")
    print("\nValidation artifacts:")
    print(f"- {VALIDATION_DIR / 'validation_summary.json'}")
    print(f"- {VALIDATION_DIR / 'anomalies.json'}")
    print(f"- {VALIDATION_DIR / 'anomalies.txt'}")
    print(f"- {STATS_DIR / 'raw_statistics.json'}")
    print(f"- {STATS_DIR / 'processed_statistics.json'}")
    print(f"- {STATS_DIR / 'feature_statistics.json'}")

    return {
        "status": summary["status"],
        "anomaly_count": summary["anomaly_count"],
        "raw_instance_count": raw_stats["instance_count"],
        "processed_instances": len(instances),
        "processed_facilities": len(facilities),
        "processed_customers": len(customers),
        "processed_assignment_costs": len(assignments),
        "feature_instances": len(instance_features),
        "feature_facilities": len(facility_features),
        "feature_customers": len(customer_features),
        "validation_summary_path": str(VALIDATION_DIR / "validation_summary.json"),
        "anomalies_path": str(VALIDATION_DIR / "anomalies.json"),
        "raw_stats_path": str(STATS_DIR / "raw_statistics.json"),
        "processed_stats_path": str(STATS_DIR / "processed_statistics.json"),
        "feature_stats_path": str(STATS_DIR / "feature_statistics.json"),
    }


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
