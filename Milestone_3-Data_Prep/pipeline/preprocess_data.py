from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

try:
    from .common import safe_mean, write_dataclass_rows_csv, write_dataclass_rows_json
except ImportError:
    from common import safe_mean, write_dataclass_rows_csv, write_dataclass_rows_json

try:
    from ..paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

ALLOWED_INSTANCE_FILES = {
    "cap71.txt",
    "cap72.txt",
    "cap73.txt",
    "cap74.txt",
    "cap101.txt",
    "cap102.txt",
    "cap103.txt",
    "cap104.txt",
    "cap131.txt",
    "cap132.txt",
    "cap133.txt",
    "cap134.txt",
    "capa.txt",
    "capb.txt",
    "capc.txt",
}


@dataclass
class UFLPInstance:
    instance_id: str
    m: int
    n: int
    fixed_costs: list[float]
    costs: list[list[float]]  # costs[customer_id][facility_id]


@dataclass
class InstanceRow:
    instance_id: str
    facility_count_m: int
    customer_count_n: int
    total_fixed_cost: float
    avg_fixed_cost: float
    min_fixed_cost: float
    max_fixed_cost: float
    total_assignment_cost_entries: int


@dataclass
class FacilityRow:
    instance_id: str
    facility_id: int
    fixed_cost: float


@dataclass
class CustomerRow:
    instance_id: str
    customer_id: int
    min_assignment_cost: float
    max_assignment_cost: float
    avg_assignment_cost: float


@dataclass
class AssignmentCostRow:
    instance_id: str
    customer_id: int
    facility_id: int
    assignment_cost: float


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def extract_numeric_tokens(line: str) -> list[float]:
    return [float(token) for token in NUMBER_PATTERN.findall(line)]


def parse_orlib_uncap(instance_path: Path) -> UFLPInstance:
    """Parse OR-Library UFLP files that may use either numeric or labeled facility rows."""
    lines = [line.strip() for line in instance_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty instance file: {instance_path}")

    header = extract_numeric_tokens(lines[0])
    if len(header) < 2:
        raise ValueError(f"Invalid header in {instance_path.name}: {lines[0]!r}")

    m = int(header[0])
    n = int(header[1])
    line_idx = 1

    fixed_costs: list[float] = []
    for facility_idx in range(m):
        if line_idx >= len(lines):
            raise ValueError(f"Unexpected end of file while reading facilities in {instance_path.name}")

        facility_values = extract_numeric_tokens(lines[line_idx])
        line_idx += 1

        if not facility_values:
            raise ValueError(f"Missing facility values for facility {facility_idx} in {instance_path.name}")

        # Small instances store `capacity fixed_cost`; large ones store `capacity <fixed_cost>` with
        # the capacity token written literally, leaving only one numeric value on the line.
        fixed_costs.append(facility_values[-1])

    costs: list[list[float]] = []
    for customer_idx in range(n):
        if line_idx >= len(lines):
            raise ValueError(f"Unexpected end of file while reading customer {customer_idx} in {instance_path.name}")

        demand_values = extract_numeric_tokens(lines[line_idx])
        line_idx += 1
        if not demand_values:
            raise ValueError(f"Missing demand value for customer {customer_idx} in {instance_path.name}")

        row: list[float] = []
        while len(row) < m:
            if line_idx >= len(lines):
                raise ValueError(
                    f"Unexpected end of file while reading assignment costs for customer {customer_idx} in {instance_path.name}"
                )
            row.extend(extract_numeric_tokens(lines[line_idx]))
            line_idx += 1

        costs.append(row[:m])

    return UFLPInstance(
        instance_id=instance_path.stem,
        m=m,
        n=n,
        fixed_costs=fixed_costs,
        costs=costs,
    )


def discover_instance_files(raw_dir: Path) -> list[Path]:
    """Return only the curated OR-Library instance files for this milestone."""
    files: list[Path] = []
    skipped: list[str] = []

    for path in sorted(raw_dir.glob("*.txt")):
        if path.name.lower() in ALLOWED_INSTANCE_FILES:
            files.append(path)
        else:
            skipped.append(path.name)

    print("=== Raw File Discovery ===")
    print(f"Accepted instance files: {len(files)}")
    if files:
        for p in files:
            print(f"  - {p.name}")

    if skipped:
        print(f"Skipped non-instance text files: {len(skipped)}")
        for name in skipped:
            print(f"  - {name}")

    return files


def build_rows(instances: Iterable[UFLPInstance]) -> tuple[
    list[InstanceRow],
    list[FacilityRow],
    list[CustomerRow],
    list[AssignmentCostRow],
]:
    instance_rows: list[InstanceRow] = []
    facility_rows: list[FacilityRow] = []
    customer_rows: list[CustomerRow] = []
    assignment_rows: list[AssignmentCostRow] = []

    for inst in instances:
        total_fixed_cost = sum(inst.fixed_costs)
        all_assignment_values = [value for row in inst.costs for value in row]

        instance_rows.append(
            InstanceRow(
                instance_id=inst.instance_id,
                facility_count_m=inst.m,
                customer_count_n=inst.n,
                total_fixed_cost=total_fixed_cost,
                avg_fixed_cost=safe_mean(inst.fixed_costs),
                min_fixed_cost=min(inst.fixed_costs) if inst.fixed_costs else 0.0,
                max_fixed_cost=max(inst.fixed_costs) if inst.fixed_costs else 0.0,
                total_assignment_cost_entries=len(all_assignment_values),
            )
        )

        for facility_id, fixed_cost in enumerate(inst.fixed_costs):
            facility_rows.append(
                FacilityRow(
                    instance_id=inst.instance_id,
                    facility_id=facility_id,
                    fixed_cost=fixed_cost,
                )
            )

        for customer_id, row in enumerate(inst.costs):
            customer_rows.append(
                CustomerRow(
                    instance_id=inst.instance_id,
                    customer_id=customer_id,
                    min_assignment_cost=min(row) if row else 0.0,
                    max_assignment_cost=max(row) if row else 0.0,
                    avg_assignment_cost=safe_mean(row),
                )
            )

            for facility_id, assignment_cost in enumerate(row):
                assignment_rows.append(
                    AssignmentCostRow(
                        instance_id=inst.instance_id,
                        customer_id=customer_id,
                        facility_id=facility_id,
                        assignment_cost=assignment_cost,
                    )
                )

    return instance_rows, facility_rows, customer_rows, assignment_rows


def write_csv(rows: list[object], output_path: Path, *, fieldnames: tuple[str, ...]) -> None:
    write_dataclass_rows_csv(rows, output_path, fieldnames=fieldnames)


def write_json(rows: list[object], output_path: Path) -> None:
    write_dataclass_rows_json(rows, output_path)


def run_preprocessing() -> dict[str, object]:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    instance_files = discover_instance_files(RAW_DATA_DIR)
    if not instance_files:
        raise FileNotFoundError(
            f"No instance .txt files found in {RAW_DATA_DIR}. "
            "Run ingestion and ensure the shared OR-Library files are present."
        )

    parsed_instances: list[UFLPInstance] = []
    failed_files: list[str] = []

    for path in instance_files:
        try:
            parsed_instances.append(parse_orlib_uncap(path))
        except Exception as e:
            failed_files.append(f"{path.name}: {e}")

    if failed_files:
        print("\nSkipped files due to parse errors:")
        for msg in failed_files:
            print(f"- {msg}")

    if not parsed_instances:
        raise RuntimeError("No valid OR-Library instances could be parsed.")

    instance_rows, facility_rows, customer_rows, assignment_rows = build_rows(parsed_instances)

    write_csv(
        instance_rows,
        PROCESSED_DATA_DIR / "instances.csv",
        fieldnames=(
            "instance_id",
            "facility_count_m",
            "customer_count_n",
            "total_fixed_cost",
            "avg_fixed_cost",
            "min_fixed_cost",
            "max_fixed_cost",
            "total_assignment_cost_entries",
        ),
    )
    write_csv(
        facility_rows,
        PROCESSED_DATA_DIR / "facilities.csv",
        fieldnames=("instance_id", "facility_id", "fixed_cost"),
    )
    write_csv(
        customer_rows,
        PROCESSED_DATA_DIR / "customers.csv",
        fieldnames=(
            "instance_id",
            "customer_id",
            "min_assignment_cost",
            "max_assignment_cost",
            "avg_assignment_cost",
        ),
    )
    write_csv(
        assignment_rows,
        PROCESSED_DATA_DIR / "assignment_costs.csv",
        fieldnames=("instance_id", "customer_id", "facility_id", "assignment_cost"),
    )

    write_json(instance_rows, PROCESSED_DATA_DIR / "instances.json")
    write_json(facility_rows, PROCESSED_DATA_DIR / "facilities.json")
    write_json(customer_rows, PROCESSED_DATA_DIR / "customers.json")
    write_json(assignment_rows, PROCESSED_DATA_DIR / "assignment_costs.json")

    print("=== Milestone 3: Data Preprocessing Summary ===")
    print(f"Instances parsed: {len(parsed_instances)}")
    print(f"Instance rows written: {len(instance_rows)}")
    print(f"Facility rows written: {len(facility_rows)}")
    print(f"Customer rows written: {len(customer_rows)}")
    print(f"Assignment cost rows written: {len(assignment_rows)}")
    print("\nProcessed files written to:")
    print(f"- {PROCESSED_DATA_DIR / 'instances.csv'}")
    print(f"- {PROCESSED_DATA_DIR / 'facilities.csv'}")
    print(f"- {PROCESSED_DATA_DIR / 'customers.csv'}")
    print(f"- {PROCESSED_DATA_DIR / 'assignment_costs.csv'}")

    return {
        "raw_directory": str(RAW_DATA_DIR),
        "accepted_instance_count": len(instance_files),
        "parsed_instance_count": len(parsed_instances),
        "failed_instance_count": len(failed_files),
        "instances_csv": str(PROCESSED_DATA_DIR / "instances.csv"),
        "facilities_csv": str(PROCESSED_DATA_DIR / "facilities.csv"),
        "customers_csv": str(PROCESSED_DATA_DIR / "customers.csv"),
        "assignment_costs_csv": str(PROCESSED_DATA_DIR / "assignment_costs.csv"),
        "instance_rows": len(instance_rows),
        "facility_rows": len(facility_rows),
        "customer_rows": len(customer_rows),
        "assignment_cost_rows": len(assignment_rows),
    }


def main() -> None:
    run_preprocessing()


if __name__ == "__main__":
    main()
