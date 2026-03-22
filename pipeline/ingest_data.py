from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .common import write_dataclass_rows_csv, write_dataclass_rows_json
except ImportError:
    from common import write_dataclass_rows_csv, write_dataclass_rows_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
OPTIMA_FILE = RAW_DATA_DIR / "uncapopt.txt"
MANIFEST_FIELDNAMES = (
    "instance_id",
    "file_name",
    "file_path",
    "source",
    "file_size_bytes",
    "facility_count_m",
    "customer_count_n",
    "has_known_optimum",
    "ingestion_status",
)


@dataclass
class RawInstanceRecord:
    instance_id: str
    file_name: str
    file_path: str
    source: str
    file_size_bytes: int
    facility_count_m: int
    customer_count_n: int
    has_known_optimum: bool
    ingestion_status: str


def parse_instance_header(instance_path: Path) -> tuple[int, int]:
    """Read the leading `m n` pair from a raw instance file."""
    tokens = instance_path.read_text(encoding="utf-8").split()
    if len(tokens) < 2:
        raise ValueError(f"File {instance_path.name} does not contain enough tokens.")
    return int(tokens[0]), int(tokens[1])


def load_optimum_instance_names(optima_path: Path) -> set[str]:
    """Collect instance names listed in `uncapopt.txt`."""
    if not optima_path.exists():
        return set()

    instance_names: set[str] = set()
    for line in optima_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        instance_name = parts[0].replace(".txt", "")
        instance_names.add(instance_name)
    return instance_names


def discover_raw_instances(raw_dir: Path) -> list[Path]:
    """Return all candidate raw instance files, excluding the optima file."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    return [path for path in sorted(raw_dir.glob("*.txt")) if path.name.lower() != "uncapopt.txt"]


def build_manifest(raw_dir: Path, optima_file: Path) -> list[RawInstanceRecord]:
    instance_files = discover_raw_instances(raw_dir)
    known_optima = load_optimum_instance_names(optima_file)
    records: list[RawInstanceRecord] = []

    for instance_path in instance_files:
        instance_id = instance_path.stem

        try:
            m, n = parse_instance_header(instance_path)
            status = "ok"
        except (OSError, ValueError):
            m, n = -1, -1
            status = "parse_error"

        record = RawInstanceRecord(
            instance_id=instance_id,
            file_name=instance_path.name,
            file_path=str(instance_path.resolve()),
            source="OR-Library UFLP",
            file_size_bytes=instance_path.stat().st_size,
            facility_count_m=m,
            customer_count_n=n,
            has_known_optimum=instance_id in known_optima,
            ingestion_status=status,
        )
        records.append(record)

    return records


def write_manifest_csv(records: list[RawInstanceRecord], output_path: Path) -> None:
    write_dataclass_rows_csv(records, output_path, fieldnames=MANIFEST_FIELDNAMES)


def write_manifest_json(records: list[RawInstanceRecord], output_path: Path) -> None:
    write_dataclass_rows_json(records, output_path)


def print_summary(records: list[RawInstanceRecord], optima_file_exists: bool) -> None:
    total = len(records)
    ok_count = sum(r.ingestion_status == "ok" for r in records)
    parse_error_count = sum(r.ingestion_status != "ok" for r in records)
    known_opt_count = sum(r.has_known_optimum for r in records)

    print("=== Milestone 3: Raw Data Ingestion Summary ===")
    print(f"Raw directory: {RAW_DATA_DIR}")
    print(f"Optima file present: {optima_file_exists}")
    print(f"Total instance files discovered: {total}")
    print(f"Successfully parsed headers: {ok_count}")
    print(f"Header parse errors: {parse_error_count}")
    print(f"Instances with known optimum entries: {known_opt_count}")

    if records:
        print("\nSample records:")
        for record in records[:5]:
            print(
                f"- {record.instance_id}: "
                f"m={record.facility_count_m}, "
                f"n={record.customer_count_n}, "
                f"known_opt={record.has_known_optimum}, "
                f"status={record.ingestion_status}"
            )


def main() -> None:
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Expected raw data directory at {RAW_DATA_DIR}. "
            "Place OR-Library UFLP files in data/raw/."
        )

    optima_exists = OPTIMA_FILE.exists()
    records = build_manifest(RAW_DATA_DIR, OPTIMA_FILE)

    csv_path = INTERIM_DATA_DIR / "dataset_manifest.csv"
    json_path = INTERIM_DATA_DIR / "dataset_manifest.json"

    write_manifest_csv(records, csv_path)
    write_manifest_json(records, json_path)
    print_summary(records, optima_exists)

    print("\nManifest files written:")
    print(f"- {csv_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()
