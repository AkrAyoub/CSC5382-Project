from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text_file(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def write_dataclass_rows_csv(rows: Sequence[object], output_path: Path, *, fieldnames: Sequence[str]) -> None:
    ensure_parent_dir(output_path)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_dataclass_rows_json(rows: Iterable[object], output_path: Path) -> None:
    payload = [asdict(row) for row in rows]
    write_json_file(output_path, payload)


def is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def group_count(rows: Iterable[Dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts
