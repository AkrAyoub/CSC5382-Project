from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

from ortools.linear_solver import pywraplp

from m4_model_dev.paths import DATA_RAW_DIR


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class BenchmarkInstance:
    instance_id: str
    instance_path: Path
    facility_count_m: int
    customer_count_n: int
    fixed_costs: list[float]
    costs: list[list[float]]


@dataclass(frozen=True)
class ReferenceSolveResult:
    instance_id: str
    objective: float
    best_known: float | None
    gap_percent: float | None
    runtime_s: float
    open_facilities: list[int]
    assignments: list[int]


def extract_numeric_tokens(line: str) -> list[float]:
    return [float(token) for token in NUMBER_PATTERN.findall(line)]


def discover_raw_instances(raw_dir: Path | None = None) -> list[Path]:
    raw_dir = raw_dir or DATA_RAW_DIR
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw benchmark directory not found: {raw_dir}")
    return [path for path in sorted(raw_dir.glob("*.txt")) if path.name.lower() != "uncapopt.txt"]


def parse_uncapopt(optima_path: Path) -> dict[str, float]:
    if not optima_path.exists():
        return {}

    out: dict[str, float] = {}
    lines = [line.strip() for line in optima_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0].replace(".txt", "")] = float(parts[1])
    return out


def parse_orlib_uncap(instance_path: Path) -> BenchmarkInstance:
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
        values = extract_numeric_tokens(lines[line_idx])
        line_idx += 1
        if not values:
            raise ValueError(f"Missing facility values for facility {facility_idx} in {instance_path.name}")
        fixed_costs.append(values[-1])

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

    return BenchmarkInstance(
        instance_id=instance_path.stem,
        instance_path=instance_path,
        facility_count_m=m,
        customer_count_n=n,
        fixed_costs=fixed_costs,
        costs=costs,
    )


def solve_reference_cbc(instance: BenchmarkInstance, best_known: float | None = None) -> ReferenceSolveResult:
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver not available in this OR-Tools build.")

    m = instance.facility_count_m
    n = instance.customer_count_n
    y = [solver.BoolVar(f"y[{i}]") for i in range(m)]
    x = [[solver.BoolVar(f"x[{j},{i}]") for i in range(m)] for j in range(n)]

    for j in range(n):
        solver.Add(sum(x[j][i] for i in range(m)) == 1)

    for j in range(n):
        for i in range(m):
            solver.Add(x[j][i] <= y[i])

    obj = solver.Objective()
    for i in range(m):
        obj.SetCoefficient(y[i], instance.fixed_costs[i])
    for j in range(n):
        for i in range(m):
            obj.SetCoefficient(x[j][i], instance.costs[j][i])
    obj.SetMinimization()

    t0 = time.time()
    status = solver.Solve()
    runtime_s = time.time() - t0

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Reference solver failed on {instance.instance_id}. Status={status}")

    objective = float(obj.Value())
    open_facilities = [i for i in range(m) if y[i].solution_value() > 0.5]

    assignments: list[int] = [-1] * n
    for j in range(n):
        chosen = -1
        for i in range(m):
            if x[j][i].solution_value() > 0.5:
                chosen = i
                break
        if chosen < 0:
            raise RuntimeError(f"No assignment chosen for customer {j} in {instance.instance_id}")
        assignments[j] = chosen

    gap_percent = None
    if best_known is not None:
        denom = max(1.0, abs(best_known))
        gap_percent = (objective - best_known) / denom * 100.0

    return ReferenceSolveResult(
        instance_id=instance.instance_id,
        objective=objective,
        best_known=best_known,
        gap_percent=gap_percent,
        runtime_s=runtime_s,
        open_facilities=open_facilities,
        assignments=assignments,
    )


def build_reference_solver_module_source() -> str:
    return """from pathlib import Path
import re
from ortools.linear_solver import pywraplp

NUMBER_PATTERN = re.compile(r"[-+]?(?:\\d+\\.\\d*|\\d+|\\.\\d+)(?:[eE][-+]?\\d+)?")


def _numbers(line: str) -> list[float]:
    return [float(token) for token in NUMBER_PATTERN.findall(line)]


def solve(instance_path: str) -> dict:
    lines = [line.strip() for line in Path(instance_path).read_text().splitlines() if line.strip()]
    header = _numbers(lines[0])
    m = int(header[0])
    n = int(header[1])
    line_idx = 1

    fixed = []
    for _ in range(m):
        values = _numbers(lines[line_idx])
        line_idx += 1
        fixed.append(values[-1])

    cost = []
    for _ in range(n):
        _demand = _numbers(lines[line_idx])[0]
        line_idx += 1
        row = []
        while len(row) < m:
            row.extend(_numbers(lines[line_idx]))
            line_idx += 1
        cost.append(row[:m])

    solver = pywraplp.Solver.CreateSolver("CBC")
    y = [solver.BoolVar(f"y[{i}]") for i in range(m)]
    x = [[solver.BoolVar(f"x[{j},{i}]") for i in range(m)] for j in range(n)]

    for j in range(n):
        solver.Add(sum(x[j][i] for i in range(m)) == 1)
    for j in range(n):
        for i in range(m):
            solver.Add(x[j][i] <= y[i])

    obj = solver.Objective()
    for i in range(m):
        obj.SetCoefficient(y[i], fixed[i])
    for j in range(n):
        for i in range(m):
            obj.SetCoefficient(x[j][i], cost[j][i])
    obj.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("solver failed")

    open_facilities = [i for i in range(m) if y[i].solution_value() > 0.5]
    assignments = []
    for j in range(n):
        chosen = -1
        for i in range(m):
            if x[j][i].solution_value() > 0.5:
                chosen = i
                break
        if chosen < 0:
            raise RuntimeError("invalid assignment")
        assignments.append(chosen)

    return {
        "objective": solver.Objective().Value(),
        "open_facilities": open_facilities,
        "assignments": assignments,
    }
"""
