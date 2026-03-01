from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ortools.linear_solver import pywraplp


@dataclass(frozen=True)
class UFLPInstance:
    name: str
    m: int
    n: int
    fixed_costs: List[float]
    costs: List[List[float]]  # costs[j][i]


def parse_orlib_uncap(instance_path: str) -> UFLPInstance:
    tokens = Path(instance_path).read_text().split()
    it = iter(tokens)

    m = int(next(it))
    n = int(next(it))

    fixed_costs: List[float] = []
    for _ in range(m):
        _capacity = float(next(it))  # must parse but ignore
        f_i = float(next(it))
        fixed_costs.append(f_i)

    costs: List[List[float]] = []
    for _ in range(n):
        _demand = float(next(it))  # must parse but ignore
        row = [float(next(it)) for _ in range(m)]
        costs.append(row)

    name = Path(instance_path).stem
    return UFLPInstance(name=name, m=m, n=n, fixed_costs=fixed_costs, costs=costs)


def parse_uncapopt(opt_path: str) -> Dict[str, float]:
    lines = Path(opt_path).read_text().strip().splitlines()
    out: Dict[str, float] = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = float(parts[1])
    return out


@dataclass(frozen=True)
class BaselineResult:
    instance: str
    objective: float
    best_known: Optional[float]
    gap_percent: Optional[float]
    runtime_s: Optional[float]
    open_facilities: List[int]
    assignments: List[int]

    # Backward-compatible alias
    @property
    def gap_pct(self) -> Optional[float]:
        return self.gap_percent


def solve_uflp_cbc(
    inst: UFLPInstance, time_limit_s: float | None = None
) -> Tuple[float, float, List[int], List[int]]:
    import time

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver not available in this OR-Tools build.")

    if time_limit_s is not None:
        solver.SetTimeLimit(int(time_limit_s * 1000))

    m, n = inst.m, inst.n

    y = [solver.BoolVar(f"y[{i}]") for i in range(m)]
    x = [[solver.BoolVar(f"x[{j},{i}]") for i in range(m)] for j in range(n)]

    # Each customer assigned exactly once
    for j in range(n):
        solver.Add(sum(x[j][i] for i in range(m)) == 1)

    # Assignment implies facility open
    for j in range(n):
        for i in range(m):
            solver.Add(x[j][i] <= y[i])

    # Objective: fixed + assignment costs (no demand multiplication)
    obj = solver.Objective()
    for i in range(m):
        obj.SetCoefficient(y[i], inst.fixed_costs[i])
    for j in range(n):
        for i in range(m):
            obj.SetCoefficient(x[j][i], inst.costs[j][i])
    obj.SetMinimization()

    t0 = time.time()
    status = solver.Solve()
    t1 = time.time()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Solver failed. Status={status}")

    objective_value = float(obj.Value())

    open_facilities: List[int] = [i for i in range(m) if y[i].solution_value() > 0.5]

    assignments: List[int] = [-1] * n
    for j in range(n):
        chosen = -1
        for i in range(m):
            if x[j][i].solution_value() > 0.5:
                chosen = i
                break
        if chosen < 0:
            raise RuntimeError(f"No assignment chosen for customer j={j}")
        assignments[j] = chosen

    return objective_value, (t1 - t0), open_facilities, assignments


def run_baseline(
    instance_path: str,
    opt_path: str | None = None,
    time_limit_s: float | None = None,
) -> BaselineResult:
    inst = parse_orlib_uncap(instance_path)

    best_known = None
    if opt_path is not None and Path(opt_path).exists():
        best_map = parse_uncapopt(opt_path)
        best_known = best_map.get(inst.name)

    obj, rt, opens, assigns = solve_uflp_cbc(inst, time_limit_s=time_limit_s)

    gap = None
    if best_known is not None:
        denom = max(1.0, abs(best_known))
        gap = (obj - best_known) / denom * 100.0

    return BaselineResult(
        instance=inst.name,
        objective=obj,
        best_known=best_known,
        gap_percent=gap,
        runtime_s=rt,
        open_facilities=opens,
        assignments=assigns,
    )