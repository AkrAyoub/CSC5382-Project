import math
from pathlib import Path
from typing import Dict, List, Tuple

from ortools.linear_solver import pywraplp


def parse_orlib_uncap(path: str) -> Tuple[int, int, List[float], List[List[float]]]:
    """
    Parses OR-Library uncapacitated warehouse location instances (uncapinfo).

    Format:
      m n
      (capacity_i fixed_cost_i) for i=1..m
      for each customer j=1..n:
          demand_j
          costs c_ij for i=1..m (may span multiple lines)

    Note: capacity_i and demand_j are present but ignored for uncapacitated model.
    """
    tokens = Path(path).read_text().split()
    it = iter(tokens)

    m = int(next(it))
    n = int(next(it))

    fixed_costs: List[float] = []
    for _ in range(m):
        _capacity = float(next(it))  # ignored
        f_i = float(next(it))
        fixed_costs.append(f_i)

    costs: List[List[float]] = []
    for _ in range(n):
        _demand = float(next(it))  # ignored
        row = [float(next(it)) for _ in range(m)]
        costs.append(row)

    return m, n, fixed_costs, costs


def parse_uncapopt(path: str) -> Dict[str, float]:
    """
    Parses uncapopt.txt:
      Data file   Optimal solution value
      cap71       932615.750
      ...
    """
    lines = Path(path).read_text().strip().splitlines()
    out: Dict[str, float] = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = float(parts[1])
    return out


def solve_uflp_cbc(m: int, n: int, fixed_costs: List[float], costs: List[List[float]]) -> float:
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver not available in this OR-Tools build.")

    # Decision variables:
    # y[i] = 1 if facility i opened
    y = [solver.BoolVar(f"y[{i}]") for i in range(m)]

    # x[j][i] = 1 if customer j assigned to facility i
    x = [[solver.BoolVar(f"x[{j},{i}]") for i in range(m)] for j in range(n)]

    # Constraints:
    # Each customer assigned exactly once
    for j in range(n):
        solver.Add(sum(x[j][i] for i in range(m)) == 1)

    # Can assign only to open facilities
    for j in range(n):
        for i in range(m):
            solver.Add(x[j][i] <= y[i])

    # Objective: min sum f_i y_i + sum c_ij x_ij
    obj = solver.Objective()
    for i in range(m):
        obj.SetCoefficient(y[i], fixed_costs[i])
    for j in range(n):
        for i in range(m):
            obj.SetCoefficient(x[j][i], costs[j][i])
    obj.SetMinimization()

    # solver.SetTimeLimit(60_000)  # 60 seconds

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Solver failed. Status={status}")

    return obj.Value()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="Path to instance file, e.g., cap71.txt")
    ap.add_argument("--optfile", default="uncapopt.txt", help="Path to uncapopt.txt")
    args = ap.parse_args()

    inst_path = Path(args.instance)
    opt_path = Path(args.optfile)

    m, n, fixed_costs, costs = parse_orlib_uncap(str(inst_path))
    print(f"Loaded instance: {inst_path.name} (m={m}, n={n})")

    best_known = None
    if opt_path.exists():
        opt_map = parse_uncapopt(str(opt_path))
        best_known = opt_map.get(inst_path.stem, None)

    val = solve_uflp_cbc(m, n, fixed_costs, costs)

    # OR-Library values are shown with 3 decimals; print similarly.
    print(f"Solver objective: {val:.3f}")

    if best_known is not None:
        gap = (val - best_known) / max(1.0, abs(best_known)) * 100.0
        print(f"Best known (uncapopt): {best_known:.3f}")
        print(f"Gap vs best known: {gap:.6f}%")
    else:
        print("No best-known value found for this instance name in uncapopt.txt.")


if __name__ == "__main__":
    main()
