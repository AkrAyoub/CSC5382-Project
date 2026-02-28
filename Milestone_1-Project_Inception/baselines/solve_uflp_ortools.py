#!/usr/bin/env python3
"""Deterministic UFLP baseline using OR-Tools CBC solver.

Usage:
  python baselines/solve_uflp_ortools.py --instance data/raw/cap71.txt --opt data/raw/uncapopt.txt
  python baselines/solve_uflp_ortools.py --dir data/raw --opt data/raw/uncapopt.txt

The script parses OR-Library instance files, builds the canonical UFLP MILP,
solves with CBC, and reports objective, best-known value and gap.
"""
from __future__ import annotations
import argparse
import glob
import os
import time
from typing import Dict, List, Tuple

try:
    from ortools.linear_solver import pywraplp
except Exception:
    pywraplp = None


def parse_orlib_instance(path: str) -> Tuple[int, int, List[float], List[List[float]]]:
    """Parse an OR-Library UFLP-like instance file.

    Returns: m, n, f (list of fixed costs length m), c (m x n cost matrix transposed as list of lists)
    """
    with open(path, "r") as f:
        tokens = f.read().split()
    if len(tokens) < 2:
        raise ValueError(f"Unexpected instance format: {path}")
    idx = 0
    m = int(float(tokens[idx])); idx += 1
    n = int(float(tokens[idx])); idx += 1

    f_costs: List[float] = []
    # OR-Library capacitated variants often list (capacity, fixed) pairs per facility
    for _ in range(m):
        cap = float(tokens[idx]); idx += 1
        fixed = float(tokens[idx]); idx += 1
        f_costs.append(fixed)

    # For each customer: demand followed by m allocation costs
    c = [[0.0 for _ in range(n)] for _ in range(m)]
    for j in range(n):
        demand = float(tokens[idx]); idx += 1
        for i in range(m):
            c[i][j] = float(tokens[idx]); idx += 1

    return m, n, f_costs, c


def parse_uncapopt(path: str) -> Dict[str, float]:
    """Parse uncapopt.txt mapping instance name -> best known objective.

    Expected formats (robust):
      cap71 932615.750
      cap71 : 932615.750
    """
    mapping: Dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(':', ' ').split() if p]
            if len(parts) >= 2:
                name = parts[0]
                try:
                    val = float(parts[1])
                except ValueError:
                    continue
                mapping[name] = val
    return mapping


def solve_uflp(m: int, n: int, f_costs: List[float], c: List[List[float]], time_limit: int = 30):
    if pywraplp is None:
        raise RuntimeError("OR-Tools not available. Install with: pip install ortools")

    solver = pywraplp.Solver.CreateSolver('CBC')
    if solver is None:
        raise RuntimeError('CBC solver unavailable')

    # binary variables
    y = [solver.IntVar(0, 1, f'y_{i}') for i in range(m)]
    x = [[solver.IntVar(0, 1, f'x_{i}_{j}') for j in range(n)] for i in range(m)]

    # Each customer assigned exactly once
    for j in range(n):
        solver.Add(sum(x[i][j] for i in range(m)) == 1)

    # Assignment only to open facilities
    for i in range(m):
        for j in range(n):
            solver.Add(x[i][j] <= y[i])

    # Objective
    objective = solver.Objective()
    for i in range(m):
        objective.SetCoefficient(y[i], f_costs[i])
        for j in range(n):
            objective.SetCoefficient(x[i][j], c[i][j])
    objective.SetMinimization()

    # optional time limit
    try:
        solver.SetTimeLimit(int(time_limit * 1000))
    except Exception:
        pass

    start = time.time()
    status = solver.Solve()
    elapsed = time.time() - start

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None

    obj = objective.Value()
    return dict(objective=obj, time=elapsed, status=status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, help='Single OR-Library instance file path')
    parser.add_argument('--dir', type=str, help='Directory with instance files (cap*.txt)')
    parser.add_argument('--opt', type=str, default='data/raw/uncapopt.txt', help='uncapopt.txt file')
    parser.add_argument('--time-limit', type=int, default=30, help='Solver time limit in seconds')
    args = parser.parse_args()

    opt_map = parse_uncapopt(args.opt) if args.opt and os.path.exists(args.opt) else {}

    instances = []
    if args.instance:
        instances = [args.instance]
    elif args.dir:
        instances = sorted(glob.glob(os.path.join(args.dir, 'cap*.txt')))
    else:
        parser.error('Either --instance or --dir must be provided')

    results = []
    for inst in instances:
        name = os.path.splitext(os.path.basename(inst))[0]
        print(f'Processing {name} ...')
        m, n, f_costs, c = parse_orlib_instance(inst)
        res = solve_uflp(m, n, f_costs, c, time_limit=args.time_limit)
        if res is None:
            print(f'  No solution (status) for {name}')
            continue
        best_known = opt_map.get(name)
        gap = None
        if best_known is not None and best_known != 0:
            gap = 100.0 * (res['objective'] - best_known) / best_known
        print(f"  Obj={res['objective']:.3f}, BestKnown={best_known}, Gap%={gap}, Time={res['time']:.3f}s")
        results.append((name, m, n, res['objective'], best_known, gap, res['time']))

    # Simple CSV output
    if results:
        out = 'baseline_results.csv'
        with open(out, 'w') as f:
            f.write('Instance,m,n,SolverObjective,BestKnown,GapPercent,TimeSec\n')
            for r in results:
                f.write(','.join(str(x) if x is not None else '' for x in r) + '\n')
        print(f'Wrote results to {out}')


if __name__ == '__main__':
    main()
