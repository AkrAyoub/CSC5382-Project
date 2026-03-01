from __future__ import annotations

import argparse
from pathlib import Path

from baseline_solver import run_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="Path like data/raw/cap71.txt")
    ap.add_argument("--optfile", default="data/raw/uncapopt.txt")
    ap.add_argument("--time_limit_s", type=float, default=None)
    args = ap.parse_args()

    res = run_baseline(args.instance, args.optfile, time_limit_s=args.time_limit_s)

    print(f"Loaded instance: {Path(args.instance).name}")
    print(f"Solver objective: {res.objective:.3f}")
    if res.best_known is not None:
        print(f"Best known (uncapopt): {res.best_known:.3f}")
        print(f"Gap vs best known: {res.gap_percent:.6f}%")
    else:
        print("No best-known value found for this instance name in uncapopt.txt.")
    print(f"Runtime (s): {res.runtime_s:.3f}")


if __name__ == "__main__":
    main()