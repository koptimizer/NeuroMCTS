"""Pure CP baseline (OR-Tools classic constraint_solver, no CDCL/clause learning) --
tests whether classic finite-domain propagation + backtracking search competes
with CP-SAT on the same hard instances."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from ortools.constraint_solver import pywrapcp


def solve(A, b, time_limit_ms):
	m, n = A.shape
	solver = pywrapcp.Solver("classic_cp")
	x_vars = [solver.IntVar(0, 1, f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		solver.Add(solver.Sum([x_vars[j] for j in idx]) == int(b[i]))

	db = solver.Phase(x_vars, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
	solver.NewSearch(db, [solver.TimeLimit(time_limit_ms)])
	solved = solver.NextSolution()
	sol = np.array([x_vars[j].Value() for j in range(n)], dtype=np.int64) if solved else None
	solver.EndSearch()
	return solved, sol


def load_data(data_dir):
	files = sorted(Path(data_dir).glob("*.json"))
	data = []
	for f in files:
		d = json.load(open(f))
		A = np.array(d["A"], dtype=np.int64)
		b = np.array(d["b"], dtype=np.int64)
		gt_feasible = bool(d.get("feasible", True))
		data.append((A, b, gt_feasible))
	return data


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--time_limit', type=float, default=20.0)
	a = ap.parse_args()
	data = load_data(a.data_dir)

	tp = tn = fp_unknown = 0
	times = []
	for A, b, gt_feasible in data:
		t0 = time.perf_counter()
		found, sol = solve(A, b, int(a.time_limit * 1000))
		dt = time.perf_counter() - t0
		times.append(dt)
		if found:
			ok = bool(np.all(A.dot(sol) == b))
			if gt_feasible and ok:
				tp += 1
		else:
			fp_unknown += 1

	print(f"n={len(data)}  gt_feasible={sum(1 for _,_,g in data if g)}  "
	      f"solved(found valid sol)={tp}  not_found(timeout or proved infeasible, indistinguishable here)={fp_unknown}")
	print(f"avg_time={np.mean(times):.3f}s  total_time={np.sum(times):.1f}s")


if __name__ == '__main__':
	main()
