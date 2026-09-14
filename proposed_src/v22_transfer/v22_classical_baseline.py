"""Classical complete solvers on the same evaluation instances as the learned arms.

The branching comparison in v22_cascade_search.py pits the learned heuristic against
other heuristics inside our own search loop, which measures branching quality but says
nothing about how the whole system stands against an off-the-shelf solver. Omitting that
comparison would let a within-loop improvement read as a solver-level one. This runs
CP-SAT and SCIP on exactly the instances the learned arms are evaluated on, single
threaded and under the same wall-clock budget, so the two are directly comparable.
"""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from ortools.sat.python import cp_model


def solve_cpsat(A, b, tl):
	mm, n = A.shape
	mo = cp_model.CpModel()
	xv = [mo.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(mm):
		idx = np.where(A[i] == 1)[0]
		mo.Add(sum(xv[j] for j in idx) == int(b[i]))
	s = cp_model.CpSolver()
	s.parameters.num_search_workers = 1
	s.parameters.max_time_in_seconds = tl
	t0 = time.time()
	st = s.Solve(mo)
	el = time.time() - t0
	return (s.StatusName(st) in ('OPTIMAL', 'FEASIBLE')), el


def solve_scip(A, b, tl):
	from pyscipopt import Model
	mm, n = A.shape
	mo = Model()
	mo.hideOutput()
	mo.setParam('limits/time', tl)
	mo.setParam('parallel/maxnthreads', 1)
	xv = [mo.addVar(vtype='B', name=f"x{j}") for j in range(n)]
	for i in range(mm):
		idx = np.where(A[i] == 1)[0]
		mo.addCons(sum(xv[j] for j in idx) == int(b[i]))
	t0 = time.time()
	mo.optimize()
	el = time.time() - t0
	return (mo.getStatus() in ('optimal', 'bestsollimit')), el


def one(task):
	path, solver, tl = task
	d = json.load(open(path))
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	ok, el = (solve_cpsat if solver == 'cpsat' else solve_scip)(A, b, tl)
	return dict(instance=Path(path).name, solved=bool(ok), time_sec=round(el, 4))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--solvers', nargs='+', default=['cpsat', 'scip'])
	ap.add_argument('--time_limit', type=float, default=600.0)
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	files = sorted(Path(a.data_dir).glob('*.json'))
	out = {}
	for sv in a.solvers:
		with Pool(a.jobs) as pool:
			rows = pool.map(one, [(str(f), sv, a.time_limit) for f in files])
		sol = [r for r in rows if r['solved']]
		ts = np.array([r['time_sec'] for r in sol])
		out[sv] = rows
		print(f"  {sv:6s} {len(sol)}/{len(rows)} solved   "
		      f"median={np.median(ts):8.3f}s  mean={ts.mean():8.3f}s  max={ts.max():8.3f}s"
		      if sol else f"  {sv:6s} 0/{len(rows)} solved", flush=True)
	json.dump(out, open(a.out, 'w'), indent=1)
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
