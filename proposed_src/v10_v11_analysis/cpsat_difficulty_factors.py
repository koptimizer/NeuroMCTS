# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

#!/usr/bin/env python3
"""
Tests whether integer-solution multiplicity and/or LP-relaxation polytope
"size" predict CP-SAT search effort, on freshly generated feasible instances
(gen_planted_instance, no uniqueness filtering -- solution count varies
naturally). For each instance, measures:
  n_solutions        - number of distinct 0/1 solutions (CP-SAT enumeration, capped)
  lp_sum_spread      - max-sum LP objective minus min-sum LP objective
  lp_frac            - fractionality of the plain feasibility LP point
  n_vertex_distinct  - distinct LP vertices found under 5 random objectives
  vertex_spread      - mean pairwise L2 distance among those vertices
  n_dup_col_pairs    - number of exactly-duplicate column pairs in A (symmetry proxy)
and CP-SAT's own search effort on the feasibility problem:
  wall_time, num_branches, num_conflicts

Usage:
  python cpsat_difficulty_factors.py --size 10x25 --n 400 --out ../../runs/cpsat_difficulty_10x25.jsonl --jobs 8
"""
import argparse
import json
import time
import numpy as np
from multiprocessing import Pool
from ortools.sat.python import cp_model
from pyscipopt import Model as ScipModel

import sys
sys.path.insert(0, '.')
from LPneuroBLS_v7 import gen_planted_instance


class Counter(cp_model.CpSolverSolutionCallback):
	def __init__(self, limit):
		super().__init__()
		self.count = 0
		self.limit = limit

	def on_solution_callback(self):
		self.count += 1
		if self.count >= self.limit:
			self.StopSearch()


def count_solutions(A, b, cap=30):
	m, n = A.shape
	model = cp_model.CpModel()
	xv = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(xv[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.enumerate_all_solutions = True
	solver.parameters.num_search_workers = 1
	solver.parameters.max_time_in_seconds = 15.0
	cb = Counter(cap)
	status = solver.Solve(model, cb)
	capped = cb.count >= cap
	return cb.count, capped


def solve_lp_obj(A, b, c):
	m, n = A.shape
	model = ScipModel(); model.hideOutput()
	xv = [model.addVar(vtype="C", lb=0.0, ub=1.0, name=f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.addCons(sum(xv[j] for j in idx) == int(b[i]))
	model.setObjective(sum(float(c[j]) * xv[j] for j in range(n)), "minimize")
	model.optimize()
	if model.getStatus() != "optimal":
		return None
	return np.array([model.getVal(v) for v in xv], dtype=np.float64)


def lp_features(A, b, rng):
	n = A.shape[1]
	x_min = solve_lp_obj(A, b, np.ones(n))
	x_max = solve_lp_obj(A, b, -np.ones(n))
	spread = float(x_max.sum() - x_min.sum()) if (x_min is not None and x_max is not None) else None
	lp_frac = float(np.abs(x_min - np.rint(x_min)).mean()) if x_min is not None else None
	verts = []
	for _ in range(5):
		c = rng.normal(0, 1, size=n)
		v = solve_lp_obj(A, b, c)
		if v is not None:
			verts.append(v)
	n_distinct = 0
	vertex_spread = 0.0
	if verts:
		uniq = []
		for v in verts:
			if not any(np.linalg.norm(v - u) < 1e-4 for u in uniq):
				uniq.append(v)
		n_distinct = len(uniq)
		if len(uniq) > 1:
			dists = [np.linalg.norm(uniq[i] - uniq[j]) for i in range(len(uniq)) for j in range(i + 1, len(uniq))]
			vertex_spread = float(np.mean(dists))
	return spread, lp_frac, n_distinct, vertex_spread


def cpsat_effort(A, b):
	m, n = A.shape
	model = cp_model.CpModel()
	xv = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(xv[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.log_search_progress = False
	solver.parameters.max_time_in_seconds = 30.0
	t0 = time.perf_counter()
	status = solver.Solve(model)
	wall = time.perf_counter() - t0
	return wall, solver.NumBranches(), solver.NumConflicts(), solver.WallTime()


def dup_col_pairs(A):
	n = A.shape[1]
	cols = [tuple(A[:, j].tolist()) for j in range(n)]
	from collections import Counter as PyCounter
	cnt = PyCounter(cols)
	return sum(k * (k - 1) // 2 for k in cnt.values())


def process(task):
	idx, m, n, seed = task
	rng = np.random.default_rng(seed)
	A, b, x = gen_planted_instance(m, n, rng)
	A = A.astype(np.int64); b = b.astype(np.int64)
	n_sol, capped = count_solutions(A, b, cap=30)
	spread, lp_frac, n_distinct, vertex_spread = lp_features(A, b, rng)
	wall, branches, conflicts, cpsat_wall = cpsat_effort(A, b)
	ndup = dup_col_pairs(A)
	return dict(idx=idx, m=m, n=n, n_solutions=n_sol, n_solutions_capped=capped,
	            lp_sum_spread=spread, lp_frac=lp_frac, n_vertex_distinct=n_distinct,
	            vertex_spread=vertex_spread, n_dup_col_pairs=ndup,
	            cpsat_wall=wall, cpsat_branches=branches, cpsat_conflicts=conflicts)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', default='10x25')
	ap.add_argument('--n', type=int, default=400)
	ap.add_argument('--out', required=True)
	ap.add_argument('--jobs', type=int, default=8)
	a = ap.parse_args()
	m, n = (int(v) for v in a.size.split('x'))
	tasks = [(i, m, n, 1_000_000 + i) for i in range(a.n)]
	t0 = time.time()
	with Pool(a.jobs) as pool, open(a.out, 'w') as f:
		done = 0
		for rec in pool.imap_unordered(process, tasks, chunksize=2):
			f.write(json.dumps(rec) + '\n'); f.flush()
			done += 1
			if done % 50 == 0:
				print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
	print(f"done: {a.n} instances, {time.time()-t0:.0f}s total")


if __name__ == '__main__':
	main()
