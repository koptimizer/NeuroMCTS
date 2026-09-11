#!/usr/bin/env python3
"""
Generates CP-SAT-hard binary linear system instances by directly selecting
for a large LP-relaxation polytope (vertex_spread: mean pairwise L2 distance
among LP vertices found under random objective directions) -- the strongest,
most robust driver of CP-SAT search effort identified this cycle (Spearman
rho up to 0.87 with CP-SAT conflicts, ~120x wall-time gap between bottom and
top decile at 20x50).

An earlier version tried to reach this indirectly via a cheap proxy (forcing
b_i into the middle 25-75% of each row's achievable range). That was found
to be a near no-op: with x planted as a random n/2-of-n subset and A at
density 0.5, b already lands in that band 83-100% of the time by pure
concentration of measure (worse -- i.e. more of a no-op -- at larger sizes).
This version selects directly on vertex_spread instead: for a fixed A, K
candidate plantings of x are drawn and the one with the largest vertex_spread
is kept (K=20 empirically captures most of the achievable gain per unit
cost; mean best-of-20 vertex_spread is ~35-78% above a single random draw
at 10x25/20x50, with diminishing returns beyond K=20-25).

No uniqueness check is applied to the planted solution (dropped by design;
see CLAUDE.md and this cycle's finding that uniqueness-filtering, as the
legacy generator did, biases toward *easier*, small-polytope instances).

Infeasible instances: start from a hardness-selected feasible pair, perturb
1-3 rows of b by +-1 (clipped to [0, rowsum]), and confirm INFEASIBLE with
CP-SAT (bounded time; retries with a fresh perturbation on timeout/failure).

Usage:
  python generate_hard_instances.py --size 10x25 --train_n 10000 --test_n 1000 --jobs 16 --K 20
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from ortools.sat.python import cp_model
from pyscipopt import Model as ScipModel

FEASIBLE_RATIO = 0.8


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


def vertex_spread(A, b, rng, k=4):
	n = A.shape[1]
	verts = []
	for _ in range(k):
		c = rng.normal(0, 1, size=n)
		v = solve_lp_obj(A, b, c)
		if v is not None:
			verts.append(v)
	if len(verts) < 2:
		return 0.0
	uniq = []
	for v in verts:
		if not any(np.linalg.norm(v - u) < 1e-4 for u in uniq):
			uniq.append(v)
	if len(uniq) < 2:
		return 0.0
	dists = [np.linalg.norm(uniq[i] - uniq[j]) for i in range(len(uniq)) for j in range(i + 1, len(uniq))]
	return float(np.mean(dists))


def gen_hard_feasible(m, n, rng, K=20, k_dirs=4, max_A_retry=50):
	for _ in range(max_A_retry):
		A = (rng.random((m, n)) < 0.5).astype(np.int8)
		if A.sum(axis=1).min() < 2 or A.sum(axis=0).min() < 1:
			continue
		best_vs = -1.0
		best = None
		for _ in range(K):
			x = np.zeros(n, dtype=np.int8)
			x[rng.choice(n, n // 2, replace=False)] = 1
			b = A.astype(np.int64).dot(x.astype(np.int64))
			vs = vertex_spread(A, b, rng, k=k_dirs)
			if vs > best_vs:
				best_vs = vs
				best = (A, b, x)
		if best is not None:
			return best[0], best[1], best[2], best_vs
	raise RuntimeError("gen_hard_feasible: exhausted retries")


def cpsat_infeasible(A, b, time_limit):
	m, n = A.shape
	model = cp_model.CpModel()
	xv = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(xv[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.max_time_in_seconds = time_limit
	status = solver.Solve(model)
	if status == cp_model.INFEASIBLE:
		return True
	if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		return False
	return None  # unresolved within time_limit


def gen_hard_infeasible(m, n, rng, time_limit, max_perturb_attempts=6, max_base_attempts=15):
	# Deliberately does NOT reuse the caller's (large) K: a maximally hard
	# (large-polytope) feasible base absorbs small b-perturbations without
	# becoming infeasible (verified empirically -- 5-row +-3 perturbations
	# barely dent a K=20 base's LP feasibility), making construction
	# intractable. Infeasible hardness is a different mechanism (proof
	# difficulty, not solution-space size), so the base here uses K=1
	# (the original unselected planting), matching the perturbation recipe
	# already validated to produce hard-to-prove infeasible instances.
	for _ in range(max_base_attempts):
		A, b_feas, x, _ = gen_hard_feasible(m, n, rng, K=1)
		row_sum = A.sum(axis=1)
		for _ in range(max_perturb_attempts):
			k = int(rng.integers(1, 4))
			rows = rng.choice(m, k, replace=False)
			b_inf = b_feas.copy()
			b_inf[rows] += rng.choice([-1, 1], size=k)
			b_inf = np.clip(b_inf, 0, row_sum)
			if np.array_equal(b_inf, b_feas):
				continue
			result = cpsat_infeasible(A, b_inf, time_limit)
			if result is True:
				return A, b_inf
	raise RuntimeError("gen_hard_infeasible: exhausted retries")


def make_one(task):
	idx, m, n, seed, is_feasible, K, time_limit = task
	rng = np.random.default_rng(seed)
	if is_feasible:
		A, b, x, vs = gen_hard_feasible(m, n, rng, K=K)
		return idx, dict(A=A.tolist(), b=b.tolist(), x=x.tolist(), feasible=True)
	else:
		A, b = gen_hard_infeasible(m, n, rng, time_limit=time_limit)
		return idx, dict(A=A.tolist(), b=b.tolist(), x=None, feasible=False)


def build_split(m, n, count, seed0, out_dir, prefix, jobs, K, time_limit):
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	n_feasible = int(round(count * FEASIBLE_RATIO))
	labels = [True] * n_feasible + [False] * (count - n_feasible)
	rng_shuffle = np.random.default_rng(seed0 + 999_999)
	rng_shuffle.shuffle(labels)
	tasks = [(i, m, n, seed0 + i, labels[i], K, time_limit) for i in range(count)]
	t0 = time.time()
	done = 0
	with Pool(jobs) as pool:
		for i, rec in pool.imap_unordered(make_one, tasks, chunksize=1):
			path = out_dir / f"{prefix}-{i:05d}.json"
			with open(path, 'w') as f:
				json.dump(rec, f)
			done += 1
			if done % 100 == 0:
				print(f"  [{prefix}] {done}/{count} ({time.time()-t0:.0f}s)", flush=True)
	print(f"[{prefix}] done: {count} instances, {time.time()-t0:.0f}s total", flush=True)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', required=True)
	ap.add_argument('--train_n', type=int, default=10000)
	ap.add_argument('--test_n', type=int, default=1000)
	ap.add_argument('--jobs', type=int, default=16)
	ap.add_argument('--K', type=int, default=20)
	ap.add_argument('--time_limit', type=float, default=15.0)
	ap.add_argument('--out_root', default='../../instances')
	a = ap.parse_args()
	m, n = (int(v) for v in a.size.split('x'))

	train_dir = f"{a.out_root}/train_instances_hard_{m}x{n}_{a.train_n}"
	test_dir = f"{a.out_root}/test_instances_hard_{m}x{n}_{a.test_n}"
	prefix_train = f"train_instances_hard_{m}x{n}_{a.train_n}"
	prefix_test = f"test_instances_hard_{m}x{n}_{a.test_n}"

	build_split(m, n, a.train_n, seed0=10_000_000, out_dir=train_dir, prefix=prefix_train,
	            jobs=a.jobs, K=a.K, time_limit=a.time_limit)
	build_split(m, n, a.test_n, seed0=20_000_000, out_dir=test_dir, prefix=prefix_test,
	            jobs=a.jobs, K=a.K, time_limit=a.time_limit)


if __name__ == '__main__':
	main()
