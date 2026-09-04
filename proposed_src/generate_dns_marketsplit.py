#!/usr/bin/env python3
"""
Market-split-style DNS instance generator (hard family).

Feasible instances: plant x with exactly n/2 ones and set b = Ax, so every
b_i sits near rowsum/2 — the profile of the Cornuejols-Dawande market split
family — and b leaks no information about the planted density.

Infeasible instances: perturb k rows of a planted b by +/-1 (the DNS noise
semantics) and keep the instance only if CP-SAT proves infeasibility within
the time limit, so labels are certified.

Usage:
  python generate_dns_marketsplit.py --out_root ../instances --sizes 10x25 20x50 40x100
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from ortools.sat.python import cp_model


def gen_planted(m, n, rng):
	# Dense Bernoulli(0.5) matrix with no empty row/column, plus a planted
	# half-density solution so that b_i ~ rowsum/2 hides the solution profile
	while True:
		A = (rng.random((m, n)) < 0.5).astype(np.int8)
		if A.sum(axis=1).min() >= 2 and A.sum(axis=0).min() >= 1:
			break
	x = np.zeros(n, dtype=np.int8)
	x[rng.choice(n, n // 2, replace=False)] = 1
	b = A.astype(np.int32).dot(x.astype(np.int32))
	return A, b, x


def cpsat_status(A, b, time_limit):
	# Returns 'feasible' | 'infeasible' | 'unknown' with a certified solve
	m, n = A.shape
	model = cp_model.CpModel()
	xv = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(xv[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 4
	solver.parameters.max_time_in_seconds = time_limit
	st = solver.Solve(model)
	if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		return 'feasible'
	if st == cp_model.INFEASIBLE:
		return 'infeasible'
	return 'unknown'


def gen_infeasible(m, n, rng, time_limit, max_tries=50):
	# DNS noise model: +/-1 corruption on k rows of a planted b, certified
	# infeasible by CP-SAT; returns None if no certified instance was found
	for _ in range(max_tries):
		A, b, _ = gen_planted(m, n, rng)
		k = int(rng.integers(1, 4))
		rows = rng.choice(m, k, replace=False)
		b2 = b.copy()
		b2[rows] += rng.choice([-1, 1], size=k)
		rowsum = A.sum(axis=1)
		b2 = np.clip(b2, 0, rowsum)
		if np.array_equal(b2, b):
			continue
		if cpsat_status(A, b2, time_limit) == 'infeasible':
			return A, b2
	return None, None


def write_instance(path, A, b, x, feasible):
	with open(path, 'w') as f:
		json.dump({
			'A': A.astype(int).tolist(),
			'b': b.astype(int).tolist(),
			'x': x.astype(int).tolist() if x is not None else None,
			'feasible': bool(feasible),
		}, f)


def gen_split(out_dir, m, n, n_feas, n_infeas, seed, time_limit):
	out_dir.mkdir(parents=True, exist_ok=True)
	rng = np.random.default_rng(seed)
	total = n_feas + n_infeas
	t0 = time.time()
	discarded = 0
	for i in range(n_feas):
		A, b, x = gen_planted(m, n, rng)
		write_instance(out_dir / f"{out_dir.name}-{i:05d}.json", A, b, x, True)
	for i in range(n_feas, total):
		A, b = gen_infeasible(m, n, rng, time_limit)
		while A is None:
			discarded += 1
			A, b = gen_infeasible(m, n, rng, time_limit)
		write_instance(out_dir / f"{out_dir.name}-{i:05d}.json", A, b, None, False)
		if (i - n_feas + 1) % 100 == 0:
			el = time.time() - t0
			print(f"  [{out_dir.name}] infeasible {i - n_feas + 1}/{n_infeas}  ({el:.0f}s, {discarded} retries-exhausted)", flush=True)
	print(f"[{out_dir.name}] done: {total} instances in {time.time() - t0:.0f}s", flush=True)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out_root', type=str, default='../instances')
	ap.add_argument('--sizes', nargs='+', default=['10x25', '20x50', '40x100'])
	ap.add_argument('--train_feas', type=int, default=8000)
	ap.add_argument('--train_infeas', type=int, default=2000)
	ap.add_argument('--test_feas', type=int, default=800)
	ap.add_argument('--test_infeas', type=int, default=200)
	ap.add_argument('--time_limit', type=float, default=30.0)
	ap.add_argument('--seed', type=int, default=7)
	args = ap.parse_args()

	root = Path(args.out_root)
	for s_idx, size in enumerate(args.sizes):
		m, n = (int(v) for v in size.split('x'))
		n_train = args.train_feas + args.train_infeas
		n_test = args.test_feas + args.test_infeas
		gen_split(root / f"train_instances_dnsms_{size}_{n_train}", m, n,
			args.train_feas, args.train_infeas, args.seed + 10 * s_idx, args.time_limit)
		gen_split(root / f"test_instances_dnsms_{size}_{n_test}", m, n,
			args.test_feas, args.test_infeas, args.seed + 10 * s_idx + 5, args.time_limit)


if __name__ == '__main__':
	main()
