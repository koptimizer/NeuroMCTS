#!/usr/bin/env python3
"""
NeuroSAT-style paired feasible/infeasible instance collector.

Unlike the independent feasible/infeasible pools used in prior cycles, each
pair here shares the SAME coefficient matrix A and differs only by a minimal
perturbation of b (1-3 rows, +/-1), mirroring SR(n)'s single-literal-flip
construction: the twin problems are matched on every superficial statistic
(density, row degrees, size), so any signal a classifier finds must reflect
the actual arithmetic feasibility structure, not dataset-level shortcuts.

Only pairs surviving propagation/probing and the LP hard rule on BOTH members
are kept (the genuinely ambiguous residual set, consistent with
collect_feas_signal.py's methodology); each kept instance is probed with the
same aggregate multi-restart pump + cheap AHL statistics as before.

Usage:
  python collect_paired_signal.py --out ../runs/paired_signal_20x50.jsonl \
      --size 20x50 --n_pairs 3000 --jobs 8
"""
import argparse
import json
import time
import numpy as np
from multiprocessing import Pool
from ortools.sat.python import cp_model

import sys
sys.path.insert(0, '.')
from LPneuroBLS_v7 import FastBinaryEnv, solve_lp_relaxation, gen_planted_instance
from lattice_enum import ahl_solve_trace


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
	return solver.Solve(model) == cp_model.INFEASIBLE


def kernel_pump_once(x0, A, b, A_pinv, max_iter=200, seed=0):
	rng = np.random.default_rng(seed)
	Af = A.astype(np.float64); bf = b.astype(np.float64)
	x = np.clip(x0.astype(np.float64), 0.0, 1.0)
	prev = None
	res = np.inf
	for it in range(1, max_iter + 1):
		x_r = np.round(np.clip(x, 0.0, 1.0))
		res = np.abs(Af.dot(x_r) - bf).sum()
		if res == 0:
			return True, 0.0
		if prev is not None and np.array_equal(x_r, prev):
			flip = rng.choice(x_r.shape[0], size=max(1, x_r.shape[0] // 5), replace=False)
			x_r[flip] = 1.0 - x_r[flip]
		prev = x_r.copy()
		x = x_r - A_pinv.dot(Af.dot(x_r) - bf)
	return False, res


def probe_features(A, b, pump_restarts, ahl_tries, ahl_block):
	# Returns None if resolved trivially by propagation/LP (not part of the
	# ambiguous residual set), else the aggregate probe feature dict
	env = FastBinaryEnv(A, b).propagate_constraints().apply_probing()
	if env.is_invalid or env.is_terminal():
		return None
	x_lp, lp_ok = solve_lp_relaxation(A, b)
	if not lp_ok:
		return None
	A_pinv = np.linalg.pinv(A.astype(np.float64))
	pump_residuals = []
	pump_solved_any = False
	for s in range(pump_restarts):
		solved, res = kernel_pump_once(x_lp, A, b, A_pinv, max_iter=200, seed=s)
		pump_residuals.append(res / A.shape[0])
		if solved:
			pump_solved_any = True
			break
	pump_residuals = np.array(pump_residuals)
	ahl_solved = False
	ahl_min_norms = []
	if not pump_solved_any:
		sol, hit, trace = ahl_solve_trace(A, b, block=ahl_block, tries=ahl_tries, seed=0)
		ahl_solved = sol is not None
		ahl_min_norms = [t['min_norm'] / np.sqrt(A.shape[1] + 1.0) for t in trace]
	lp_frac = float(np.abs(x_lp - np.rint(x_lp)).mean())
	return {
		'lp_frac': lp_frac,
		'pump_solved_any': pump_solved_any,
		'pump_res_min': float(pump_residuals.min()),
		'pump_res_mean': float(pump_residuals.mean()),
		'pump_res_std': float(pump_residuals.std()),
		'ahl_solved': ahl_solved,
		'ahl_min_norm_min': float(min(ahl_min_norms)) if ahl_min_norms else None,
		'ahl_min_norm_mean': float(np.mean(ahl_min_norms)) if ahl_min_norms else None,
		'ahl_min_norm_std': float(np.std(ahl_min_norms)) if ahl_min_norms else None,
	}


def make_pair(task):
	pair_id, m, n, time_limit, pump_restarts, ahl_tries, ahl_block = task
	rng = np.random.default_rng(500000 + pair_id)
	for attempt in range(20):
		A, b_feas, x = gen_planted_instance(m, n, rng)
		A = A.astype(np.int64)
		b_feas = b_feas.astype(np.int64)
		rowsum = A.sum(axis=1)
		k = int(rng.integers(1, 4))
		rows = rng.choice(m, k, replace=False)
		b_infeas = b_feas.copy()
		b_infeas[rows] += rng.choice([-1, 1], size=k)
		b_infeas = np.clip(b_infeas, 0, rowsum)
		if np.array_equal(b_infeas, b_feas):
			continue
		if not cpsat_infeasible(A, b_infeas, time_limit):
			continue
		# both members share A; probe each independently
		feat_f = probe_features(A, b_feas, pump_restarts, ahl_tries, ahl_block)
		feat_i = probe_features(A, b_infeas, pump_restarts, ahl_tries, ahl_block)
		if feat_f is None or feat_i is None:
			return None  # one side resolved trivially; not a useful ambiguous pair
		A_list = A.tolist()
		rec_f = {'pair_id': pair_id, 'm': m, 'n': n, 'gt_feasible': True, 'A': A_list, 'b': b_feas.tolist(), **feat_f}
		rec_i = {'pair_id': pair_id, 'm': m, 'n': n, 'gt_feasible': False, 'A': A_list, 'b': b_infeas.tolist(), **feat_i}
		return rec_f, rec_i
	return None


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--size', default='20x50')
	ap.add_argument('--n_pairs', type=int, default=3000)
	ap.add_argument('--time_limit', type=float, default=20.0)
	ap.add_argument('--pump_restarts', type=int, default=15)
	ap.add_argument('--ahl_tries', type=int, default=8)
	ap.add_argument('--ahl_block', type=int, default=10)
	ap.add_argument('--jobs', type=int, default=8)
	a = ap.parse_args()
	m, n = (int(v) for v in a.size.split('x'))

	tasks = [(pid, m, n, a.time_limit, a.pump_restarts, a.ahl_tries, a.ahl_block) for pid in range(a.n_pairs)]
	t0 = time.time()
	kept = skipped = 0
	with Pool(a.jobs) as pool, open(a.out, 'w') as out:
		for i, res in enumerate(pool.imap_unordered(make_pair, tasks, chunksize=2)):
			if res is None:
				skipped += 1
				continue
			rec_f, rec_i = res
			out.write(json.dumps(rec_f) + '\n')
			out.write(json.dumps(rec_i) + '\n')
			out.flush()
			kept += 1
			if (i + 1) % 200 == 0:
				print(f"  {i+1}/{len(tasks)}  (kept_pairs={kept}, skipped={skipped}, {time.time()-t0:.0f}s)", flush=True)
	print(f"done: kept {kept} matched pairs ({2*kept} instances), skipped {skipped}, {time.time()-t0:.0f}s total")


if __name__ == '__main__':
	main()
