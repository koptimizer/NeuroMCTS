#!/usr/bin/env python3
"""
Offline collector testing a new hypothesis for learned infeasibility detection:
does the AGGREGATE outcome of many independent symbolic probes (kernel-pump
restarts, cheap AHL tries) separate true-infeasible instances from hard-but-
feasible ones? This differs from the v9 scheduler question (will THIS retry
succeed on a KNOWN-feasible instance) — here the label is ground-truth
feasibility itself, and the feature is "how close did many independent
attempts get, in aggregate" rather than any single attempt's trace.

Only the RESIDUAL set is probed: instances that survive propagation/probing
and the LP hard rule (LP itself is feasible), since those are the only ones
where feasibility is actually ambiguous — propagation and LP infeasibility
already resolve the rest for free.

Usage:
  python collect_feas_signal.py --data_dir ../instances/train_instances_dnsms_20x50_10000 \
      --out ../runs/feas_signal_20x50.jsonl --n_feas 1500 --n_infeas 1500 \
      --pump_restarts 15 --ahl_tries 8 --ahl_block 10 --jobs 8
"""
import argparse
import json
import random
import time
import numpy as np
from multiprocessing import Pool
from pathlib import Path

import sys
sys.path.insert(0, '.')
from LPneuroBLS_v7 import FastBinaryEnv, solve_lp_relaxation
from lattice_enum import ahl_solve_trace


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


def probe_one(task):
	path, pump_restarts, ahl_tries, ahl_block = task
	d = json.load(open(path))
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	gt = bool(d.get('feasible', True))
	m, n = A.shape

	env = FastBinaryEnv(A, b).propagate_constraints().apply_probing()
	if env.is_invalid:
		return None  # resolved by propagation, not part of the residual set
	if env.is_terminal():
		return None

	x_lp, lp_ok = solve_lp_relaxation(A, b)
	if not lp_ok:
		return None  # resolved by LP hard rule

	t0 = time.time()
	A_pinv = np.linalg.pinv(A.astype(np.float64))
	pump_residuals = []
	pump_solved_any = False
	for s in range(pump_restarts):
		solved, res = kernel_pump_once(x_lp, A, b, A_pinv, max_iter=200, seed=s)
		pump_residuals.append(res / m)  # normalize by row count
		if solved:
			pump_solved_any = True
			break
	pump_residuals = np.array(pump_residuals)

	ahl_solved = False
	ahl_min_norms = []
	if not pump_solved_any:
		sol, hit, trace = ahl_solve_trace(A, b, block=ahl_block, tries=ahl_tries, seed=0)
		ahl_solved = sol is not None
		ahl_min_norms = [tr['min_norm'] / np.sqrt(n + 1.0) for tr in trace]

	lp_frac = float(np.abs(x_lp - np.rint(x_lp)).mean())
	elapsed = time.time() - t0

	return {
		'path': str(path), 'm': m, 'n': n, 'gt_feasible': gt,
		'lp_frac': lp_frac,
		'pump_solved_any': pump_solved_any,
		'pump_res_min': float(pump_residuals.min()) if len(pump_residuals) else None,
		'pump_res_mean': float(pump_residuals.mean()) if len(pump_residuals) else None,
		'pump_res_std': float(pump_residuals.std()) if len(pump_residuals) else None,
		'pump_n_restarts_used': len(pump_residuals),
		'ahl_solved': ahl_solved,
		'ahl_min_norm_min': float(min(ahl_min_norms)) if ahl_min_norms else None,
		'ahl_min_norm_mean': float(np.mean(ahl_min_norms)) if ahl_min_norms else None,
		'ahl_min_norm_std': float(np.std(ahl_min_norms)) if ahl_min_norms else None,
		'elapsed_sec': elapsed,
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--n_feas', type=int, default=1500)
	ap.add_argument('--n_infeas', type=int, default=1500)
	ap.add_argument('--pump_restarts', type=int, default=15)
	ap.add_argument('--ahl_tries', type=int, default=8)
	ap.add_argument('--ahl_block', type=int, default=10)
	ap.add_argument('--jobs', type=int, default=8)
	ap.add_argument('--seed', type=int, default=0)
	a = ap.parse_args()

	files = sorted(Path(a.data_dir).glob('*.json'))
	feas, infeas = [], []
	for f in files:
		d = json.load(open(f))
		(feas if d.get('feasible', True) else infeas).append(f)
	rng = random.Random(a.seed)
	rng.shuffle(feas); rng.shuffle(infeas)
	sel = feas[:a.n_feas] + infeas[:a.n_infeas]
	print(f"pool: {len(feas)} feasible / {len(infeas)} infeasible available; sampling {len(sel)}", flush=True)

	tasks = [(f, a.pump_restarts, a.ahl_tries, a.ahl_block) for f in sel]
	t0 = time.time()
	kept = skipped = 0
	with Pool(a.jobs) as pool, open(a.out, 'w') as out:
		for i, rec in enumerate(pool.imap_unordered(probe_one, tasks, chunksize=4)):
			if rec is None:
				skipped += 1
				continue
			out.write(json.dumps(rec) + '\n'); out.flush()
			kept += 1
			if (i + 1) % 200 == 0:
				print(f"  {i+1}/{len(tasks)}  (kept={kept}, skipped-resolved={skipped}, {time.time()-t0:.0f}s)", flush=True)
	print(f"done: kept {kept} residual instances, {skipped} resolved trivially, {time.time()-t0:.0f}s total")


if __name__ == '__main__':
	main()
