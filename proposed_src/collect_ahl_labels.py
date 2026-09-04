#!/usr/bin/env python3
"""
Offline collector for AHL try-to-hit labels and per-try traces.

Fresh planted instances (feasible by construction) are generated per size and
solved with ahl_solve_trace under a full try budget; the resulting records
serve both the hazard-model training set (experiment 0) and the coverage-time
Pareto grid over BKZ block sizes (experiment 1).

Usage:
  python collect_ahl_labels.py --out runs/ahl_labels.jsonl \
      --jobs 8 --spec 20x50:40:1500 20x50:20:500 20x50:60:500 30x75:40:600 40x100:40:200
"""
import argparse
import json
import time
import numpy as np
from multiprocessing import Pool

import sys
sys.path.insert(0, '.')
from lattice_enum import ahl_solve_trace
from LPneuroBLS_v7 import gen_planted_instance, solve_lp_relaxation


def run_one(task):
	size, block, idx, tries = task
	mm, nn = size
	rng = np.random.default_rng(100000 * block + idx)
	A, b, x = gen_planted_instance(mm, nn, rng)
	x_lp, _ = solve_lp_relaxation(A, b)
	frac = float(np.abs(x_lp - np.rint(x_lp)).mean())
	t0 = time.time()
	sol, hit, trace = ahl_solve_trace(A.astype(np.int64), b.astype(np.int64), block=block, tries=tries, seed=idx)
	return {
		'm': mm, 'n': nn, 'block': block, 'idx': idx,
		'hit_try': hit, 'solved': sol is not None,
		'lp_frac': frac, 'total_sec': time.time() - t0,
		'trace': trace,
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--jobs', type=int, default=8)
	ap.add_argument('--spec', nargs='+', required=True, help='mxn:block:count ...')
	ap.add_argument('--tries', type=int, default=10)
	args = ap.parse_args()

	tasks = []
	for spec in args.spec:
		size_s, block_s, cnt_s = spec.split(':')
		mm, nn = (int(v) for v in size_s.split('x'))
		for i in range(int(cnt_s)):
			tasks.append(((mm, nn), int(block_s), i, args.tries))
	print(f"{len(tasks)} tasks, {args.jobs} workers", flush=True)

	done = 0
	t0 = time.time()
	with Pool(args.jobs) as pool, open(args.out, 'w') as f:
		for rec in pool.imap_unordered(run_one, tasks, chunksize=4):
			f.write(json.dumps(rec) + '\n')
			f.flush()
			done += 1
			if done % 100 == 0:
				print(f"  {done}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)
	print(f"done: {len(tasks)} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
	main()
