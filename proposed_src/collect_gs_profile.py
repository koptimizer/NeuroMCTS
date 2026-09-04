#!/usr/bin/env python3
"""
Collects the full per-index Gram-Schmidt log-norm profile from the BKZ-reduced
AHL lattice (the raw sequence ahl_solve_trace already computes internally but
previously discarded after fitting a single slope) for each instance in a
matched-pair dataset, for training a sequence model on the profile shape
instead of hand-picked summary statistics (min/mean/std) that only reached
AUC 0.588 in prior work.

Usage:
  python collect_gs_profile.py --data ../runs/paired_signal_20x50.jsonl \
      --out ../runs/gs_profile_20x50.jsonl --block 10 --tries 8 --jobs 8
"""
import argparse
import json
import time
import numpy as np
from multiprocessing import Pool

import sys
sys.path.insert(0, '.')
from lattice_enum import ahl_solve_trace


def process(task):
	rec, block, tries = task
	A = np.array(rec['A'], dtype=np.int64)
	b = np.array(rec['b'], dtype=np.int64)
	sol, hit, trace = ahl_solve_trace(A, b, block=block, tries=tries, seed=0)
	best = min(trace, key=lambda t: t['min_norm'])
	return {
		'pair_id': rec['pair_id'], 'm': rec['m'], 'n': rec['n'],
		'gt_feasible': rec['gt_feasible'],
		'profile': best['profile'], 'min_norm': best['min_norm'],
		'ahl_solved': sol is not None,
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--block', type=int, default=10)
	ap.add_argument('--tries', type=int, default=8)
	ap.add_argument('--jobs', type=int, default=8)
	a = ap.parse_args()

	recs = [json.loads(l) for l in open(a.data)]
	tasks = [(r, a.block, a.tries) for r in recs]
	t0 = time.time()
	with Pool(a.jobs) as pool, open(a.out, 'w') as out:
		for i, rec in enumerate(pool.imap(process, tasks, chunksize=8)):
			out.write(json.dumps(rec) + '\n')
			if (i + 1) % 500 == 0:
				print(f"  {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
	print(f"done: {len(tasks)} instances, {time.time()-t0:.0f}s total")


if __name__ == '__main__':
	main()
