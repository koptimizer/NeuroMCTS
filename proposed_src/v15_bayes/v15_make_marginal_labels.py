"""Build exact marginal labels for the soft-target experiment.

Every training run in this project has used the planted x* as the per-variable label.
But x* is one uniform sample from the solution set S, not the answer: at 10x25 the
median |S| is 29, and even a Bayes-optimal predictor agrees with x* only 66% of the
time (v15_bayes_limit.py). Training on x* is therefore training on ~34% label noise,
and each instance is visited about once, so the noise never averages out.

This script enumerates S exactly with CP-SAT and stores the true marginal
    p_j = P(x_j = 1 | A, b) = |{x in S : x_j = 1}| / |S|
so a model can be trained against the posterior instead of a sample from it.
Instances whose enumeration hits the cap are dropped: a truncated S gives biased
marginals, which would reintroduce exactly the noise this experiment removes.
"""
# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

from v15_bayes_limit import enumerate_solutions


def make_one(task):
	idx, path, cap, time_limit = task
	d = json.load(open(path))
	if not d.get('feasible', True) or d.get('x') is None:
		return idx, None
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	S, complete = enumerate_solutions(A, b, cap, time_limit)
	if len(S) == 0 or not complete:
		return idx, None
	return idx, dict(A=d['A'], b=d['b'], x=d['x'],
	                  marginals=S.mean(axis=0).tolist(), n_sols=int(len(S)))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--limit', type=int, default=4000)
	ap.add_argument('--cap', type=int, default=200000)
	ap.add_argument('--time_limit', type=float, default=60.0)
	ap.add_argument('--jobs', type=int, default=16)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	files = sorted(Path(a.data_dir).glob('*.json'))[:a.limit]
	tasks = [(i, str(f), a.cap, a.time_limit) for i, f in enumerate(files)]
	print(f"enumerating {len(tasks)} instances from {a.data_dir}", flush=True)

	out = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		done = 0
		for idx, rec in pool.imap_unordered(make_one, tasks, chunksize=4):
			done += 1
			if rec is not None:
				out.append(rec)
			if done % 250 == 0:
				print(f"  {done}/{len(tasks)}  kept={len(out)}  ({time.time()-t0:.0f}s)", flush=True)

	ns = np.array([r['n_sols'] for r in out])
	print(f"\nkept {len(out)}/{len(tasks)} instances "
	      f"({100*len(out)/max(1,len(tasks)):.1f}%; the rest were infeasible or hit the cap)")
	print(f"|S|: median={np.median(ns):.0f}  mean={ns.mean():.1f}  max={ns.max()}")
	json.dump(out, open(a.out, 'w'))
	print(f"saved: {a.out}  ({Path(a.out).stat().st_size/1e6:.1f} MB)")


if __name__ == '__main__':
	main()
