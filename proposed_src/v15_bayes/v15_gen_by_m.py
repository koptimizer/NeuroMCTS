"""Generate n=25 instances at a chosen m, with exact marginal labels attached.

Holding n fixed and raising m shrinks the solution set, which raises the Bayes ceiling
(measured: m=10 -> |S| median 24, ceiling 70.4%; m=20 -> |S|=1 always, ceiling 100%).
That turns the ceiling into an experimental knob and lets us ask the project's central
question directly: when the ceiling rises, does a learned predictor follow it?

  accuracy tracks the ceiling -> solution multiplicity was the binding constraint
  accuracy stays flat         -> the information is present but not extractable,
                                 i.e. a computational rather than informational wall
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
from multiprocessing import Pool
import numpy as np

from generate_hard_instances import gen_hard_feasible
from v15_bayes_limit import enumerate_solutions


def make_one(task):
	idx, mm, n, K, cap, time_limit = task
	rng = np.random.default_rng(idx)
	A, b, x, _ = gen_hard_feasible(mm, n, rng, K=K)
	S, complete = enumerate_solutions(A, b, cap, time_limit)
	if len(S) == 0 or not complete:
		return idx, None
	return idx, dict(A=A.tolist(), b=b.tolist(), x=x.tolist(),
	                  marginals=S.mean(axis=0).tolist(), n_sols=int(len(S)))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--m', type=int, required=True)
	ap.add_argument('--n', type=int, default=25)
	ap.add_argument('--count', type=int, default=800)
	ap.add_argument('--K', type=int, default=20)
	ap.add_argument('--cap', type=int, default=50000)
	ap.add_argument('--time_limit', type=float, default=20.0)
	ap.add_argument('--jobs', type=int, default=16)
	ap.add_argument('--seed0', type=int, default=900000)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	tasks = [(a.seed0 + i, a.m, a.n, a.K, a.cap, a.time_limit) for i in range(a.count)]
	print(f"generating {a.count} instances at {a.m}x{a.n} (K={a.K})", flush=True)
	out = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		done = 0
		for idx, rec in pool.imap_unordered(make_one, tasks, chunksize=2):
			done += 1
			if rec is not None:
				out.append(rec)
			if done % 200 == 0:
				print(f"  {done}/{a.count}  kept={len(out)}  ({time.time()-t0:.0f}s)", flush=True)

	ns = np.array([r['n_sols'] for r in out])
	ceil = np.mean([np.maximum(np.array(r['marginals']), 1 - np.array(r['marginals'])).mean() for r in out])
	print(f"\n{a.m}x{a.n}: kept {len(out)}/{a.count}")
	print(f"  |S|: median={np.median(ns):.0f}  unique-fraction={100*np.mean(ns==1):.1f}%")
	print(f"  Bayes ceiling: {100*ceil:.1f}%")
	json.dump(out, open(a.out, 'w'))
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
