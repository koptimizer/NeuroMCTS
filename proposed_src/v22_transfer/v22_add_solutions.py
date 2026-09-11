"""Attach full solution sets to instance files that carry only marginals.

v15_gen_by_m.py stores p_j but drops S itself, and v17_make_conditional_data.py needs S
to build depth-d states (it samples a prefix from a real solution and recomputes the
marginal over the survivors). Re-enumerating is cheap at the sizes involved, so this
backfills rather than regenerating the instances.

Instances whose enumeration no longer completes are dropped, keeping the invariant that
every stored marginal is exact.
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

from v15_bayes_limit import enumerate_solutions


def one(task):
	idx, rec, cap, time_limit = task
	A = np.array(rec['A'], dtype=np.int64)
	b = np.array(rec['b'], dtype=np.int64)
	S, ok = enumerate_solutions(A, b, cap, time_limit)
	if not ok or len(S) == 0:
		return idx, None
	return idx, dict(A=rec['A'], b=rec['b'], x=rec['x'],
	                  solutions=S.tolist(), n_sols=int(len(S)))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--inp', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--cap', type=int, default=100000)
	ap.add_argument('--time_limit', type=float, default=30.0)
	ap.add_argument('--jobs', type=int, default=12)
	a = ap.parse_args()

	recs = json.load(open(a.inp))
	tasks = [(i, r, a.cap, a.time_limit) for i, r in enumerate(recs)]
	print(f"re-enumerating {len(recs)} instances from {a.inp}", flush=True)
	out = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		for idx, r in pool.imap_unordered(one, tasks, chunksize=4):
			if r is not None:
				out.append(r)
	ns = np.array([r['n_sols'] for r in out])
	print(f"kept {len(out)}/{len(recs)} in {time.time()-t0:.0f}s  |S| median={np.median(ns):.0f}")
	json.dump(out, open(a.out, 'w'))
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
