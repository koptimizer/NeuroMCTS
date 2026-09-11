"""Build a test-instance directory at a chosen (m, n), with |S| recorded per instance.

Conditions C and D need a test size where search still discriminates between guidance
arms. n=100 is ruled out (v22_sbound_n100.py: CP-SAT finds zero solutions in 20s despite
a planted one), so the size is chosen from the n=60/70 sweeps and built here.

Unlike the training sizes, these instances need no marginal labels -- only whether an arm
solves them. |S| is still enumerated and stored so the |S|-matching claim can be checked
rather than assumed; instances whose enumeration does not complete are kept but marked,
since an unverified |S| must not silently pass as a verified one.
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

from generate_hard_instances import gen_hard_feasible
from v15_bayes_limit import enumerate_solutions


def make_one(task):
	idx, mm, n, K, cap, tl = task
	rng = np.random.default_rng(idx)
	A, b, x, _ = gen_hard_feasible(mm, n, rng, K=K)
	S, ok = enumerate_solutions(A, b, cap, tl)
	return idx, dict(A=A.tolist(), b=b.tolist(), x=x.tolist(), feasible=True,
	                  n_sols=(int(len(S)) if ok else -1), s_verified=bool(ok))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--m', type=int, required=True)
	ap.add_argument('--n', type=int, required=True)
	ap.add_argument('--count', type=int, default=30)
	ap.add_argument('--K', type=int, default=20)
	ap.add_argument('--cap', type=int, default=100000)
	ap.add_argument('--time_limit', type=float, default=120.0)
	ap.add_argument('--jobs', type=int, default=10)
	ap.add_argument('--out_dir', required=True)
	a = ap.parse_args()

	tasks = [(700000 + i, a.m, a.n, a.K, a.cap, a.time_limit) for i in range(a.count)]
	out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
	print(f"building {a.count} instances at {a.m}x{a.n}", flush=True)
	recs = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		for idx, r in pool.imap_unordered(make_one, tasks, chunksize=1):
			recs.append(r)
	for i, r in enumerate(sorted(recs, key=lambda z: z['n_sols'])):
		json.dump(r, open(out_dir / f"inst-{i:04d}.json", 'w'))
	ver = [r['n_sols'] for r in recs if r['s_verified']]
	print(f"wrote {len(recs)} to {out_dir} in {time.time()-t0:.0f}s")
	print(f"  |S| verified for {len(ver)}/{len(recs)}"
	      + (f"; median={np.median(ver):.0f} min={min(ver)} max={max(ver)}" if ver else ""))


if __name__ == '__main__':
	main()
