"""v24: in-tree states with a chosen number of free variables, labelled with exact marginals.

For each instance, a prefix of the LP-confidence ordering is fixed to the planted solution so
that n_free variables remain (the reduction identity, Proposition 1), vacuous rows are dropped,
and the reduced instance is enumerated. Only enumerations that run to exhaustion produce a
record (OPTIMAL/INFEASIBLE rule of v15_bayes_limit); everything else is counted and reported,
because the states that enumerate within the limit are biased toward small surviving solution
sets and that bias has to be visible in the log rather than silently baked into the labels.

Difference from v17_make_conditional_data.py: the prefix comes from the planted solution rather
than a random member of S, since S is not enumerated at the root here.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import time
from multiprocessing import Pool
import numpy as np

import LPneuroBLS_v7 as m
from v15_bayes_limit import enumerate_solutions


def one(task):
	idx, rec, sizes, cap, tl = task
	A = np.array(rec['A'], dtype=np.int64)
	b = np.array(rec['b'], dtype=np.int64)
	x = np.array(rec['x'], dtype=np.int64)
	n = A.shape[1]
	rng = np.random.default_rng(10_000 + idx)
	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if not ok:
		return idx, [], []
	order = np.argsort(-np.abs(x_lp - 0.5))
	out, stats = [], []
	for nf in sizes:
		d = n - nf
		fixed, keep = order[:d], order[d:]
		A_red = A[:, keep]
		b_red = b - A[:, fixed].dot(x[fixed]) if d > 0 else b.copy()
		live = ~((A_red.sum(axis=1) == 0) & (b_red == 0))
		A_red, b_red = A_red[live], b_red[live]
		t0 = time.time()
		S, complete = enumerate_solutions(A_red, b_red, cap, tl)
		el = time.time() - t0
		stats.append(dict(n_free=nf, m_rows=int(live.sum()), complete=bool(complete),
		                  n_sols=int(len(S)), sec=round(el, 1)))
		if not complete or len(S) == 0:
			continue
		out.append(dict(A=A_red.tolist(), b=b_red.tolist(), x=S[rng.integers(len(S))].tolist(),
		                 marginals=S.mean(axis=0).tolist(), depth=d, n_free=nf,
		                 n_surviving=int(len(S)), m_rows=int(live.sum()), src=idx))
	return idx, out, stats


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--pool', required=True)
	ap.add_argument('--sizes', type=int, nargs='+', default=[30, 35, 40, 45, 50, 55, 60])
	ap.add_argument('--cap', type=int, default=200000)
	ap.add_argument('--time_limit', type=float, default=90.0)
	ap.add_argument('--jobs', type=int, default=12)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	recs = json.load(open(a.pool))
	tasks = [(i, r, a.sizes, a.cap, a.time_limit) for i, r in enumerate(recs)]
	print(f"building range states from {len(recs)} instances, n_free in {a.sizes}, "
	      f"enumeration limit {a.time_limit}s", flush=True)
	out, stats = [], []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		for k, (idx, exs, st) in enumerate(pool.imap_unordered(one, tasks, chunksize=1), 1):
			out.extend(exs); stats.extend(st)
			if k % 50 == 0:
				print(f"  {k}/{len(recs)} instances, {len(out)} states ({time.time()-t0:.0f}s)", flush=True)
	print(f"\n{len(out)} states in {time.time()-t0:.0f}s")
	print(f"{'n_free':>7}{'attempted':>11}{'enumerated':>12}{'|S| median':>12}{'sec median':>12}{'m/|K| median':>14}")
	for nf in a.sizes:
		st = [s for s in stats if s['n_free'] == nf]
		ok = [s for s in st if s['complete'] and s['n_sols'] > 0]
		if st:
			print(f"{nf:>7}{len(st):>11}{len(ok):>12}"
			      f"{(np.median([s['n_sols'] for s in ok]) if ok else float('nan')):>12.0f}"
			      f"{np.median([s['sec'] for s in st]):>12.1f}"
			      f"{(np.median([s['m_rows'] / nf for s in ok]) if ok else float('nan')):>14.2f}")
	json.dump(out, open(a.out, 'w'))
	json.dump(stats, open(a.out.replace('.json', '_stats.json'), 'w'))
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
