"""v17: build training data for CONDITIONAL marginal prediction (prediction inside the tree).

v15 trained on root states only, where the ceiling is 70.5%. v16 showed the ceiling rises
to 93.4% by depth 8 while propagation still forces only 8.2% of the remaining variables --
a window where signal exists and deduction has not yet closed it. This script produces
training examples from inside that window.

Key simplification: conditioning on a partial assignment is the same thing as looking at
the reduced instance. Fixing variables S to values v leaves
    {x_free : A_free x_free = b - A_S v}
whose marginals ARE the conditional marginals. So a depth-d example is just an ordinary
(A', b', marginals) record on a smaller instance, and the v15 model/trainer work unchanged.

Depths are sampled uniformly over the range a search actually visits, and the prefix is
taken from a real solution so the state is reachable (a prefix inconsistent with every
solution is a dead node that search would have pruned, not a state worth learning on).
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

import LPneuroBLS_v7 as m


def make_examples(task):
	idx, rec, max_depth, per_instance, seed = task
	A = np.array(rec['A'], dtype=np.int64)
	b = np.array(rec['b'], dtype=np.int64)
	S = np.array(rec['solutions'], dtype=np.int64)
	n = A.shape[1]
	rng = np.random.default_rng(seed)

	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if not ok:
		return idx, []
	order = np.argsort(-np.abs(x_lp - 0.5))     # the order a search guided by LP would use

	out = []
	for _ in range(per_instance):
		d = int(rng.integers(0, max_depth + 1))
		if d >= n - 2:
			continue
		x_star = S[rng.integers(len(S))]
		fixed = order[:d]
		keep = order[d:]
		mask = np.ones(len(S), dtype=bool) if d == 0 else np.all(S[:, fixed] == x_star[fixed], axis=1)
		Sd = S[mask]
		if len(Sd) == 0:
			continue
		A_red = A[:, keep]
		b_red = b - A[:, fixed].dot(x_star[fixed]) if d > 0 else b.copy()
		# drop rows that became vacuous; they carry no constraint and only add noise
		live = ~((A_red.sum(axis=1) == 0) & (b_red == 0))
		if live.sum() == 0:
			continue
		out.append(dict(A=A_red[live].tolist(), b=b_red[live].tolist(),
		                 x=Sd[rng.integers(len(Sd))][keep].tolist(),
		                 marginals=Sd[:, keep].mean(axis=0).tolist(),
		                 depth=d, n_surviving=int(len(Sd))))
	return idx, out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sols', required=True, help='json with A, b and enumerated solutions')
	ap.add_argument('--max_depth', type=int, default=12)
	ap.add_argument('--per_instance', type=int, default=6)
	ap.add_argument('--jobs', type=int, default=12)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	recs = json.load(open(a.sols))
	tasks = [(i, r, a.max_depth, a.per_instance, 7000 + i) for i, r in enumerate(recs)]
	print(f"building conditional examples from {len(recs)} instances "
	      f"(depths 0..{a.max_depth}, {a.per_instance} per instance)", flush=True)

	out = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		for idx, exs in pool.imap_unordered(make_examples, tasks, chunksize=4):
			out.extend(exs)
	depths = np.array([r['depth'] for r in out])
	ns = np.array([r['n_surviving'] for r in out])
	print(f"\n{len(out)} examples in {time.time()-t0:.0f}s")
	print(f"  depth   : min={depths.min()} median={np.median(depths):.0f} max={depths.max()}")
	print(f"  |S_d|   : median={np.median(ns):.1f}  (1 means the prefix already pins the solution)")
	ceil = np.mean([np.maximum(np.array(r['marginals']), 1 - np.array(r['marginals'])).mean() for r in out])
	print(f"  mean conditional ceiling over the sampled states: {100*ceil:.1f}%")
	json.dump(out, open(a.out, 'w'))
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
