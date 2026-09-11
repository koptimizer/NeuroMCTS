"""v16: is there a depth range where learned guidance could actually pay?

v15 measured the ceiling for one-shot prediction from (A, b): 70.5% on the hard family.
But search does not predict from the root -- it descends, and deduction manufactures
information on the way down. So the relevant quantity is the CONDITIONAL ceiling at
depth d, and the honest question is not whether it rises (it must; at depth n you know
everything) but whether it rises in a range where deduction has NOT already settled the
matter. Learned guidance only has value in that window.

Two curves are measured per depth d, conditioning on a partial assignment taken from a
real solution (i.e. assuming search never strays off a correct path -- the optimistic
case for learning):

  conditional ceiling : mean_j max(p_j^(d), 1 - p_j^(d)) over still-free j, where
                        p^(d) is the marginal over solutions consistent with the prefix
  propagation power   : how many still-free variables constraint propagation forces
                        outright at that node

Readout:
  ceiling high AND propagation low  -> a real window for learned guidance
  ceiling high AND propagation high -> deduction already does it; learning is redundant
                                        (consistent with the measured 0% MCTS contribution)
  ceiling low                        -> still guessing; no signal to learn from
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

import LPneuroBLS_v7 as m


def analyze_one(task):
	idx, rec, depths, order_mode, seed = task
	A = np.array(rec['A'], dtype=np.int64)
	b = np.array(rec['b'], dtype=np.int64)
	S = np.array(rec['solutions'], dtype=np.int64)
	n = A.shape[1]
	rng = np.random.default_rng(seed)

	x_star = S[rng.integers(len(S))]          # the path search is assumed to follow
	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if not ok:
		return idx, None
	if order_mode == 'lp':
		order = np.argsort(-np.abs(x_lp - 0.5))   # most LP-confident first
	else:
		order = rng.permutation(n)

	out = []
	for d in depths:
		if d >= n:
			continue
		fixed = order[:d]
		# solutions consistent with the depth-d prefix of x_star
		mask = np.ones(len(S), dtype=bool) if d == 0 else np.all(S[:, fixed] == x_star[fixed], axis=1)
		Sd = S[mask]
		free = order[d:]
		p = Sd[:, free].mean(axis=0)
		ceiling = float(np.maximum(p, 1 - p).mean())

		env = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
		for j in fixed:
			env.step_inplace(int(j), int(x_star[j]))
		env.propagate_constraints().apply_probing()
		n_forced = int(np.sum(env.assignment[free] != -1))

		out.append(dict(depth=int(d), n_surviving=int(len(Sd)), ceiling=ceiling,
		                 n_free=int(len(free)), n_forced=n_forced,
		                 forced_frac=float(n_forced / max(1, len(free)))))
	return idx, out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--labels', required=True, help='json with A,b and enumerated solutions')
	ap.add_argument('--limit', type=int, default=120)
	ap.add_argument('--depths', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10, 12, 15, 18])
	ap.add_argument('--order', choices=['lp', 'random'], default='lp')
	ap.add_argument('--jobs', type=int, default=12)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	recs = json.load(open(a.labels))[:a.limit]
	tasks = [(i, r, a.depths, a.order, 3000 + i) for i, r in enumerate(recs)]
	print(f"analyzing {len(tasks)} instances, order={a.order}", flush=True)

	rows = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		for idx, res in pool.imap_unordered(analyze_one, tasks, chunksize=2):
			if res is not None:
				rows.append(res)
	print(f"done in {time.time()-t0:.0f}s\n", flush=True)

	print("=" * 88)
	print(f"v16: conditional information vs search depth  (n=25, order={a.order}, {len(rows)} instances)")
	print("=" * 88)
	print(f"{'depth':>6}{'surviving |S|':>15}{'cond. ceiling':>15}{'propagation forces':>20}{'learning window':>17}")
	for di, d in enumerate(a.depths):
		vals = [r[di] for r in rows if di < len(r)]
		if not vals:
			continue
		ns = np.mean([v['n_surviving'] for v in vals])
		ce = np.mean([v['ceiling'] for v in vals])
		ff = np.mean([v['forced_frac'] for v in vals])
		# a window exists where the ceiling is informative but deduction has not resolved it
		window = (ce - 0.5) * (1 - ff)
		print(f"{d:>6}{ns:>15.1f}{100*ce:>14.1f}%{100*ff:>19.1f}%{window:>17.3f}")
	print("\nlearning window = (ceiling - 0.5) x (fraction NOT forced by propagation);")
	print("high only where signal exists AND deduction has not already settled it.")
	json.dump(rows, open(a.out, 'w'))
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
