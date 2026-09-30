"""v25 G1: instance-distribution variants of the market-split-style generator, to test whether the
learned arms transfer beyond the single distribution they were trained and evaluated on.

Variants (all 21x60 unless stated, 30 instances, seeds 750000+ disjoint from every other set):
  dense03   entry probability 0.3 (sparser rows), planting n/2, K=20
  dense07   entry probability 0.7 (denser rows),  planting n/2, K=20
  plant03   entry probability 0.5, planting 0.3 n ones,          K=20
  nosel     entry probability 0.5, planting n/2, K=1 (no vertex-spread selection)
Feasibility holds by construction. Output: instances/v25var_<name>/inst-XXXX.json
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, json, os
from multiprocessing import Pool
import numpy as np
from generate_hard_instances import vertex_spread

VARIANTS = dict(dense03=dict(p=0.3, ratio=0.5, K=20), dense07=dict(p=0.7, ratio=0.5, K=20), plant03=dict(p=0.5, ratio=0.3, K=20), nosel=dict(p=0.5, ratio=0.5, K=1))


def gen(m, n, rng, p, ratio, K):
	for _ in range(200):
		A = (rng.random((m, n)) < p).astype(np.int8)
		if A.sum(axis=1).min() < 2 or A.sum(axis=0).min() < 1:
			continue
		best, best_vs = None, -1.0
		for _ in range(K):
			x = np.zeros(n, dtype=np.int8); x[rng.choice(n, int(round(ratio * n)), replace=False)] = 1
			b = A.astype(np.int64).dot(x.astype(np.int64))
			vs = vertex_spread(A, b, rng, k=4) if K > 1 else 0.0
			if vs > best_vs:
				best_vs, best = vs, (A, b, x)
		return best[0], best[1], best[2], best_vs
	raise RuntimeError('exhausted')


def one(task):
	name, idx, m, n, seed = task
	v = VARIANTS[name]; rng = np.random.default_rng(seed)
	A, b, x, vs = gen(m, n, rng, v['p'], v['ratio'], v['K'])
	return name, idx, dict(A=A.tolist(), b=b.tolist(), x=x.tolist(), feasible=True, n_sols=-1, s_verified=False, variant=name, seed=seed, vertex_spread=float(vs))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--m', type=int, default=21); ap.add_argument('--n', type=int, default=60)
	ap.add_argument('--count', type=int, default=30); ap.add_argument('--seed0', type=int, default=750000)
	ap.add_argument('--variants', nargs='+', default=list(VARIANTS)); ap.add_argument('--jobs', type=int, default=6)
	a = ap.parse_args()
	tasks = [(name, i, a.m, a.n, a.seed0 + 1000 * k + i) for k, name in enumerate(a.variants) for i in range(a.count)]
	with Pool(a.jobs) as pool:
		for name, idx, rec in pool.imap_unordered(one, tasks):
			d = f'../../instances/v25var_{name}'; os.makedirs(d, exist_ok=True)
			json.dump(rec, open(f'{d}/inst-{idx:04d}.json', 'w'))
	for name in a.variants:
		print(name, 'done', a.count, 'instances ->', f'instances/v25var_{name}', flush=True)


if __name__ == '__main__':
	main()
