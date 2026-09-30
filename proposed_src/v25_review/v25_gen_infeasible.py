"""v25 G3: infeasible 18x50 instances for the appendix experiment (refutation trees), built with the
legacy recipe gen_hard_infeasible (K=1 base, 1-3 rows perturbed by +-1, CP-SAT proves infeasibility
within the time limit). Output: instances/v25inf_18x50/inst-XXXX.json (feasible=False).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, json, os
from multiprocessing import Pool
import numpy as np
from generate_hard_instances import gen_hard_infeasible


def one(task):
	idx, m, n, seed, tl = task
	rng = np.random.default_rng(seed)
	A, b = gen_hard_infeasible(m, n, rng, time_limit=tl)
	return idx, dict(A=A.tolist(), b=b.tolist(), x=None, feasible=False, seed=seed)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--m', type=int, default=18); ap.add_argument('--n', type=int, default=50)
	ap.add_argument('--count', type=int, default=30); ap.add_argument('--seed0', type=int, default=760000)
	ap.add_argument('--time_limit', type=float, default=120.0); ap.add_argument('--jobs', type=int, default=6)
	a = ap.parse_args()
	d = f'../../instances/v25inf_{a.m}x{a.n}'; os.makedirs(d, exist_ok=True)
	with Pool(a.jobs) as pool:
		for idx, rec in pool.imap_unordered(one, [(i, a.m, a.n, a.seed0 + i, a.time_limit) for i in range(a.count)]):
			json.dump(rec, open(f'{d}/inst-{idx:04d}.json', 'w'))
	print('done', a.count, 'infeasible instances ->', d, flush=True)


if __name__ == '__main__':
	main()
