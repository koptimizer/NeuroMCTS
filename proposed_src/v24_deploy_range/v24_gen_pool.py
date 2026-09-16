"""v24: a 21x60 training pool for deployment-range supervision.

v23 located the network's entire margin over relaxation rounding at 50-60 free variables of a
21x60 instance, outside the 13-25 range of every training state. This pool exists so that
training states can be drawn from that regime. Seeds 800000+ are disjoint from the 10x25/18x50
training pools (900000+) and from every evaluation set (700000+). No enumeration here: the
planted solution is all that the state builder needs for prefixes.
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

from generate_hard_instances import gen_hard_feasible


def make_one(task):
	idx, mm, n, K = task
	rng = np.random.default_rng(idx)
	A, b, x, vs = gen_hard_feasible(mm, n, rng, K=K)
	return idx, dict(A=A.tolist(), b=b.tolist(), x=x.tolist(), seed=idx, vertex_spread=float(vs))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--m', type=int, default=21)
	ap.add_argument('--n', type=int, default=60)
	ap.add_argument('--count', type=int, default=500)
	ap.add_argument('--K', type=int, default=20)
	ap.add_argument('--seed0', type=int, default=800000)
	ap.add_argument('--test_frac', type=float, default=0.2)
	ap.add_argument('--jobs', type=int, default=12)
	ap.add_argument('--out_train', required=True)
	ap.add_argument('--out_test', required=True)
	a = ap.parse_args()
	tasks = [(a.seed0 + i, a.m, a.n, a.K) for i in range(a.count)]
	print(f"generating {a.count} instances at {a.m}x{a.n} (K={a.K}, seeds from {a.seed0})", flush=True)
	t0 = time.time()
	out = []
	with Pool(a.jobs) as pool:
		for idx, rec in pool.imap_unordered(make_one, tasks, chunksize=2):
			out.append(rec)
	out.sort(key=lambda r: r['seed'])
	ntr = int(len(out) * (1 - a.test_frac))
	json.dump(out[:ntr], open(a.out_train, 'w'))
	json.dump(out[ntr:], open(a.out_test, 'w'))
	print(f"saved train={ntr} test={len(out)-ntr} in {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
	main()
