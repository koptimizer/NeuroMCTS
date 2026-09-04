"""Append extra hard test instances to an existing test dir without regenerating what's already there."""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from generate_hard_instances import make_one, FEASIBLE_RATIO


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', required=True)
	ap.add_argument('--start_idx', type=int, required=True)
	ap.add_argument('--extra_n', type=int, required=True)
	ap.add_argument('--seed0', type=int, default=30_000_000)
	ap.add_argument('--jobs', type=int, default=16)
	ap.add_argument('--K', type=int, default=20)
	ap.add_argument('--time_limit', type=float, default=15.0)
	ap.add_argument('--out_dir', required=True)
	ap.add_argument('--prefix', required=True)
	a = ap.parse_args()
	m, n = (int(v) for v in a.size.split('x'))

	n_feasible = int(round(a.extra_n * FEASIBLE_RATIO))
	labels = [True] * n_feasible + [False] * (a.extra_n - n_feasible)
	rng_shuffle = np.random.default_rng(a.seed0 + 999_999)
	rng_shuffle.shuffle(labels)
	tasks = [(a.start_idx + i, m, n, a.seed0 + i, labels[i], a.K, a.time_limit) for i in range(a.extra_n)]

	out_dir = Path(a.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	t0 = time.time()
	done = 0
	with Pool(a.jobs) as pool:
		for i, rec in pool.imap_unordered(make_one, tasks, chunksize=1):
			path = out_dir / f"{a.prefix}-{i:05d}.json"
			with open(path, 'w') as f:
				json.dump(rec, f)
			done += 1
			print(f"  [{a.prefix}] +{done}/{a.extra_n} ({time.time()-t0:.0f}s)", flush=True)
	print(f"[{a.prefix}] extend done: {a.extra_n} instances, {time.time()-t0:.0f}s total", flush=True)


if __name__ == '__main__':
	main()
