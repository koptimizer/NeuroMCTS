"""Sweep AHL/BKZ block size x instance to build a labeled dataset for a block-selection policy."""
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
from lattice_enum import ahl_solve
from difficulty_index import instance_metrics

BLOCK_GRID = [6, 8, 10, 15, 20, 25, 30, 40]
SAFE_BLOCKS = {
	'10x25': [6, 8, 10, 15, 20, 25, 30, 40],
	'20x50': [6, 8, 10, 15, 20, 25, 30, 40],
	'40x100': [6, 8, 10, 15, 20, 25, 30],
	'60x150': [6, 8, 10, 15, 20],
	'80x200': [6, 8, 10, 15],
}
TRIES = {'10x25': 3, '20x50': 8, '40x100': 5, '60x150': 3, '80x200': 3}


def features(A, b, seed):
	m, n = A.shape
	rng = np.random.default_rng(seed)
	vs, bt, lp_ok = instance_metrics(A, b, rng)
	density = float(A.mean())
	return dict(m=m, n=n, density=density, vertex_spread=vs, b_tightness=bt, lp_feasible=lp_ok)


def sweep_one(task):
	idx, path, size, seed = task
	with open(path) as f:
		d = json.load(f)
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	feat = features(A, b, seed)
	tries = TRIES[size]
	results = {}
	for block in SAFE_BLOCKS[size]:
		t0 = time.time()
		x = ahl_solve(A, b, block=block, tries=tries, seed=seed)
		dt = time.time() - t0
		solved = x is not None and bool(np.all(A.dot(x.astype(np.int64)) == b))
		results[block] = dict(solved=solved, time=dt)
	return idx, dict(path=str(path), size=size, feat=feat, results=results)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--jobs', type=int, default=16)
	ap.add_argument('--n_10x25', type=int, default=100)
	ap.add_argument('--n_20x50', type=int, default=100)
	ap.add_argument('--n_40x100', type=int, default=60)
	ap.add_argument('--n_60x150', type=int, default=30)
	ap.add_argument('--n_80x200', type=int, default=20)
	a = ap.parse_args()

	pools = {
		'10x25': ('../../instances/sub_hard_10x25_200', a.n_10x25),
		'20x50': ('../../instances/sub_hard_20x50_200', a.n_20x50),
		'40x100': ('../../instances/sub_hard_40x100_40', a.n_40x100),
		'60x150': ('../../instances/test_instances_hard_60x150_40', a.n_60x150),
		'80x200': ('../../instances/test_instances_hard_80x200_30', a.n_80x200),
	}

	tasks = []
	seed = 0
	for size, (dir_path, count) in pools.items():
		files = sorted(Path(dir_path).glob('*.json'))
		feas_files = []
		for fp in files:
			with open(fp) as f:
				d = json.load(f)
			if d.get('feasible', True):
				feas_files.append(fp)
		use = feas_files[:count]
		print(f"{size}: {len(use)}/{len(feas_files)} feasible instances queued", flush=True)
		for fp in use:
			tasks.append((seed, fp, size, seed))
			seed += 1

	t0 = time.time()
	out = [None] * len(tasks)
	with Pool(a.jobs) as pool:
		done = 0
		for idx, rec in pool.imap_unordered(sweep_one, tasks, chunksize=1):
			out[idx] = rec
			done += 1
			if done % 20 == 0:
				print(f"  {done}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)

	with open(a.out, 'w') as f:
		json.dump(out, f)
	print(f"done: {len(tasks)} instances, {time.time()-t0:.0f}s total -> {a.out}")


if __name__ == '__main__':
	main()
