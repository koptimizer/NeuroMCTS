"""Higher-tries re-sweep for 60x150/80x200 only, on the smallest (cheapest) safe blocks --
the base sweep found zero successes there at tries=3; check whether more tries surfaces signal."""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from lattice_enum import ahl_solve
from difficulty_index import instance_metrics

PLAN = {
	'60x150': dict(dir='../instances/test_instances_hard_60x150_40', n=30, blocks=[6, 8, 10], tries=20),
	'80x200': dict(dir='../instances/test_instances_hard_80x200_30', n=20, blocks=[6, 8], tries=20),
}


def features(A, b, seed):
	m, n = A.shape
	rng = np.random.default_rng(seed)
	vs, bt, lp_ok = instance_metrics(A, b, rng)
	density = float(A.mean())
	return dict(m=m, n=n, density=density, vertex_spread=vs, b_tightness=bt, lp_feasible=lp_ok)


def sweep_one(task):
	idx, path, size, seed, blocks, tries = task
	with open(path) as f:
		d = json.load(f)
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	feat = features(A, b, seed)
	results = {}
	for block in blocks:
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
	a = ap.parse_args()

	tasks = []
	idx_ctr = 0
	seed = 100000
	for size, cfg in PLAN.items():
		files = sorted(Path(cfg['dir']).glob('*.json'))
		feas = []
		for fp in files:
			with open(fp) as f:
				d = json.load(f)
			if d.get('feasible', True):
				feas.append(fp)
		use = feas[:cfg['n']]
		print(f"{size}: {len(use)} feasible instances, blocks={cfg['blocks']}, tries={cfg['tries']}", flush=True)
		for fp in use:
			tasks.append((idx_ctr, fp, size, seed, cfg['blocks'], cfg['tries']))
			idx_ctr += 1
			seed += 1

	t0 = time.time()
	out = [None] * len(tasks)
	with Pool(a.jobs) as pool:
		done = 0
		for idx, rec in pool.imap_unordered(sweep_one, tasks, chunksize=1):
			out[idx] = rec
			done += 1
			print(f"  {done}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)

	with open(a.out, 'w') as f:
		json.dump(out, f)
	print(f"done: {len(tasks)} instances, {time.time()-t0:.0f}s total -> {a.out}")


if __name__ == '__main__':
	main()
