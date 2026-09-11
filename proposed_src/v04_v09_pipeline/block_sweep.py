# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

#!/usr/bin/env python3
"""
Matched BKZ block-size sweep: the SAME instances solved at several block sizes
to build an honest coverage-vs-time Pareto (experiment 1), unconfounded by
instance-set differences.
"""
import argparse
import json
import time
import numpy as np
from multiprocessing import Pool
import sys
sys.path.insert(0, '.')
from lattice_enum import ahl_solve
from LPneuroBLS_v7 import gen_planted_instance


def run(task):
	size, idx, blocks, tries = task
	mm, nn = size
	A, b, x = gen_planted_instance(mm, nn, np.random.default_rng(7_000_000 + idx))
	A64 = A.astype(np.int64); b64 = b.astype(np.int64)
	out = {'m': mm, 'n': nn, 'idx': idx}
	for blk in blocks:
		t0 = time.time()
		sol = ahl_solve(A64, b64, block=blk, tries=tries, seed=idx)
		out[f'blk{blk}_solved'] = sol is not None
		out[f'blk{blk}_sec'] = time.time() - t0
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--size', default='20x50')
	ap.add_argument('--count', type=int, default=300)
	ap.add_argument('--tries', type=int, default=10)
	ap.add_argument('--blocks', type=int, nargs='+', default=[10, 20, 30, 40, 60])
	ap.add_argument('--jobs', type=int, default=8)
	a = ap.parse_args()
	mm, nn = (int(v) for v in a.size.split('x'))
	tasks = [((mm, nn), i, a.blocks, a.tries) for i in range(a.count)]
	t0 = time.time()
	with Pool(a.jobs) as pool, open(a.out, 'w') as f:
		done = 0
		for rec in pool.imap_unordered(run, tasks, chunksize=2):
			f.write(json.dumps(rec) + '\n'); f.flush()
			done += 1
			if done % 50 == 0:
				print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
	# summary
	recs = [json.loads(l) for l in open(a.out)]
	print(f"\n{a.size}: matched block sweep on {len(recs)} instances, tries={a.tries}")
	print(f"{'block':>6} {'cov%':>7} {'avg_s':>8} {'med_s':>8}")
	for blk in a.blocks:
		cov = np.mean([r[f'blk{blk}_solved'] for r in recs]) * 100
		avg = np.mean([r[f'blk{blk}_sec'] for r in recs])
		med = np.median([r[f'blk{blk}_sec'] for r in recs])
		print(f"{blk:>6} {cov:>7.1f} {avg:>8.3f} {med:>8.3f}")


if __name__ == '__main__':
	main()
