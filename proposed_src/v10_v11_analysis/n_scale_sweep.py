# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

#!/usr/bin/env python3
"""
Matched AHL embedding-scale (N) x BKZ block sweep for large sizes (60x150+),
where the fixed N=1000 tuned at 20x50/40x100 was never re-validated: as m
grows, more tail columns compete for the reduction's attention, so the
scale gap between the diagonal-2 rows and the b-columns may need to grow
with m to keep the target vector uniquely shortest.
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
	size, idx, blocks, Ns, tries = task
	mm, nn = size
	A, b, x = gen_planted_instance(mm, nn, np.random.default_rng(9_000_000 + idx))
	A64 = A.astype(np.int64); b64 = b.astype(np.int64)
	out = {'m': mm, 'n': nn, 'idx': idx}
	for N in Ns:
		for blk in blocks:
			t0 = time.time()
			sol = ahl_solve(A64, b64, block=blk, tries=tries, N=N, seed=idx)
			out[f'N{N}_blk{blk}_solved'] = sol is not None
			out[f'N{N}_blk{blk}_sec'] = time.time() - t0
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--size', default='60x150')
	ap.add_argument('--count', type=int, default=20)
	ap.add_argument('--tries', type=int, default=8)
	ap.add_argument('--blocks', type=int, nargs='+', default=[10, 15, 20])
	ap.add_argument('--Ns', type=int, nargs='+', default=[1000, 3000, 10000, 30000])
	ap.add_argument('--jobs', type=int, default=8)
	a = ap.parse_args()
	mm, nn = (int(v) for v in a.size.split('x'))
	tasks = [((mm, nn), i, a.blocks, a.Ns, a.tries) for i in range(a.count)]
	t0 = time.time()
	with Pool(a.jobs) as pool, open(a.out, 'w') as f:
		done = 0
		for rec in pool.imap_unordered(run, tasks, chunksize=1):
			f.write(json.dumps(rec) + '\n'); f.flush()
			done += 1
			print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
	recs = [json.loads(l) for l in open(a.out)]
	print(f"\n{a.size}: N x block sweep on {len(recs)} instances, tries={a.tries}")
	print(f"{'N':>8} {'block':>6} {'cov%':>7} {'avg_s':>8} {'med_s':>8}")
	for N in a.Ns:
		for blk in a.blocks:
			cov = np.mean([r[f'N{N}_blk{blk}_solved'] for r in recs]) * 100
			avg = np.mean([r[f'N{N}_blk{blk}_sec'] for r in recs])
			med = np.median([r[f'N{N}_blk{blk}_sec'] for r in recs])
			print(f"{N:>8} {blk:>6} {cov:>7.1f} {avg:>8.3f} {med:>8.3f}")


if __name__ == '__main__':
	main()
