"""Lattice stage alone on the 21x60 100-instance, 24x70 and 28x80 sets, three seeds (260930).

Same protocol as v23_cascade_warm.py: block 12, 10 column permutations, per-instance seed
crc32(name) + seed_off * 1_000_003. Run from the repository root.
Output: runs/revision/lattice_only_60_70_80.json
"""
import sys, json, glob, time, zlib
sys.path[:0] = ['proposed_src/v23_ablation', 'proposed_src/util', 'proposed_src']
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from v23_cascade_warm import ahl_stage


def one(t):
	f, block, so = t
	d = json.load(open(f))
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	seed = (zlib.crc32(Path(f).name.encode()) + so * 1_000_003) % (2 ** 31)
	sol, sec = ahl_stage(A, b, block, 10, seed)
	return dict(inst=Path(f).name, seed_off=so, solved=sol is not None, sec=round(sec, 3))


if __name__ == '__main__':
	out = {}
	for name, block in [('v24test_21x60_100', 12), ('v24test_24x70', 12), ('v24test_28x80', 12)]:
		fs = sorted(glob.glob(f'instances/{name}/*.json'))
		for so in (0, 1, 2):
			t0 = time.time()
			with Pool(6) as p:
				rs = p.map(one, [(f, block, so) for f in fs])
			k = sum(r['solved'] for r in rs)
			secs = [r['sec'] for r in rs]
			print(f"{name:20s} seed{so}  lattice {k}/{len(rs)} ({100*k/len(rs):.1f}%)  median sec {np.median(secs):.3f}  max {max(secs):.2f}  wall {time.time()-t0:.0f}s", flush=True)
			out[f'{name}_S{so}'] = rs
	json.dump(out, open('runs/revision/lattice_only_60_70_80.json', 'w'), indent=1)
