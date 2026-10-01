"""v25 B3: is the lattice stage of the paper (BKZ-12, 10 permutations) too weak? Sweep stronger
settings on the evaluation sets and record coverage and time per instance:
  block in {12, 20, 30, 40}, permutations in {10, 30}, plus a bounded lattice enumeration after
  BKZ (fpylll Enumeration on the reduced AHL basis, radius^2 = n+1, run in a subprocess with a
  wall-clock cap) which is the ingredient Wassermann (2025) relies on.
Output: runs/v25/lattice_strong.json and a printed table. Run from the repository root.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = ['proposed_src/util', 'proposed_src/v23_ablation', 'proposed_src']
import argparse, glob, json, time, zlib
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, GSO, Enumeration, EnumerationError
from v13_pipeline import ahl_try_with_perm


def basis(A, b, perm, N=1000):
	mm, n = A.shape; Ap = A[:, perm]; M = IntegerMatrix(n + 1, n + 1 + mm)
	for j in range(n):
		M[j, j] = 2
		for i in range(mm):
			M[j, n + 1 + i] = int(N * Ap[i, j])
	for j in range(n):
		M[n, j] = 1
	M[n, n] = 1
	for i in range(mm):
		M[n, n + 1 + i] = int(N * b[i])
	return M


def decode(v, A, b, perm):
	n = A.shape[1]
	for s in (1, -1):
		w = s * v
		if np.any(w[n + 1:] != 0) or abs(w[n]) != 1:
			continue
		x = (-w[:n] * w[n] + 1)
		if np.all((x == 0) | (x == 2)):
			xo = np.empty(n, dtype=np.int64); xo[perm] = x // 2
			if np.array_equal(A.dot(xo), b):
				return xo
	return None


def enum_worker(q, A, b, perm, block, max_sols):
	# BKZ-reduce, then enumerate every lattice vector of squared length <= n+1 (the solution length)
	n = A.shape[1]; M = basis(A, b, perm)
	LLL.reduction(M); BKZ.reduction(M, BKZ.Param(block_size=min(block, n + 1)))
	for r in range(n + 1):
		xo = decode(np.array([M[r, c] for c in range(M.ncols)], dtype=np.int64), A, b, perm)
		if xo is not None:
			q.put(('basis', xo.tolist())); return
	G = GSO.Mat(M); G.update_gso(); E = Enumeration(G, nr_solutions=max_sols)
	try:
		sols = E.enumerate(0, n + 1, n + 1 + 0.5, 0)
	except EnumerationError:
		q.put(('none', None)); return
	B = np.array([[M[r, c] for c in range(M.ncols)] for r in range(n + 1)], dtype=np.int64)
	for _, coeffs in sols:
		v = np.rint(np.array(coeffs)).astype(np.int64).dot(B)
		xo = decode(v, A, b, perm)
		if xo is not None:
			q.put(('enum', xo.tolist())); return
	q.put(('none', None))


def one(task):
	f, block, tries, enum_cap, seed_off = task
	d = json.load(open(f)); A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64); n = A.shape[1]
	rng = np.random.default_rng((zlib.crc32(_Path(f).name.encode()) + seed_off * 1_000_003) % (2 ** 31))
	t0 = time.time(); how = None
	for t in range(tries):
		perm = np.arange(n) if t == 0 else rng.permutation(n)
		if ahl_try_with_perm(A, b, perm, block) is not None:
			how = 'bkz'; break
	sec_bkz = time.time() - t0
	if how is None and enum_cap > 0:
		q = Queue(); p = Process(target=enum_worker, args=(q, A, b, np.arange(n), block, 200000)); p.start(); p.join(enum_cap)
		if p.is_alive():
			p.terminate(); p.join(); how = 'enum_timeout'
		else:
			how = q.get()[0] if not q.empty() else 'enum_none'
	return dict(inst=_Path(f).name, n=n, block=block, tries=tries, how=how, sec_bkz=round(sec_bkz, 2), sec_total=round(time.time() - t0, 1))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sets', nargs='+', default=['v22test_21x60', 'v24test_24x70', 'v24test_28x80'])
	ap.add_argument('--settings', nargs='+', default=['12x10', '20x10', '30x10', '40x10', '20x30', '40x30'], help='block x permutations')
	ap.add_argument('--enum_cap', type=float, default=300.0, help='seconds for the enumeration after the strongest BKZ setting (0 = off)')
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--out', default='runs/v25/lattice_strong.json')
	a = ap.parse_args()
	out = {}
	for s in a.sets:
		fs = sorted(glob.glob(f'instances/{s}/*.json'))
		for k, st in enumerate(a.settings):
			block, tries = map(int, st.split('x'))
			cap = a.enum_cap if k == len(a.settings) - 1 else 0.0
			with ProcessPoolExecutor(a.jobs) as ex:
				rs = list(ex.map(one, [(f, block, tries, cap, 0) for f in fs]))
			out[f'{s}|{st}'] = rs
			by = {}
			for r in rs: by[r['how']] = by.get(r['how'], 0) + 1
			print(f"{s:16s} block {block:2d} x {tries:2d} perms  solved by BKZ {by.get('bkz', 0)}/{len(rs)}  enum: {by}  median sec {np.median([r['sec_total'] for r in rs]):.1f}  max {max(r['sec_total'] for r in rs):.1f}", flush=True)
			json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
	main()
