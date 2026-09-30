"""Why the lattice stage fails with size (260930 paper, Fig. lattice): for every evaluation instance,
reduce the AHL basis once (identity permutation, LLL then BKZ block 12, N = 1000 as in v13_pipeline)
and record which short vectors the reduced basis contains: solution-type rows (constraint part 0,
indicator +-1, squared length n+1) and kernel-type rows (constraint part 0, indicator 0, A z = 0).
Run from the repository root. Output: runs/revision/lattice_mech.json
"""
import sys, json, glob, time
sys.path[:0] = ['proposed_src/util', 'proposed_src']
from multiprocessing import Pool
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ


def one(f):
	d = json.load(open(f))
	A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
	mm, n = A.shape; N = 1000; dim = n + 1
	M = IntegerMatrix(dim, n + 1 + mm)
	for j in range(n):
		M[j, j] = 2
		for i in range(mm):
			M[j, n + 1 + i] = int(N * A[i, j])
	for j in range(n):
		M[n, j] = 1
	M[n, n] = 1
	for i in range(mm):
		M[n, n + 1 + i] = int(N * b[i])
	t0 = time.time()
	LLL.reduction(M)
	BKZ.reduction(M, BKZ.Param(block_size=min(12, dim)))
	sec = time.time() - t0
	sol_rows, ker_len = 0, []
	for r in range(dim):
		v = np.array([M[r, c] for c in range(n + 1 + mm)], dtype=np.int64)
		if np.any(v[n + 1:] != 0):
			continue
		if abs(v[n]) == 1:
			x = (-v[:n] * v[n] + 1)
			if np.all((x == 0) | (x == 2)) and np.array_equal(A.dot(x // 2), b):
				sol_rows += 1
		elif v[n] == 0:
			ker_len.append(int((v[:n] ** 2).sum()))
	return dict(inst=f.split('/')[-1], n=int(n), m=int(mm), sec=round(sec, 3), sol_rows=sol_rows, n_kernel=len(ker_len),
	            kernel_min=min(ker_len) if ker_len else None, kernel_len=sorted(ker_len)[:5], sol_len=n + 1)


if __name__ == '__main__':
	out = {}
	for name in ['v22test_10x25', 'v22test_18x50', 'v22test_21x60', 'v24test_24x70', 'v24test_28x80']:
		fs = sorted(glob.glob(f'instances/{name}/*.json'))
		with Pool(6) as p:
			rs = p.map(one, fs)
		out[name] = rs
		km = [r['kernel_min'] for r in rs if r['kernel_min'] is not None]
		print(f"{name:16s} n+1={rs[0]['sol_len']:3d}  solution rows>0: {sum(r['sol_rows']>0 for r in rs)}/{len(rs)}  "
		      f"kernel rows median {np.median([r['n_kernel'] for r in rs]):.0f}  shortest kernel len^2 median {np.median(km):.0f} (min {min(km)}, max {max(km)})  "
		      f"median sec {np.median([r['sec'] for r in rs]):.2f}", flush=True)
	json.dump(out, open('runs/revision/lattice_mech.json', 'w'), indent=1)
	print('saved runs/revision/lattice_mech.json')
