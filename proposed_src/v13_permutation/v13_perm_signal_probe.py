"""Pre-RL sanity probe for v13: does the AHL column permutation actually matter?

The v13 plan is to learn a permutation policy for ahl_solve's randomized column
ordering. That is only worth building if permutations differ in success rate on
the SAME instance -- i.e. if some orderings solve an instance that others miss.
This probe measures that directly before any policy code is written:

  * per-instance success rate across many independent random permutations
  * whether the distribution is bimodal (instance-level easy/hard) or genuinely
    permutation-dependent (mid-range rates => headroom for a learned policy)

If nearly every instance is at 0% or 100%, the permutation carries no signal and
the whole v13 direction should be abandoned rather than dressed up.
"""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ


def ahl_try_with_perm(A, b, perm, block, N=1000):
	# single AHL/BKZ attempt with a caller-supplied column permutation
	m, n = A.shape
	Ap = A[:, perm]
	dim = n + 1
	M = IntegerMatrix(dim, n + 1 + m)
	for j in range(n):
		M[j, j] = 2
		for i in range(m):
			M[j, n + 1 + i] = int(N * Ap[i, j])
	for j in range(n):
		M[n, j] = 1
	M[n, n] = 1
	for i in range(m):
		M[n, n + 1 + i] = int(N * b[i])
	LLL.reduction(M)
	BKZ.reduction(M, BKZ.Param(block_size=min(block, dim)))
	for r in range(dim):
		v = np.array([M[r, c] for c in range(n + 1 + m)], dtype=np.int64)
		for sgn in (1, -1):
			w = sgn * v
			if np.any(w[n + 1:] != 0) or abs(w[n]) != 1:
				continue
			x = (w[:n] * w[n] * -1 + 1)
			if np.all((x == 0) | (x == 2)):
				xs = (x // 2).astype(np.int64)
				xo = np.empty(n, dtype=np.int64)
				xo[perm] = xs
				if np.array_equal(A.dot(xo), b):
					return True
	return False


def probe_one(task):
	idx, path, block, n_perms, seed = task
	with open(path) as f:
		d = json.load(f)
	if not d.get('feasible', True):
		return idx, None
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	n = A.shape[1]
	rng = np.random.default_rng(seed)
	hits = 0
	t0 = time.time()
	for t in range(n_perms):
		perm = np.arange(n) if t == 0 else rng.permutation(n)
		if ahl_try_with_perm(A, b, perm, block):
			hits += 1
	return idx, dict(instance=Path(path).name, n_perms=n_perms, hits=hits,
	                 rate=hits / n_perms, time=time.time() - t0)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--block', type=int, required=True)
	ap.add_argument('--n_perms', type=int, default=30)
	ap.add_argument('--limit', type=int, default=20)
	ap.add_argument('--jobs', type=int, default=8)
	ap.add_argument('--out_json', default=None)
	a = ap.parse_args()

	files = sorted(Path(a.data_dir).glob('*.json'))[:a.limit]
	tasks = [(i, str(f), a.block, a.n_perms, 1000 + i) for i, f in enumerate(files)]
	print(f"probing {len(tasks)} instances x {a.n_perms} permutations, block={a.block}", flush=True)

	out = []
	with Pool(a.jobs) as pool:
		for idx, rec in pool.imap_unordered(probe_one, tasks, chunksize=1):
			if rec is None:
				continue
			out.append(rec)
			print(f"  {rec['instance']}: {rec['hits']}/{rec['n_perms']} "
			      f"({100*rec['rate']:.0f}%)  {rec['time']:.1f}s", flush=True)

	rates = np.array([r['rate'] for r in out])
	n_zero = int((rates == 0).sum())
	n_full = int((rates == 1).sum())
	n_mid = len(rates) - n_zero - n_full
	print("\n=== permutation-signal summary ===")
	print(f"  feasible instances probed : {len(rates)}")
	print(f"  always fail   (rate = 0)  : {n_zero}")
	print(f"  always solve  (rate = 1)  : {n_full}")
	print(f"  permutation-dependent     : {n_mid}  <-- headroom for a learned policy")
	if len(rates):
		print(f"  mean rate                 : {rates.mean():.3f}")
		print(f"  rate quartiles            : {np.percentile(rates,[25,50,75])}")
	if a.out_json:
		with open(a.out_json, 'w') as f:
			json.dump(out, f, indent=2)
		print(f"saved: {a.out_json}")


if __name__ == '__main__':
	main()
