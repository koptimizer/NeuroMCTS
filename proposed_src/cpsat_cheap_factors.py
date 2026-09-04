#!/usr/bin/env python3
"""
Supplements cpsat_difficulty_factors.py with structural features computable
directly from (A, b) at generation time -- no LP/CP-SAT solve needed except
where noted -- to see whether any of them predict CP-SAT effort as cheaply
as vertex_spread does. Regenerates the identical instances (same seed
formula) so results join by idx against the existing outcome data.

Usage:
  python cpsat_cheap_factors.py --size 10x25 --n 500 --out ../runs/cpsat_cheap_10x25.jsonl
"""
import argparse
import json
import numpy as np

import sys
sys.path.insert(0, '.')
from LPneuroBLS_v7 import gen_planted_instance


def features(A, b):
	m, n = A.shape
	row_deg = A.sum(axis=1).astype(np.float64)
	col_deg = A.sum(axis=0).astype(np.float64)
	row_tight = np.minimum(b, row_deg - b) / np.maximum(row_deg, 1)
	# mean pairwise Jaccard similarity between rows (cheap for small m)
	inter = A.astype(np.float64) @ A.T.astype(np.float64)
	rd = row_deg[:, None] + row_deg[None, :] - inter
	jac = np.divide(inter, rd, out=np.zeros_like(inter), where=rd > 0)
	iu = np.triu_indices(m, k=1)
	row_jaccard_mean = float(jac[iu].mean()) if len(iu[0]) else 0.0
	return dict(
		row_deg_mean=float(row_deg.mean()), row_deg_std=float(row_deg.std()),
		col_deg_mean=float(col_deg.mean()), col_deg_std=float(col_deg.std()),
		b_tightness_mean=float(row_tight.mean()), b_tightness_min=float(row_tight.min()),
		row_jaccard_mean=row_jaccard_mean,
	)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', default='10x25')
	ap.add_argument('--n', type=int, default=500)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	m, n = (int(v) for v in a.size.split('x'))
	with open(a.out, 'w') as f:
		for idx in range(a.n):
			rng = np.random.default_rng(1_000_000 + idx)
			A, b, x = gen_planted_instance(m, n, rng)
			feat = features(A.astype(np.int64), b.astype(np.int64))
			feat['idx'] = idx
			f.write(json.dumps(feat) + '\n')
	print(f"wrote {a.n} records to {a.out}")


if __name__ == '__main__':
	main()
