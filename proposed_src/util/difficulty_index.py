#!/usr/bin/env python3
"""
5-level CP-SAT difficulty index built from the two confirmed drivers:
  (1) LP relaxation polytope size (vertex_spread: mean pairwise L2 distance
      among LP vertices found under 5 random objective directions)
  (2) b's slack (b_tightness_mean: how close each b_i sits to the center of
      its row's achievable range [0, rowsum_i], averaged over rows)
Score = average of each metric's percentile rank within a reference
population (of the same m x n size); percentile is used rather than raw
z-score so the two metrics (different units, different cost) combine on a
common 0-1 scale. Level = ceil(score_percentile * 5), 1=easiest, 5=hardest.
"""
import numpy as np
from pyscipopt import Model as ScipModel


def solve_lp_obj(A, b, c, tl=None):
	m, n = A.shape
	model = ScipModel(); model.hideOutput()
	if tl:
		model.setRealParam('limits/time', tl)
	xv = [model.addVar(vtype="C", lb=0.0, ub=1.0, name=f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.addCons(sum(xv[j] for j in idx) == int(b[i]))
	model.setObjective(sum(float(c[j]) * xv[j] for j in range(n)), "minimize")
	model.optimize()
	if model.getStatus() != "optimal":
		return None
	return np.array([model.getVal(v) for v in xv], dtype=np.float64)


def vertex_spread(A, b, rng, k=5):
	n = A.shape[1]
	verts = []
	for _ in range(k):
		c = rng.normal(0, 1, size=n)
		v = solve_lp_obj(A, b, c)
		if v is not None:
			verts.append(v)
	if len(verts) < 2:
		return 0.0, False
	uniq = []
	for v in verts:
		if not any(np.linalg.norm(v - u) < 1e-4 for u in uniq):
			uniq.append(v)
	if len(uniq) < 2:
		return 0.0, True
	dists = [np.linalg.norm(uniq[i] - uniq[j]) for i in range(len(uniq)) for j in range(i + 1, len(uniq))]
	return float(np.mean(dists)), True


def b_tightness_mean(A, b):
	row_deg = A.sum(axis=1).astype(np.float64)
	row_tight = np.minimum(b, row_deg - b) / np.maximum(row_deg, 1)
	return float(row_tight.mean())


def instance_metrics(A, b, rng):
	# Returns (vspread, btight, lp_feasible)
	vs, lp_ok = vertex_spread(A, b, rng)
	bt = b_tightness_mean(A, b)
	return vs, bt, lp_ok


def percentile_of(x, ref_sorted):
	# fraction of reference values <= x
	import bisect
	return bisect.bisect_right(ref_sorted, x) / len(ref_sorted)


def build_reference(vs_list, bt_list):
	return dict(vs_sorted=sorted(vs_list), bt_sorted=sorted(bt_list))


def score_level(vs, bt, ref):
	pv = percentile_of(vs, ref['vs_sorted'])
	pb = percentile_of(bt, ref['bt_sorted'])
	score = 0.5 * pv + 0.5 * pb
	level = min(5, int(score * 5) + 1)
	return level, score, pv, pb
