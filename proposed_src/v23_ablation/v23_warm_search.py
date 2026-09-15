"""Guided backtracking search with a persistent, warm-started node relaxation.

v20_guided_search.py rebuilds a SCIP model for every node relaxation. That is where most
of its 3.9 ms/node goes (4.3 ms measured at a 21x60 root, against 0.08 ms for a warm
re-optimization of the same LP), and it is the single largest reason the Python loop
processes 254 nodes/s where CP-SAT processes 19,782. This module keeps one LP over the
ORIGINAL variables for the whole search and expresses each node as a set of bound
changes, so consecutive nodes -- a child, or a sibling after backtracking -- differ by a
few bounds and the dual simplex resumes from the parent basis.

Everything else (bound check, propagation on the reduced instance, the network's graph
features) is unchanged from v20, so a difference in node counts between the two versions
can only come from the LP returning a different optimal vertex. With a zero objective the
relaxation is a pure feasibility problem and its solution is an arbitrary vertex of the
relaxation polytope; SCIP and Gurobi need not agree on which one. Both arms consume the
same vertex, so the comparison stays fair, but absolute node counts are not expected to
reproduce v20's exactly.
"""
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].iterdir())
                 if p.is_dir() and not p.name.startswith('.')]

import time
import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

import LPneuroBLS_v7 as m
from v23_warmstart_cost import features

torch.set_num_threads(1)


class WarmLP:
	"""One relaxation over all n original variables, re-optimized after bound changes."""

	def __init__(self, A, b, env):
		self.n = A.shape[1]
		self.mo = gp.Model(env=env)
		self.mo.Params.Method = 1                 # dual simplex: what bound changes want
		self.x = self.mo.addMVar(self.n, lb=0.0, ub=1.0)
		self.mo.addConstr(A.astype(float) @ self.x == b.astype(float))
		self.mo.setObjective(0)

	def set_node(self, fixed):
		# fixed: list of (original index, value). Everything else free in [0, 1].
		self.x.LB = 0.0
		self.x.UB = 1.0
		for j, v in fixed:
			self.x[j].LB = self.x[j].UB = float(v)
		self.mo.update()

	def solve(self):
		self.mo.optimize()
		return self.mo.Status == GRB.OPTIMAL

	def values(self, alive):
		return np.array(self.x.X, dtype=np.float64)[alive]

	def probe_infeasible(self, j, val):
		# True if forcing x_j = val (original index) makes the relaxation infeasible.
		lo, up = self.x[j].LB, self.x[j].UB
		self.x[j].LB = self.x[j].UB = float(val)
		self.mo.optimize()
		bad = self.mo.Status != GRB.OPTIMAL
		self.x[j].LB, self.x[j].UB = lo, up
		# gurobipy defers attribute changes until update(); without this the next
		# probe reads the pre-restore (fixed) bounds and fixings accumulate silently.
		self.mo.update()
		return bad

	def dispose(self):
		self.mo.dispose()


# guidance functions: (A_red, b_red, fixed, alive, warm, model, device, rng)
#   -> (local index to branch on | None, first value, list of (local idx, val) forced)

def g_random(A, b, fixed, alive, warm, model, device, rng):
	return int(rng.integers(A.shape[1])), int(rng.integers(2)), []


def g_lp(A, b, fixed, alive, warm, model, device, rng):
	warm.set_node(fixed)
	if not warm.solve():
		return None, None, []
	x = warm.values(alive)
	j = int(np.argmax(np.abs(x - 0.5)))
	return j, int(x[j] >= 0.5), []


def g_lp_probe(A, b, fixed, alive, warm, model, device, rng):
	warm.set_node(fixed)
	if not warm.solve():
		return None, None, []
	x = warm.values(alive)
	forced = []
	for loc, j in enumerate(alive):
		if warm.probe_infeasible(int(j), 0):
			forced.append((loc, 1))
		elif warm.probe_infeasible(int(j), 1):
			forced.append((loc, 0))
	if forced:
		return None, None, forced
	j = int(np.argmax(np.abs(x - 0.5)))
	return j, int(x[j] >= 0.5), []


def g_model(A, b, fixed, alive, warm, model, device, rng):
	warm.set_node(fixed)
	if not warm.solve():
		return None, None, []
	x = warm.values(alive)
	with torch.no_grad():
		p = torch.sigmoid(model(*features(A, b, x, device))).numpy()
	j = int(np.argmax(np.abs(p - 0.5)))
	return j, int(p[j] >= 0.5), []


GUIDES = dict(random=g_random, lp=g_lp, lp_probe=g_lp_probe, model=g_model)


def search(A0, b0, guide, model, device, time_limit, seed, env):
	"""DFS over reduced instances with a warm-started relaxation. Returns (solved, nodes, sec)."""
	rng = np.random.default_rng(seed)
	warm = WarmLP(A0, b0, env)
	t0 = time.time()
	nodes = 0
	stack = [(A0.copy(), b0.copy(), [], np.arange(A0.shape[1]))]
	try:
		while stack:
			if time.time() - t0 > time_limit:
				return False, nodes, time.time() - t0
			A, b, fixed, alive = stack.pop()
			nodes += 1
			if np.any(b < 0) or np.any(b > A.sum(axis=1)):
				continue
			live = A.sum(axis=1) > 0
			if not live.all():
				A, b = A[live], b[live]
				if A.shape[0] == 0:
					return True, nodes, time.time() - t0
			env_ = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
			env_.propagate_constraints()
			if env_.is_invalid:
				continue
			det = env_.assignment
			if np.all(det != -1):
				if np.all(A.dot(det.astype(np.int64)) == b):
					return True, nodes, time.time() - t0
				continue
			if np.any(det != -1):
				idx = np.where(det != -1)[0]
				vals = det[idx].astype(np.int64)
				keep = np.setdiff1d(np.arange(A.shape[1]), idx)
				stack.append((A[:, keep], b - A[:, idx].dot(vals),
				               fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep]))
				continue
			j, val, forced = guide(A, b, fixed, alive, warm, model, device, rng)
			if forced:
				idx = np.array([f[0] for f in forced])
				vals = np.array([f[1] for f in forced], dtype=np.int64)
				keep = np.setdiff1d(np.arange(A.shape[1]), idx)
				stack.append((A[:, keep], b - A[:, idx].dot(vals),
				               fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep]))
				continue
			if j is None:
				continue
			keep = np.setdiff1d(np.arange(A.shape[1]), [j])
			for v in (1 - val, val):          # preferred value pushed last, popped first
				stack.append((A[:, keep], b - A[:, j] * v,
				               fixed + [(int(alive[j]), int(v))], alive[keep]))
		return False, nodes, time.time() - t0
	finally:
		warm.dispose()
