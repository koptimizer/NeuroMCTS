"""v24: the warm-started DFS of v23_warm_search with the ROOT decision overridable.

R0 needs the cost of every candidate root decision under an otherwise identical search, so
search() here takes root_action=(local index, first value); when given, the first branching
decision is that action and every later decision is the guide's. Everything else is the v23
loop verbatim (same propagation, same warm relaxation, same value ordering).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import time
import numpy as np

import LPneuroBLS_v7 as m
from v23_warm_search import WarmLP


def search(A0, b0, guide, model, device, time_limit, seed, env, root_action=None):
	rng = np.random.default_rng(seed)
	warm = WarmLP(A0, b0, env)
	t0 = time.time()
	nodes = 0
	first = True
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
			if first and root_action is not None:
				j, val, forced = int(root_action[0]), int(root_action[1]), []
			else:
				j, val, forced = guide(A, b, fixed, alive, warm, model, device, rng)
			first = False
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
			for v in (1 - val, val):
				stack.append((A[:, keep], b - A[:, j] * v,
				               fixed + [(int(alive[j]), int(v))], alive[keep]))
		return False, nodes, time.time() - t0
	finally:
		warm.dispose()
