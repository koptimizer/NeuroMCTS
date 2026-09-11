"""v13 pipeline: the v9 pipeline with every zero-contribution stage removed, plus a
learnable AHL column-permutation policy in place of uniform-random permutations.

Stages dropped from v9 (measured contribution 0% across 10x25..60x150 in this
project's own waterfall analyses -- see docs/version.md v10/v12):
  * GNN feasibility head as a decision rule ('declared_infeasible')
  * MCTS search
  * policy-guided DFS as a *solution finder*

Stages kept (measured nonzero contribution):
  * constraint propagation + probing        (infeasibility + free fixing)
  * LP hard rule                            (relaxation-based infeasibility)
  * kernel pump                             (constructive certificate)
  * AHL/BKZ lattice reduction               (the pipeline's actual workhorse)
  * complete backtracking as an infeasibility *prover* only

New in v13: ahl_solve_policy() draws its per-try column permutation from a learned
distribution over variables instead of uniform random. Motivation measured before
building it (v13_perm_signal_probe.py): at 20x50/block10, 8 of 10 feasible instances
are permutation-dependent (per-instance success rate 3-40%, none at 100%), so the
ordering genuinely decides the outcome and there is headroom to exploit.
"""
import numpy as np
import torch
from fpylll import IntegerMatrix, LLL, BKZ

import LPneuroBLS_v7 as m


def ahl_try_with_perm(A, b, perm, block, N=1000):
	# one AHL/BKZ attempt under a given column permutation; verified solutions only
	mm, n = A.shape
	Ap = A[:, perm]
	dim = n + 1
	M = IntegerMatrix(dim, n + 1 + mm)
	for j in range(n):
		M[j, j] = 2
		for i in range(mm):
			M[j, n + 1 + i] = int(N * Ap[i, j])
	for j in range(n):
		M[n, j] = 1
	M[n, n] = 1
	for i in range(mm):
		M[n, n + 1 + i] = int(N * b[i])
	LLL.reduction(M)
	BKZ.reduction(M, BKZ.Param(block_size=min(block, dim)))
	for r in range(dim):
		v = np.array([M[r, c] for c in range(n + 1 + mm)], dtype=np.int64)
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
					return xo
	return None


def sample_perm(scores, rng, temperature=1.0):
	# Plackett-Luce sampling: draw an ordering without replacement from per-variable
	# scores via the Gumbel top-k trick (equivalent to sequential softmax sampling)
	g = rng.gumbel(size=scores.shape[0])
	return np.argsort(-(scores / max(temperature, 1e-6) + g))


def perm_logprob(scores, perm, temperature=1.0):
	# log P(perm) under Plackett-Luce with the given scores (torch, differentiable)
	s = scores / max(temperature, 1e-6)
	ordered = s[perm]
	# log prod_t softmax over the remaining suffix = sum_t [s_t - logsumexp(s_t..s_end)]
	rev_lse = torch.logcumsumexp(ordered.flip(0), dim=0).flip(0)
	return (ordered - rev_lse).sum()


def ahl_solve_policy(A, b, scores, rng, block, tries, temperature=1.0, collect=False):
	"""Returns (solution|None, n_tries_used, trace).
	trace is a list of (perm, solved) when collect=True, for policy-gradient updates."""
	n = A.shape[1]
	trace = []
	for t in range(tries):
		perm = sample_perm(scores, rng, temperature)
		sol = ahl_try_with_perm(A, b, perm, block)
		if collect:
			trace.append((perm, sol is not None))
		if sol is not None:
			return sol, t + 1, trace
	return None, tries, trace


def solve_instance(A, b, scores, rng, block, tries, temperature=1.0,
                    pump_iters=300, dfs_budget=0, A_pinv=None, x_lp=None, lp_ok=None):
	"""v13 symbolic pipeline with a policy-driven AHL stage.
	Returns (assignment|None, pred_feasible|None, method)."""
	env = m.FastBinaryEnv(A, b)
	env.propagate_constraints().apply_probing()
	if env.is_invalid:
		return None, False, 'propagation_infeasible'

	if lp_ok is None:
		x_lp, lp_ok = m.solve_lp_relaxation(A, b)
	if not lp_ok:
		return None, False, 'lp_hard_rule'

	if A_pinv is None:
		A_pinv = np.linalg.pinv(A.astype(np.float64))
	x_pump, pumped, _ = m.kernel_pump_stats(x_lp, A, b, A_pinv, max_iter=pump_iters)
	if pumped:
		return np.asarray(x_pump), True, 'kernel_pump'

	sol, _, _ = ahl_solve_policy(A, b, scores, rng, block, tries, temperature)
	if sol is not None:
		return sol, True, 'ahl_bkz'

	if env.is_terminal():
		if env.get_accuracy() == 1.0:
			return env.assignment.copy(), True, 'propagation_solved'

	if dfs_budget > 0:
		# retained only as an infeasibility prover; a 'solved' return is accepted if it
		# happens but is not the reason this stage exists
		order = np.arange(A.shape[1])
		val_order = np.zeros(A.shape[1], dtype=np.int64)
		s, status, _ = m.policy_dfs(A, b, env.assignment.copy(), order, val_order,
		                             node_budget=dfs_budget)
		if status == 'solved':
			return s, True, 'dfs_solved'
		if status == 'infeasible':
			return None, False, 'dfs_proved_infeasible'

	return None, None, 'unresolved'
