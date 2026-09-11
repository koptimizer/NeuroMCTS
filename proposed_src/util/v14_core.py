"""Variable-fixing machinery shared by v14 training and benchmarking.

Framework: instead of trying to make AHL succeed on the full instance, fix k
variables and hand AHL a smaller one. Measured motivation (v14_fixing_probe.py):
fixing 20% of n correctly lifts AHL from 55% -> 100% at 20x50 and 20% -> 100% at
40x100, and WHICH variables are fixed barely matters -- only whether the values
are right. Constraint propagation fixes literally zero variables on these
instances (0/50 and 0/100 across 40 probed instances), so nothing cheaper than a
learned/heuristic guess is available to do the fixing.

A wrong fix makes the reduced instance unsatisfiable, which is why every fixing
source is wrapped in a retry loop: rejection is cheap (bound check is free, LP
check is ~3-10ms) compared to an AHL attempt (0.01-0.3s per try), so a weak
per-variable predictor still becomes usable through repeated sampling.
"""
import numpy as np
import torch

import LPneuroBLS_v7 as m
from v13_pipeline import ahl_try_with_perm


def reduce_instance(A, b, fixed_idx, fixed_val):
	# substitute the fixed assignment out: A_keep x_keep = b - A_fixed v
	keep = np.setdiff1d(np.arange(A.shape[1]), fixed_idx)
	b_red = b - A[:, fixed_idx].dot(np.asarray(fixed_val, dtype=np.int64)) if len(fixed_idx) else b.copy()
	return A[:, keep], b_red, keep


def bound_reject(A_red, b_red):
	# free necessary condition: every row must still be reachable by 0/1 values
	return bool(np.any(b_red < 0) or np.any(b_red > A_red.sum(axis=1)))


def lp_reject(A_red, b_red):
	# cheap necessary condition: the reduced LP relaxation must stay feasible
	if A_red.shape[1] == 0:
		return bool(np.any(b_red != 0))
	_, ok = m.solve_lp_relaxation(A_red.astype(np.int32), b_red.astype(np.int32))
	return not ok


def ahl_on_reduced(A_red, b_red, block, tries, rng):
	# run AHL on the reduced instance; returns the reduced-space solution or None
	n = A_red.shape[1]
	if n == 0:
		return np.array([], dtype=np.int64) if np.all(b_red == 0) else None
	for t in range(tries):
		perm = np.arange(n) if t == 0 else rng.permutation(n)
		sol = ahl_try_with_perm(A_red, b_red, perm, block)
		if sol is not None:
			return sol
	return None


def reconstruct(x_red, keep, fixed_idx, fixed_val, n):
	# stitch the reduced solution back together with the fixed assignment
	x = np.zeros(n, dtype=np.int64)
	x[keep] = x_red
	if len(fixed_idx):
		x[np.asarray(fixed_idx, dtype=np.int64)] = np.asarray(fixed_val, dtype=np.int64)
	return x


def lp_confidence_fixing(A, b, k, rng, x_lp=None, stochastic=False):
	# classical baseline: fix the k variables whose LP value is closest to 0/1
	if x_lp is None:
		x_lp, _ = m.solve_lp_relaxation(A, b)
	conf = -np.abs(x_lp - np.round(x_lp))
	sel = np.argsort(-conf)[:k]
	if stochastic:
		p = np.clip(x_lp[sel], 0.02, 0.98)
		vals = (rng.random(len(sel)) < p).astype(np.int64)
	else:
		vals = np.round(np.clip(x_lp[sel], 0, 1)).astype(np.int64)
	return sel, vals


def gnn_marginals(model, A, b, x_lp, device, mp_iters=16):
	# per-variable P(x_j = 1) from the pretrained assign head
	row_deg = np.maximum((A != 0).sum(axis=1), 1)
	col_deg = (A != 0).sum(axis=0) / max(1, A.shape[0])
	xlp_v = torch.tensor(np.stack([x_lp, col_deg], -1), dtype=torch.float, device=device)
	xlp_c = torch.tensor((b - A.dot(x_lp)) / row_deg, dtype=torch.float, device=device)
	rows, cols = np.where(A == 1)
	ev2c = torch.tensor(np.array([cols, rows]), dtype=torch.long, device=device)
	ec2v = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device)
	env = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
	env.propagate_constraints().apply_probing()
	xv, xc, cx = env.get_tensor_state(device)
	with torch.no_grad():
		_, assign_logits, _, _ = model(xv, xc, xlp_v, xlp_c, ev2c, ec2v,
		                                check_extra=cx, mp_iters=mp_iters)
	return torch.softmax(assign_logits, dim=-1)[:, 1].cpu().numpy()


def gnn_fixing(p1, k, rng, stochastic=True):
	# fix the k most confident variables of the pretrained head; sampling gives
	# the retry loop its diversity (argmax is the T=1 special case)
	conf = np.abs(p1 - 0.5)
	sel = np.argsort(-conf)[:k]
	if stochastic:
		p = np.clip(p1[sel], 0.02, 0.98)
		vals = (rng.random(len(sel)) < p).astype(np.int64)
	else:
		vals = (p1[sel] >= 0.5).astype(np.int64)
	return sel, vals


def topT_joint_fixings(p1, k, T):
	"""Enumerate the T most probable joint assignments of the k most confident
	variables, instead of drawing them i.i.d. An overconfident marginal makes i.i.d.
	sampling return near-identical retries (measured: a2_gnn wasted 10 attempts on
	effectively one guess), whereas this spends attempt t on the t-th most likely
	joint pattern. Exact by brute force since 2^k is at most 256 here."""
	conf = np.abs(p1 - 0.5)
	sel = np.argsort(-conf)[:k]
	ps = np.clip(p1[sel], 1e-6, 1 - 1e-6)
	combos = np.array(np.meshgrid(*[[0, 1]] * len(sel), indexing='ij')).reshape(len(sel), -1).T
	logp = (combos * np.log(ps) + (1 - combos) * np.log(1 - ps)).sum(axis=1)
	order = np.argsort(-logp)[:T]
	return sel, combos[order].astype(np.int64)


def solve_with_fixing(A, b, propose_fn, k, T, tries, block, rng, use_lp_reject=True):
	"""Retry loop: propose k fixings, reject cheaply if they break the instance,
	otherwise let AHL attack the reduced problem.
	Returns (solution|None, stats)."""
	n = A.shape[1]
	n_ahl_tries = n_bound_rej = n_lp_rej = 0
	for attempt in range(T):
		sel, vals = propose_fn(attempt, rng)
		A_red, b_red, keep = reduce_instance(A, b, sel, vals)
		if bound_reject(A_red, b_red):
			n_bound_rej += 1
			continue
		if use_lp_reject and lp_reject(A_red, b_red):
			n_lp_rej += 1
			continue
		x_red = ahl_on_reduced(A_red, b_red, block, tries, rng)
		n_ahl_tries += tries
		if x_red is not None:
			x = reconstruct(x_red, keep, sel, vals, n)
			if np.array_equal(A.dot(x), b):
				return x, dict(attempts=attempt + 1, ahl_tries=n_ahl_tries,
				                bound_rej=n_bound_rej, lp_rej=n_lp_rej)
	return None, dict(attempts=T, ahl_tries=n_ahl_tries,
	                   bound_rej=n_bound_rej, lp_rej=n_lp_rej)


def solve_plain(A, b, total_tries, block, rng):
	# control arm: no fixing at all, spend the whole budget on the full instance
	x = ahl_on_reduced(A, b, block, total_tries, rng)
	return x, dict(attempts=1, ahl_tries=total_tries, bound_rej=0, lp_rej=0)
