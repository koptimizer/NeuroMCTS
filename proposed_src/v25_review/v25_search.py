"""v25: the warm-started depth-first loop with (a) the branching arms the reviews asked for
(value x variable cross ablation, classical CP rules, an exact-marginal oracle), (b) conflict
weights for dom/wdeg, and (c) a trace that splits the tree into the solution path and the refuted
subtrees hanging off it.

Node semantics are those of v23_warm_search.search / v24_rl_core.dfs_collect (bound check, drop
empty rows, propagation, warm-started LP, branch), written recursively so that every branching
decision knows the size of its subtree. Node counts equal the iterative loop's (checked in v24).

Arm = (variable rule, value rule):
  lp            most integral LP value,        round(x_lp)                (classical)
  model         argmax|p_hat-1/2|,             1[p_hat>=1/2]              (Stage 1)
  pv            argmax g_j,                    1[p_hat>=1/2]              (Stage 1+2)
  lpvar_s1val   most integral LP value,        1[p_hat>=1/2]
  s1var_lpval   argmax|p_hat-1/2|,             round(x_lp)
  pvvar_lpval   argmax g_j,                    round(x_lp)
  maxsd         counting-based (max row solution density b'_i/r_i), value of that density
  domwdeg       max conflict-weighted degree,  round(x_lp)
  impact        max propagation impact,        value with the smaller impact (sound fixing when a value propagates to contradiction)
  lp_probe      LP probing fixes what it can,  then the lp rule
  oracle        exact conditional marginals by enumeration (small instances only), rule of Stage 1
  random        random variable, random value (uniform branching)
  lpfrac        most fractional LP value (SCIP's mostinf),      round(x_lp)      (classical B&B rule)
  lpfracvar_s1val  most fractional LP value,                     1[p_hat>=1/2]    (P7: classical variable, marginal value)
  pv_frac       argmax g_j among variables with fractional LP value, 1[p_hat>=1/2] (P7: candidate restriction)
  pv_top5/10    argmax g_j among the 5/10 most fractional variables, 1[p_hat>=1/2] (P7: score fusion)
  bp            sum-product BP marginals (v25_bp, no learning; the inference baseline behind NSNet), rule of Stage 1
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import time
import numpy as np
import torch

import LPneuroBLS_v7 as m
from v23_warm_search import WarmLP
from v23_warmstart_cost import features
from v15_bayes_limit import enumerate_solutions

torch.set_num_threads(1)


class Budget(Exception):
	pass


def _lp(warm, fixed, alive):
	warm.set_node(fixed)
	if not warm.solve():
		return None
	return warm.values(alive)


def _marginal(model, A, b, x, device):
	if getattr(model, 'dense', False):               # v25_dense.DenseNet (or its TorchScript): (p_hat, g)
		from v25_dense import dense_features
		with torch.no_grad():
			p, g = model(*dense_features(A, b, x))
		return p.numpy(), (g.numpy() if g.numel() else None)
	with torch.no_grad():
		out = model(*features(A, b, x, device))
	if isinstance(out, tuple):                       # policy/value net: (m_logit, pol_logit, V)
		return torch.sigmoid(out[0]).numpy(), out[1].numpy()
	return torch.sigmoid(out).numpy(), None


# ---- guides: (A, b, fixed, alive, warm, model, ctx, rng) -> (j | None, value, forced list) ----
def g_random(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	return int(rng.integers(A.shape[1])), int(rng.integers(2)), []


def g_lp(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	j = int(np.argmax(np.abs(x - 0.5)))
	return j, int(x[j] >= 0.5), []


def g_lpfrac(A, b, fixed, alive, warm, model, ctx, rng):
	# most-fractional branching (SCIP's mostinf): the variable whose LP value is closest to 1/2
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	j = int(np.argmin(np.abs(x - 0.5)))
	return j, int(x[j] >= 0.5), []


def g_lpfracvar_s1val(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, _ = _marginal(model, A, b, x, ctx['device'])
	j = int(np.argmin(np.abs(x - 0.5)))
	return j, int(p[j] >= 0.5), []


def _pv_restricted(A, b, fixed, alive, warm, model, ctx, k):
	# Stage-1+2 policy restricted to fractional LP variables (k=None) or to the k most fractional ones
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, g = _marginal(model, A, b, x, ctx['device'])
	fr = np.abs(x - 0.5)
	if k is None:
		cand = np.flatnonzero(fr < 0.5 - 1e-6)
		if len(cand) == 0: cand = np.arange(len(x))
	else:
		cand = np.argsort(fr)[:min(k, len(x))]
	j = int(cand[int(np.argmax(g[cand]))])
	return j, int(p[j] >= 0.5), []


def g_pv_frac(A, b, fixed, alive, warm, model, ctx, rng): return _pv_restricted(A, b, fixed, alive, warm, model, ctx, None)
def g_pv_top5(A, b, fixed, alive, warm, model, ctx, rng): return _pv_restricted(A, b, fixed, alive, warm, model, ctx, 5)
def g_pv_top10(A, b, fixed, alive, warm, model, ctx, rng): return _pv_restricted(A, b, fixed, alive, warm, model, ctx, 10)


def g_model(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, _ = _marginal(model, A, b, x, ctx['device'])
	j = int(np.argmax(np.abs(p - 0.5)))
	return j, int(p[j] >= 0.5), []


def g_pv(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, g = _marginal(model, A, b, x, ctx['device'])
	j = int(np.argmax(g))
	return j, int(p[j] >= 0.5), []


def g_lpvar_s1val(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, _ = _marginal(model, A, b, x, ctx['device'])
	j = int(np.argmax(np.abs(x - 0.5)))
	return j, int(p[j] >= 0.5), []


def g_s1var_lpval(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, _ = _marginal(model, A, b, x, ctx['device'])
	j = int(np.argmax(np.abs(p - 0.5)))
	return j, int(x[j] >= 0.5), []


def g_pvvar_lpval(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, g = _marginal(model, A, b, x, ctx['device'])
	j = int(np.argmax(g))
	return j, int(x[j] >= 0.5), []


def g_maxsd(A, b, fixed, alive, warm, model, ctx, rng):
	# counting-based search (Pesant et al. 2012): on a 0/1 equality row the solution density of
	# x_j = 1 is exactly b_i / r_i for every variable of the row; branch on the variable-value pair
	# of maximum density over the rows, ties by the sum of densities.
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	r = np.maximum(A.sum(axis=1), 1)
	d1 = b / r                                            # density of value 1 per row
	D1 = A * d1[:, None]; D0 = A * (1 - d1)[:, None]      # rows x vars, zero where a_ij = 0
	best1 = D1.max(axis=0); best0 = D0.max(axis=0)
	s1 = best1 + 1e-6 * D1.sum(axis=0); s0 = best0 + 1e-6 * D0.sum(axis=0)
	if s1.max() >= s0.max():
		j = int(np.argmax(s1)); return j, 1, []
	j = int(np.argmax(s0)); return j, 0, []


def g_domwdeg(A, b, fixed, alive, warm, model, ctx, rng):
	# dom/wdeg (Boussemart et al. 2004): every variable has domain size 2, so the rule is the
	# maximum conflict-weighted degree; weights live in ctx['w'] indexed by original row and are
	# incremented by the search when a bound check or propagation discards a node.
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	w = ctx['w'][ctx['rows']] + 1.0                       # +1 = plain degree as the tie-breaker
	wdeg = A.T.dot(w)
	j = int(np.argmax(wdeg))
	return j, int(x[j] >= 0.5), []


def g_impact(A, b, fixed, alive, warm, model, ctx, rng):
	# impact-based search (Refalo 2004): impact of (j, v) = variables fixed by propagation after
	# fixing x_j = v; branch on the variable of largest total impact, try the value of smaller
	# impact first. A value whose propagation is contradictory is refuted, so its opposite is fixed.
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	n = A.shape[1]
	imp = np.zeros((n, 2)); forced = []
	for j in range(n):
		for v in (0, 1):
			keep = np.setdiff1d(np.arange(n), [j])
			e = m.FastBinaryEnv(A[:, keep].astype(np.int32), (b - A[:, j] * v).astype(np.int32))
			e.propagate_constraints()
			if e.is_invalid or np.any(b - A[:, j] * v < 0):
				imp[j, v] = n
				forced.append((j, 1 - v))
			else:
				imp[j, v] = 1 + int(np.sum(e.assignment != -1))
	if forced:
		seen = {}
		for j, v in forced:
			if j in seen and seen[j] != v:
				return None, None, []                      # both values refuted: dead node
			seen[j] = v
		return None, None, sorted(seen.items())
	j = int(np.argmax(imp.sum(axis=1)))
	return j, int(np.argmin(imp[j])), []


def g_lp_probe(A, b, fixed, alive, warm, model, ctx, rng):
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
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


def g_bp(A, b, fixed, alive, warm, model, ctx, rng):
	# loopy sum-product marginals of the reduced instance (damped, learning-free), Stage 1 rule
	from v25_bp import bp_marginals
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	p, _ = bp_marginals(A, b, ctx.get('bp_iters', 100), ctx.get('bp_damp', 0.9), 1e-4)
	j = int(np.argmax(np.abs(p - 0.5)))
	return j, int(p[j] >= 0.5), []


def g_oracle(A, b, fixed, alive, warm, model, ctx, rng):
	# exact conditional marginals of the reduced instance by enumeration (the Bayes-optimal value
	# rule, an upper bound for any learned marginal); used only where enumeration completes.
	x = _lp(warm, fixed, alive)
	if x is None: return None, None, []
	S, complete = enumerate_solutions(A, b, ctx.get('cap', 200000), ctx.get('enum_limit', 60.0))
	if not complete:
		p, _ = _marginal(model, A, b, x, ctx['device']) if model is not None else (np.clip(x, 0, 1), None)
	elif len(S) == 0:
		return None, None, []                             # no solution below: sound prune
	else:
		p = S.mean(axis=0)
	j = int(np.argmax(np.abs(p - 0.5)))
	return j, int(p[j] >= 0.5), []


GUIDES = dict(random=g_random, lp=g_lp, model=g_model, pv=g_pv, lpvar_s1val=g_lpvar_s1val, s1var_lpval=g_s1var_lpval,
              pvvar_lpval=g_pvvar_lpval, maxsd=g_maxsd, domwdeg=g_domwdeg, impact=g_impact, lp_probe=g_lp_probe, oracle=g_oracle, bp=g_bp, lpfrac=g_lpfrac,
              lpfracvar_s1val=g_lpfracvar_s1val, pv_frac=g_pv_frac, pv_top5=g_pv_top5, pv_top10=g_pv_top10)
NEEDS = dict(model='s1', lpvar_s1val='s1', s1var_lpval='s1', pv='pv', pvvar_lpval='pv', oracle='s1', lpfracvar_s1val='s1', pv_frac='pv', pv_top5='pv', pv_top10='pv')


def search(A0, b0, guide, model, time_limit, seed, env, ctx=None):
	"""Depth-first search; returns dict(solved, nodes, sec, path_len, dead_first, n_refuted, refuted_nodes).
	dead_first: list of (depth, cost) for path decisions whose first child was refuted; refuted_nodes:
	nodes spent in refuted subtrees (= nodes - path nodes)."""
	rng = np.random.default_rng(seed)
	ctx = dict(ctx or {}); ctx.setdefault('device', torch.device('cpu'))
	ctx['w'] = np.zeros(A0.shape[0])
	warm = WarmLP(A0, b0, env)
	t0 = time.time(); cnt = [0]; path = [0]; dead = []

	def rec(A, b, fixed, alive, rows, depth):
		if time.time() - t0 > time_limit:
			raise Budget
		cnt[0] += 1; n = 1
		bad = (b < 0) | (b > A.sum(axis=1))
		if bad.any():
			ctx['w'][rows[bad]] += 1
			return False, n
		live = A.sum(axis=1) > 0
		if not live.all():
			A, b, rows = A[live], b[live], rows[live]
			if A.shape[0] == 0:
				return True, n
		e = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
		e.propagate_constraints()
		if e.is_invalid:
			r = A.sum(axis=1); ctx['w'][rows[(b == 0) | (b == r)]] += 1
			return False, n
		det = e.assignment
		if np.all(det != -1):
			return bool(np.all(A.dot(det.astype(np.int64)) == b)), n
		if np.any(det != -1):
			idx = np.where(det != -1)[0]; vals = det[idx].astype(np.int64)
			keep = np.setdiff1d(np.arange(A.shape[1]), idx)
			f, k = rec(A[:, keep], b - A[:, idx].dot(vals), fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep], rows, depth)
			return f, n + k
		ctx['rows'] = rows
		j, val, forced = guide(A, b, fixed, alive, warm, model, ctx, rng)
		if forced:
			idx = np.array([f[0] for f in forced]); vals = np.array([f[1] for f in forced], dtype=np.int64)
			keep = np.setdiff1d(np.arange(A.shape[1]), idx)
			f, k = rec(A[:, keep], b - A[:, idx].dot(vals), fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep], rows, depth)
			return f, n + k
		if j is None:
			return False, n
		keep = np.setdiff1d(np.arange(A.shape[1]), [j])
		found, ktot = False, 0
		for t, v in enumerate((val, 1 - val)):
			f, k = rec(A[:, keep], b - A[:, j] * v, fixed + [(int(alive[j]), int(v))], alive[keep], rows, depth + 1)
			ktot += k
			if f:
				found = True
				if t == 1:
					dead.append((depth, ktot - k))          # first child refuted at cost ktot - k
				path[0] += 1
				break
		return found, n + ktot

	try:
		found, _ = rec(A0.copy(), b0.copy(), [], np.arange(A0.shape[1]), np.arange(A0.shape[0]), 0)
	except Budget:
		found = False
	finally:
		warm.dispose()
	sec = time.time() - t0
	refuted = sum(c for _, c in dead)
	return dict(solved=bool(found), nodes=cnt[0], sec=round(sec, 3), path_len=path[0],
	            dead_first=[(int(d), int(c)) for d, c in dead], n_refuted=len(dead), refuted_nodes=int(refuted))
