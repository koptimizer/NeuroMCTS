"""v24 Track 3 core: policy/value network on the MarginalNet trunk, and a recursive DFS that
records the exact cost of every branching decision.

Cost of a decision at node u = number of nodes expanded in u's subtree until a solution is found
below it or the subtree is exhausted. It is exact, deterministic given the policy, and available
from the search's own bookkeeping, which is what makes a cost-aware fine-tuning well posed here
where the v13/v14 attempts (sparse end-of-episode reward, no per-decision credit) were not.

The recursive DFS visits nodes in the same order as v23_warm_search.search (preferred value
first), counts the same nodes, and calls the same relaxation and propagation, so its node
counts must equal the iterative loop's; v24_r1_collect.py checks that on a few instances.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import LPneuroBLS_v7 as m
from v15_train_marginal import MarginalNet
from v23_warm_search import WarmLP
from v23_warmstart_cost import features

torch.set_num_threads(1)


class PolicyValueNet(nn.Module):
	"""MarginalNet trunk + marginal head (pretrained, kept) + policy head + value head.
	The policy logit is log|p_hat - 1/2| plus a zero-initialised correction, so the initial
	policy is exactly the supervised branching rule; the value head predicts log subtree cost."""

	def __init__(self, base, hidden=64):
		super().__init__()
		self.base = base
		self.pol = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
		nn.init.zeros_(self.pol[2].weight); nn.init.zeros_(self.pol[2].bias)
		self.val = nn.Sequential(nn.Linear(4 * hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

	def trunk(self, xv, xc, ev2c, ec2v):
		b = self.base
		hv = b.var_in(xv); hc = b.con_in(xc)
		for _ in range(b.layers):
			hc = b.ln_c(F.elu(b.gat_v2c((hv, hc), ev2c)) + hc)
			hv = b.ln_v(F.elu(b.gat_c2v((hc, hv), ec2v)) + hv)
		return hv, hc

	def forward(self, xv, xc, ev2c, ec2v):
		hv, hc = self.trunk(xv, xc, ev2c, ec2v)
		m_logit = self.base.head(hv).squeeze(-1)
		p = torch.sigmoid(m_logit)
		pol_logit = torch.log(torch.abs(p - 0.5) + 1e-6) + self.pol(hv).squeeze(-1)
		g = torch.cat([hv.mean(0), hv.max(0).values, hc.mean(0), hc.max(0).values,
		               torch.tensor([math.log(hv.shape[0]), math.log(hc.shape[0])])])
		return m_logit, pol_logit, self.val(g).squeeze(-1)


def load_pv(ckpt_base, ckpt_pv=None, device='cpu'):
	base = MarginalNet()
	base.load_state_dict(torch.load(ckpt_base, map_location=device, weights_only=False)['model_state_dict'])
	net = PolicyValueNet(base)
	if ckpt_pv:
		net.load_state_dict(torch.load(ckpt_pv, map_location=device, weights_only=False)['model_state_dict'])
	net.eval()
	return net


class Budget(Exception):
	pass


def dfs_collect(A0, b0, net, device, env, time_limit, tau=0.0, rng=None, record=True):
	"""Recursive DFS with the policy/value net as guide. Returns (found, nodes, records);
	each record is one branching decision with its exact subtree cost."""
	warm = WarmLP(A0, b0, env)
	t0 = time.time()
	cnt = [0]
	recs = []
	rng = rng or np.random.default_rng(0)

	def rec(A, b, fixed, alive, depth):
		if time.time() - t0 > time_limit:
			raise Budget
		cnt[0] += 1
		n = 1
		if np.any(b < 0) or np.any(b > A.sum(axis=1)):
			return False, n
		live = A.sum(axis=1) > 0
		if not live.all():
			A, b = A[live], b[live]
			if A.shape[0] == 0:
				return True, n
		e = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
		e.propagate_constraints()
		if e.is_invalid:
			return False, n
		det = e.assignment
		if np.all(det != -1):
			return bool(np.all(A.dot(det.astype(np.int64)) == b)), n
		if np.any(det != -1):
			idx = np.where(det != -1)[0]
			vals = det[idx].astype(np.int64)
			keep = np.setdiff1d(np.arange(A.shape[1]), idx)
			f, k = rec(A[:, keep], b - A[:, idx].dot(vals),
			           fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep], depth)
			return f, n + k
		warm.set_node(fixed)
		if not warm.solve():
			return False, n
		x = warm.values(alive)
		with torch.no_grad():
			m_logit, pol_logit, V = net(*features(A, b, x, device))
		p = torch.sigmoid(m_logit).numpy()
		if tau > 0:
			pr = torch.softmax(pol_logit / tau, 0).numpy()
			j = int(rng.choice(len(pr), p=pr / pr.sum()))
		else:
			j = int(torch.argmax(pol_logit))
		v = int(p[j] >= 0.5)
		keep = np.setdiff1d(np.arange(A.shape[1]), [j])
		found, ktot = False, 0
		for val in (v, 1 - v):
			f, k = rec(A[:, keep], b - A[:, j] * val, fixed + [(int(alive[j]), val)], alive[keep], depth + 1)
			ktot += k
			if f:
				found = True
				break
		if record:
			recs.append(dict(A=A.tolist(), b=b.tolist(), x_lp=x.tolist(), j=j, v=v,
			                 cost=n + ktot, found=found, depth=depth, n_free=int(A.shape[1])))
		return found, n + ktot

	try:
		found, _ = rec(A0.copy(), b0.copy(), [], np.arange(A0.shape[1]), 0)
	except Budget:
		found = False
	finally:
		warm.dispose()
	return found, cnt[0], recs
