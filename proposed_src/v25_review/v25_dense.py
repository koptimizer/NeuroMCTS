"""v25 A1: a dense-tensor re-implementation of the marginal / policy-value network.

The reduced instances are small (|K| <= 80 variables, m <= 28 rows), so the bipartite graph fits
in one dense 0/1 matrix and the PyG message passing (edge lists, scatter-softmax, per-node
gather) can be replaced by masked matrix products. The weights are the same tensors; the outputs
are verified to match the PyG network to float precision (verify()). Featurisation is also done
without building edge lists. Optionally TorchScript-compiled.

DenseNet(base_state_dict, pv_state_dict=None): forward(A, b, x_lp) -> (p_hat, g) with g = None
for a plain marginal network.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGAT(nn.Module):
	"""GATConv((64,64), 64, heads=1, add_self_loops=False) on a dense mask M (n_dst x n_src)."""

	def __init__(self, sd, prefix):
		super().__init__()
		self.W_src = nn.Parameter(sd[prefix + 'lin_src.weight'].clone())
		self.W_dst = nn.Parameter(sd[prefix + 'lin_dst.weight'].clone())
		self.a_src = nn.Parameter(sd[prefix + 'att_src'].reshape(-1).clone())
		self.a_dst = nn.Parameter(sd[prefix + 'att_dst'].reshape(-1).clone())
		self.bias = nn.Parameter(sd[prefix + 'bias'].clone())

	def forward(self, x_src, x_dst, mask):
		hs = x_src @ self.W_src.t()                       # n_src x 64
		hd = x_dst @ self.W_dst.t()
		logit = (hd @ self.a_dst)[:, None] + (hs @ self.a_src)[None, :]   # n_dst x n_src
		logit = F.leaky_relu(logit, 0.2)
		logit = logit.masked_fill(~mask, float('-inf'))
		att = torch.softmax(logit, dim=1)
		att = torch.nan_to_num(att, nan=0.0)              # rows without neighbours (never happens after row pruning)
		return att @ hs + self.bias


class DenseNet(nn.Module):
	def __init__(self, base_sd, pv_sd=None, layers=4):
		super().__init__()
		self.layers = layers
		self.var_in = nn.Linear(3, 64); self.con_in = nn.Linear(3, 64)
		self.var_in.weight.data = base_sd['var_in.weight'].clone(); self.var_in.bias.data = base_sd['var_in.bias'].clone()
		self.con_in.weight.data = base_sd['con_in.weight'].clone(); self.con_in.bias.data = base_sd['con_in.bias'].clone()
		self.gat_v2c = DenseGAT(base_sd, 'gat_v2c.'); self.gat_c2v = DenseGAT(base_sd, 'gat_c2v.')
		self.ln_v = nn.LayerNorm(64); self.ln_c = nn.LayerNorm(64)
		self.ln_v.weight.data = base_sd['ln_v.weight'].clone(); self.ln_v.bias.data = base_sd['ln_v.bias'].clone()
		self.ln_c.weight.data = base_sd['ln_c.weight'].clone(); self.ln_c.bias.data = base_sd['ln_c.bias'].clone()
		self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
		self.head[0].weight.data = base_sd['head.0.weight'].clone(); self.head[0].bias.data = base_sd['head.0.bias'].clone()
		self.head[2].weight.data = base_sd['head.2.weight'].clone(); self.head[2].bias.data = base_sd['head.2.bias'].clone()
		self.has_pol = pv_sd is not None
		self.pol = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))   # always present (TorchScript compiles both branches)
		if self.has_pol:
			self.pol[0].weight.data = pv_sd['pol.0.weight'].clone(); self.pol[0].bias.data = pv_sd['pol.0.bias'].clone()
			self.pol[2].weight.data = pv_sd['pol.2.weight'].clone(); self.pol[2].bias.data = pv_sd['pol.2.bias'].clone()
			# the pv checkpoint carries the (frozen) base too; use it so a fine-tuned trunk would be honoured
			for k in ['var_in.weight', 'var_in.bias', 'con_in.weight', 'con_in.bias', 'ln_v.weight', 'ln_v.bias', 'ln_c.weight', 'ln_c.bias', 'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias']:
				if 'base.' + k in pv_sd: dict(self.named_parameters())[k].data = pv_sd['base.' + k].clone()
			for g, pre in [(self.gat_v2c, 'base.gat_v2c.'), (self.gat_c2v, 'base.gat_c2v.')]:
				if pre + 'lin_src.weight' in pv_sd:
					g.W_src.data = pv_sd[pre + 'lin_src.weight'].clone(); g.W_dst.data = pv_sd[pre + 'lin_dst.weight'].clone()
					g.a_src.data = pv_sd[pre + 'att_src'].reshape(-1).clone(); g.a_dst.data = pv_sd[pre + 'att_dst'].reshape(-1).clone(); g.bias.data = pv_sd[pre + 'bias'].clone()
		self.eval()

	def forward(self, xv, xc, M):
		# xv: n x 3 variable features, xc: m x 3 row features, M: m x n bool mask (a_ij = 1)
		hv = self.var_in(xv); hc = self.con_in(xc)
		for _ in range(self.layers):
			hc = self.ln_c(F.elu(self.gat_v2c(hv, hc, M)) + hc)
			hv = self.ln_v(F.elu(self.gat_c2v(hc, hv, M.t())) + hv)
		m_logit = self.head(hv).squeeze(-1)
		p = torch.sigmoid(m_logit)
		if self.has_pol:
			g = torch.log(torch.abs(p - 0.5) + 1e-6) + self.pol(hv).squeeze(-1)
			return p, g
		return p, torch.zeros(0)


def dense_features(A, b, x_lp):
	"""Same six features as v23_warmstart_cost.features, from the dense 0/1 matrix."""
	mm, n = A.shape
	At = torch.as_tensor(A, dtype=torch.float32)
	bt = torch.as_tensor(b, dtype=torch.float32)
	x = torch.as_tensor(x_lp, dtype=torch.float32)
	col_deg = At.sum(0) / max(1, mm)
	frac = torch.abs(x - torch.round(x))
	r = torch.clamp(At.sum(1), min=1.0)
	xv = torch.stack([x, col_deg, frac], -1)
	xc = torch.stack([bt / r, r / max(1, n), (bt - At @ x) / r], -1)
	return xv, xc, At > 0.5


def load_dense(base_ckpt, pv_ckpt=None, script=False):
	base_sd = torch.load(base_ckpt, map_location='cpu', weights_only=False)['model_state_dict']
	pv_sd = torch.load(pv_ckpt, map_location='cpu', weights_only=False)['model_state_dict'] if pv_ckpt else None
	net = DenseNet(base_sd, pv_sd)
	if script:
		net = torch.jit.script(net)
	return net


def verify(base_ckpt, pv_ckpt, n_cases=20, seed=0):
	"""Max abs difference between the PyG networks and the dense one on random reduced instances."""
	from v15_train_marginal import MarginalNet
	from v24_rl_core import load_pv
	from v23_warmstart_cost import features
	import LPneuroBLS_v7 as m
	rng = np.random.default_rng(seed)
	s1 = MarginalNet(); s1.load_state_dict(torch.load(base_ckpt, map_location='cpu', weights_only=False)['model_state_dict']); s1.eval()
	pv = load_pv(base_ckpt, pv_ckpt)
	d1 = load_dense(base_ckpt); d2 = load_dense(base_ckpt, pv_ckpt)
	worst = 0.0
	for _ in range(n_cases):
		n = int(rng.integers(10, 61)); mm = int(rng.integers(4, 22))
		A = (rng.random((mm, n)) < 0.5).astype(np.int64)
		if A.sum(1).min() < 1 or A.sum(0).min() < 1: continue
		xs = (rng.random(n) < 0.5).astype(np.int64); b = A.dot(xs)
		x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
		if not ok: continue
		with torch.no_grad():
			f = features(A, b, x_lp, torch.device('cpu'))
			p_ref = torch.sigmoid(s1(*f)); ml, gl, _ = pv(*f)
			xv, xc, M = dense_features(A, b, x_lp)
			p1, _ = d1(xv, xc, M); p2, g2 = d2(xv, xc, M)
		worst = max(worst, float((p1 - p_ref).abs().max()), float((p2 - torch.sigmoid(ml)).abs().max()), float((g2 - gl).abs().max()))
	return worst


def bench(base_ckpt, pv_ckpt, sizes=((21, 60), (18, 50), (10, 25)), reps=50, seed=0):
	"""Per-forward time (ms): PyG featurise+forward vs dense featurise+forward vs dense scripted."""
	from v15_train_marginal import MarginalNet
	from v24_rl_core import load_pv
	from v23_warmstart_cost import features
	import LPneuroBLS_v7 as m
	torch.set_num_threads(1)
	rng = np.random.default_rng(seed)
	pv = load_pv(base_ckpt, pv_ckpt); dn = load_dense(base_ckpt, pv_ckpt); ds = load_dense(base_ckpt, pv_ckpt, script=True)
	out = {}
	for mm, n in sizes:
		A = (rng.random((mm, n)) < 0.5).astype(np.int64); xs = (rng.random(n) < 0.5).astype(np.int64); b = A.dot(xs)
		x_lp, _ = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
		res = {}
		with torch.no_grad():
			for name, fn in [('pyg', lambda: pv(*features(A, b, x_lp, torch.device('cpu')))), ('dense', lambda: dn(*dense_features(A, b, x_lp))), ('dense_script', lambda: ds(*dense_features(A, b, x_lp)))]:
				for _ in range(5): fn()
				t0 = time.perf_counter()
				for _ in range(reps): fn()
				res[name] = round(1000 * (time.perf_counter() - t0) / reps, 3)
		out[f'{mm}x{n}'] = res
	return out


if __name__ == '__main__':
	base = '../../runs/v22/model_n25/conditional.pt'; pvc = '../../runs/v24/r2/pv_best.pt'
	print('max abs diff vs PyG:', verify(base, pvc))
	print('forward ms:', bench(base, pvc))
