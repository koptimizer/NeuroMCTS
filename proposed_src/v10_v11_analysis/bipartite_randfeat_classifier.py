# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

#!/usr/bin/env python3
"""
Bipartite (variable/constraint) GNN feasibility classifier with an optional
random-node-feature injection, testing the exact fix Chen et al. (ICLR 2023,
"On Representing MILP by GNNs") prescribe for the WL-indistinguishability
limit: i.i.d. Gaussian features appended to every node, redrawn on every
forward pass, break the symmetry that makes some feasible/infeasible pairs
provably indistinguishable to a standard GNN. --random_feat_dim 0 reproduces
the standard (no-fix) architecture as the control condition.

Uses the same bipartite var/constraint representation as the project's
production BipartiteGNN and Chen et al.'s own theorem (not the dual-literal
NeuroSAT representation used in dual_node_classifier.py, which the theorem
does not directly address). Trained and evaluated on the matched-pair
dataset from collect_paired_signal.py for direct comparison against the
dual-node null (AUC ~0.500) and the frozen-embedding null (AUC 0.497,
eval_feas_classifier.py).

Usage:
  python bipartite_randfeat_classifier.py --data ../../runs/paired_signal_20x50.jsonl \
      --epochs 60 --random_feat_dim 8
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

import sys
sys.path.insert(0, '.')
from LPneuroBLS_v7 import solve_lp_relaxation


class BipartiteFeasGNN(nn.Module):
	# pool='mean' replicates the prior (concentration-of-measure-limited) null;
	# 'max'/'meanmax' test whether extreme-node statistics (the axis that
	# actually carried signal in the project's own pump/AHL scalar features,
	# e.g. ahl_min_norm_min) survive where the mean washes out
	def __init__(self, hidden_dim=64, num_layers=8, random_feat_dim=0, pool='mean'):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.random_feat_dim = random_feat_dim
		self.pool = pool
		pool_mult = 2 if pool == 'meanmax' else 1
		self.var_proj = nn.Linear(2 + random_feat_dim, hidden_dim)
		self.constr_proj = nn.Linear(2 + random_feat_dim, hidden_dim)
		self.var_msg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
		self.constr_msg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
		self.constr_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
		self.var_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
		self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2 * pool_mult, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

	def _pool(self, h, batch, dim_size):
		if self.pool == 'mean':
			return scatter(h, batch, dim=0, dim_size=dim_size, reduce='mean')
		if self.pool == 'max':
			return scatter(h, batch, dim=0, dim_size=dim_size, reduce='max')
		if self.pool == 'meanmax':
			mp = scatter(h, batch, dim=0, dim_size=dim_size, reduce='mean')
			xp = scatter(h, batch, dim=0, dim_size=dim_size, reduce='max')
			return torch.cat([mp, xp], dim=-1)
		raise ValueError(self.pool)

	def forward(self, var_feat, constr_feat, edge_v2c, var_batch, constr_batch, device, freeze_rand=None):
		if self.random_feat_dim > 0:
			if freeze_rand is None:
				rv = torch.randn(var_feat.size(0), self.random_feat_dim, device=device)
				rc = torch.randn(constr_feat.size(0), self.random_feat_dim, device=device)
			else:
				rv, rc = freeze_rand
			var_feat = torch.cat([var_feat, rv], dim=1)
			constr_feat = torch.cat([constr_feat, rc], dim=1)
		n_v, n_c = var_feat.size(0), constr_feat.size(0)
		v = self.var_proj(var_feat)
		c = self.constr_proj(constr_feat)
		v_h = torch.zeros_like(v); v_c = torch.zeros_like(v)
		c_h = torch.zeros_like(c); c_c = torch.zeros_like(c)
		src, dst = edge_v2c[0], edge_v2c[1]
		for _ in range(self.num_layers):
			msg = self.var_msg(v)[src]
			agg_c = scatter(msg, dst, dim=0, dim_size=n_c, reduce='sum')
			c, c_c = self.constr_lstm(agg_c, (c, c_c))
			msg2 = self.constr_msg(c)[dst]
			agg_v = scatter(msg2, src, dim=0, dim_size=n_v, reduce='sum')
			v, v_h = self.var_lstm(agg_v, (v, v_h))
		pool_v = self._pool(v, var_batch, int(var_batch.max()) + 1)
		pool_c = self._pool(c, constr_batch, int(constr_batch.max()) + 1)
		logit = self.classifier(torch.cat([pool_v, pool_c], dim=-1)).squeeze(-1)
		return logit


def build_graph(A, b, device):
	m, n = A.shape
	col_deg = ((A != 0).sum(axis=0) / max(1, m)).astype(np.float32)
	row_deg = np.maximum((A != 0).sum(axis=1), 1)
	b_norm = (b / row_deg).astype(np.float32)
	x_lp, _ = solve_lp_relaxation(A, b)
	lp_resid = ((b - A.dot(x_lp)) / row_deg).astype(np.float32)
	var_feat = np.stack([x_lp.astype(np.float32), col_deg], axis=1)
	constr_feat = np.stack([b_norm, lp_resid], axis=1)
	rows, cols = np.where(A != 0)
	edge_v2c = np.stack([cols, rows])
	return (torch.tensor(var_feat, dtype=torch.float, device=device),
	        torch.tensor(constr_feat, dtype=torch.float, device=device),
	        torch.tensor(edge_v2c, dtype=torch.long, device=device))


def batch_graphs(items, device):
	# items are precomputed (var_feat, constr_feat, edge_v2c) tuples (see precompute_graphs)
	vf, cf, ev, vb, cb, off_v, off_c = [], [], [], [], [], 0, 0
	for i, (v_i, c_i, e_i) in enumerate(items):
		vf.append(v_i); cf.append(c_i)
		ev.append(e_i + torch.tensor([[off_v], [off_c]], device=device))
		vb.append(torch.full((v_i.size(0),), i, dtype=torch.long, device=device))
		cb.append(torch.full((c_i.size(0),), i, dtype=torch.long, device=device))
		off_v += v_i.size(0); off_c += c_i.size(0)
	return (torch.cat(vf), torch.cat(cf), torch.cat(ev, dim=1), torch.cat(vb), torch.cat(cb))


def precompute_graphs(recs, device):
	# LP relaxation + structural features are deterministic per instance, so
	# cache them once instead of resolving the LP every epoch/eval-draw
	out = []
	for r in recs:
		A = np.array(r['A'], dtype=np.int64); b = np.array(r['b'], dtype=np.int64)
		out.append(build_graph(A, b, device))
	return out


def auc(y, x):
	from scipy.stats import rankdata
	pos = x[y == 1]; neg = x[y == 0]
	r = rankdata(np.concatenate([pos, neg]))
	return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data', required=True)
	ap.add_argument('--epochs', type=int, default=60)
	ap.add_argument('--hidden_dim', type=int, default=64)
	ap.add_argument('--num_layers', type=int, default=8)
	ap.add_argument('--batch_size', type=int, default=64)
	ap.add_argument('--random_feat_dim', type=int, default=0)
	ap.add_argument('--eval_draws', type=int, default=16,
	                 help='Average sigmoid(logit) over this many independent random-feature draws at eval')
	ap.add_argument('--val_frac', type=float, default=0.2)
	ap.add_argument('--pool', choices=['mean', 'max', 'meanmax'], default='mean')
	a = ap.parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	recs = [json.loads(l) for l in open(a.data)]
	print(f"loaded {len(recs)} instances, random_feat_dim={a.random_feat_dim}, pool={a.pool}")
	rng = np.random.default_rng(0)
	pair_ids = sorted(set(r['pair_id'] for r in recs))
	rng.shuffle(pair_ids)
	nval = int(len(pair_ids) * a.val_frac)
	val_ids = set(pair_ids[:nval])
	train_recs = [r for r in recs if r['pair_id'] not in val_ids]
	val_recs = [r for r in recs if r['pair_id'] in val_ids]
	print(f"train {len(train_recs)} / val {len(val_recs)} instances ({len(pair_ids)-nval}/{nval} pairs)")

	print("precomputing LP relaxations + graphs (cached once)...")
	train_y = np.array([1.0 if r['gt_feasible'] else 0.0 for r in train_recs])
	val_y = np.array([1.0 if r['gt_feasible'] else 0.0 for r in val_recs])
	train = list(zip(precompute_graphs(train_recs, device), train_y))
	val = list(zip(precompute_graphs(val_recs, device), val_y))
	print("done.")

	model = BipartiteFeasGNN(a.hidden_dim, a.num_layers, a.random_feat_dim, a.pool).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

	def run_epoch(data, train_mode, eval_draws=1):
		model.train(train_mode)
		idx = np.random.permutation(len(data)) if train_mode else np.arange(len(data))
		total_loss = 0.0
		all_p, all_y = [], []
		for s in range(0, len(idx), a.batch_size):
			chunk = [data[i] for i in idx[s:s + a.batch_size]]
			items = [g for g, _ in chunk]
			y = torch.tensor([float(yy) for _, yy in chunk], device=device)
			vf, cf, ev, vb, cb = batch_graphs(items, device)
			if train_mode:
				logit = model(vf, cf, ev, vb, cb, device)
				loss = F.binary_cross_entropy_with_logits(logit, y)
				opt.zero_grad(); loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
				opt.step()
				p = torch.sigmoid(logit).detach()
			else:
				with torch.no_grad():
					ps = [torch.sigmoid(model(vf, cf, ev, vb, cb, device)) for _ in range(eval_draws)]
					p = torch.stack(ps, dim=0).mean(dim=0)
					loss = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y)
			total_loss += loss.item() * len(chunk)
			all_p.append(p.cpu().numpy())
			all_y.append(y.cpu().numpy())
		p = np.concatenate(all_p); yy = np.concatenate(all_y)
		return total_loss / len(data), auc(yy, p), p, yy

	best_auc = 0.0
	for ep in range(1, a.epochs + 1):
		tr_loss, tr_auc, _, _ = run_epoch(train, True)
		if ep % 5 == 0 or ep == a.epochs:
			val_loss, val_auc, pv, yv = run_epoch(val, False, eval_draws=a.eval_draws)
			mh = pv[yv == 1].mean() if (yv == 1).any() else 0
			mm = pv[yv == 0].mean() if (yv == 0).any() else 0
			best_auc = max(best_auc, val_auc)
			print(f"ep {ep:>3} | train loss {tr_loss:.4f} auc {tr_auc:.4f} | "
			      f"val loss {val_loss:.4f} auc {val_auc:.4f} | p(hit|feas)={mh:.3f} p(hit|infeas)={mm:.3f}", flush=True)
	print(f"random_feat_dim={a.random_feat_dim} pool={a.pool}  BEST val AUC = {best_auc:.4f}")


if __name__ == '__main__':
	main()
