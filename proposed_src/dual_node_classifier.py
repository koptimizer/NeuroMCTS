#!/usr/bin/env python3
"""
NeuroSAT-inspired dual-node feasibility classifier for binary linear systems.

Unlike the project's main BipartiteGNN (one node per variable), each variable
j gets TWO "hypothesis" nodes -- one representing x_j=0, one x_j=1 -- linked
by a dedicated complement edge, mirroring NeuroSAT's literal/negation-edge
design. Constraint nodes connect to BOTH hypothesis nodes of every variable
they involve (unlike SAT clauses, which see only one polarity per variable,
a linear constraint's participation doesn't depend on the hypothesized
value, so the coefficient sign is not encoded structurally and must instead
be supplied numerically). Since (A, b) together define the problem -- unlike
CNF clauses, whose membership edges fully determine the formula -- every
constraint node is seeded with its row's target value (b_i / row degree);
this is the minimal information a feasibility classifier can possibly need,
analogous to NeuroSAT's structure-only clause embedding.

Trained end-to-end (BCE) on the matched-pair dataset from
collect_paired_signal.py: two variants are compared, --lp_features off
(structure + b only, the NeuroSAT-faithful minimal setup) and on (adds LP
relaxation values as a stronger prior).

Usage:
  python dual_node_classifier.py --data ../runs/paired_signal_20x50.jsonl \
      --epochs 60 --lp_features 0
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


class DualNodeGNN(nn.Module):
	def __init__(self, hidden_dim=64, num_layers=16, lp_features=False):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.lp_features = lp_features
		hyp_in = 1 + (1 if lp_features else 0)  # polarity bit (+ optional LP value)
		self.hyp_proj = nn.Linear(hyp_in, hidden_dim)
		constr_in = 1 + (1 if lp_features else 0)  # b_i/row_deg (+ optional LP residual)
		self.constr_proj = nn.Linear(constr_in, hidden_dim)
		self.hyp_msg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
		self.constr_msg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
		self.constr_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
		self.hyp_lstm = nn.LSTMCell(hidden_dim * 2, hidden_dim)
		self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

	def forward(self, hyp_feat, constr_feat, edge_h2c, complement_idx, hyp_batch, constr_batch):
		n_hyp, n_c = hyp_feat.size(0), constr_feat.size(0)
		h = self.hyp_proj(hyp_feat)
		c = self.constr_proj(constr_feat)
		h_h = torch.zeros_like(h); h_c = torch.zeros_like(h)
		c_h = torch.zeros_like(c); c_c = torch.zeros_like(c)
		src, dst = edge_h2c[0], edge_h2c[1]
		for _ in range(self.num_layers):
			msg = self.hyp_msg(h)[src]
			agg_c = scatter(msg, dst, dim=0, dim_size=n_c, reduce='sum')
			c, c_c = self.constr_lstm(agg_c, (c, c_c))
			msg2 = self.constr_msg(c)[dst]
			agg_h = scatter(msg2, src, dim=0, dim_size=n_hyp, reduce='sum')
			comp = h[complement_idx]
			h, h_h = self.hyp_lstm(torch.cat([agg_h, comp], dim=-1), (h, h_h))
		pool_h = scatter(h, hyp_batch, dim=0, dim_size=int(hyp_batch.max()) + 1, reduce='mean')
		pool_c = scatter(c, constr_batch, dim=0, dim_size=int(constr_batch.max()) + 1, reduce='mean')
		logit = self.classifier(torch.cat([pool_h, pool_c], dim=-1)).squeeze(-1)
		return logit


def build_graph(A, b, device, lp_features):
	m, n = A.shape
	row_deg = np.maximum((A != 0).sum(axis=1), 1)
	b_norm = (b / row_deg).astype(np.float32)
	x_lp = None
	if lp_features:
		x_lp, _ = solve_lp_relaxation(A, b)
		lp_resid = ((b - A.dot(x_lp)) / row_deg).astype(np.float32)
	# hypothesis nodes: [var0_0, var0_1, var1_0, var1_1, ...]
	polarity = np.tile([0, 1], n).astype(np.float32)
	if lp_features:
		lp_rep = np.repeat(x_lp.astype(np.float32), 2)
		hyp_feat = np.stack([polarity, lp_rep], axis=1)
	else:
		hyp_feat = polarity[:, None]
	if lp_features:
		constr_feat = np.stack([b_norm, lp_resid], axis=1)
	else:
		constr_feat = b_norm[:, None]
	rows, cols = np.where(A != 0)
	hyp_idx = np.concatenate([2 * cols, 2 * cols + 1])
	constr_idx = np.concatenate([rows, rows])
	edge_h2c = np.stack([hyp_idx, constr_idx])
	complement_idx = np.arange(2 * n).reshape(n, 2)[:, ::-1].reshape(-1)
	return (torch.tensor(hyp_feat, dtype=torch.float, device=device),
	        torch.tensor(constr_feat, dtype=torch.float, device=device),
	        torch.tensor(edge_h2c, dtype=torch.long, device=device),
	        torch.tensor(complement_idx, dtype=torch.long, device=device))


def batch_graphs(items, device, lp_features):
	hf, cf, eh, comp, hb, cb, off_h, off_c = [], [], [], [], [], [], 0, 0
	for i, (A, b) in enumerate(items):
		h_i, c_i, e_i, comp_i = build_graph(A, b, device, lp_features)
		hf.append(h_i); cf.append(c_i)
		eh.append(e_i + torch.tensor([[off_h], [off_c]], device=device))
		comp.append(comp_i + off_h)
		hb.append(torch.full((h_i.size(0),), i, dtype=torch.long, device=device))
		cb.append(torch.full((c_i.size(0),), i, dtype=torch.long, device=device))
		off_h += h_i.size(0); off_c += c_i.size(0)
	return (torch.cat(hf), torch.cat(cf), torch.cat(eh, dim=1), torch.cat(comp),
	        torch.cat(hb), torch.cat(cb))


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
	ap.add_argument('--num_layers', type=int, default=16)
	ap.add_argument('--batch_size', type=int, default=64)
	ap.add_argument('--lp_features', type=int, default=0)
	ap.add_argument('--val_frac', type=float, default=0.2)
	ap.add_argument('--out', type=str, default=None)
	a = ap.parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	recs = [json.loads(l) for l in open(a.data)]
	print(f"loaded {len(recs)} instances (both members of each pair)")
	rng = np.random.default_rng(0)
	# split by pair_id so both twins of a pair stay on the same side (no leakage)
	pair_ids = sorted(set(r['pair_id'] for r in recs))
	rng.shuffle(pair_ids)
	nval = int(len(pair_ids) * a.val_frac)
	val_ids = set(pair_ids[:nval])
	train = [r for r in recs if r['pair_id'] not in val_ids]
	val = [r for r in recs if r['pair_id'] in val_ids]
	print(f"train {len(train)} / val {len(val)} instances ({len(pair_ids)-nval}/{nval} pairs)")

	model = DualNodeGNN(a.hidden_dim, a.num_layers, bool(a.lp_features)).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

	def run_epoch(data, train_mode):
		model.train(train_mode)
		idx = np.random.permutation(len(data)) if train_mode else np.arange(len(data))
		total_loss = 0.0
		all_p, all_y = [], []
		for s in range(0, len(idx), a.batch_size):
			chunk = [data[i] for i in idx[s:s + a.batch_size]]
			items = [(np.array(r['A'], dtype=np.int64), np.array(r['b'], dtype=np.int64)) for r in chunk]
			y = torch.tensor([1.0 if r['gt_feasible'] else 0.0 for r in chunk], device=device)
			hf, cf, eh, comp, hb, cb = batch_graphs(items, device, bool(a.lp_features))
			logit = model(hf, cf, eh, comp, hb, cb)
			loss = F.binary_cross_entropy_with_logits(logit, y)
			if train_mode:
				opt.zero_grad(); loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
				opt.step()
			total_loss += loss.item() * len(chunk)
			all_p.append(torch.sigmoid(logit).detach().cpu().numpy())
			all_y.append(y.cpu().numpy())
		p = np.concatenate(all_p); yy = np.concatenate(all_y)
		return total_loss / len(data), auc(yy, p), p, yy

	best_auc = 0.0
	for ep in range(1, a.epochs + 1):
		tr_loss, tr_auc, _, _ = run_epoch(train, True)
		if ep % 5 == 0 or ep == a.epochs:
			with torch.no_grad():
				val_loss, val_auc, pv, yv = run_epoch(val, False)
			mh = pv[yv == 1].mean() if (yv == 1).any() else 0
			mm = pv[yv == 0].mean() if (yv == 0).any() else 0
			best_auc = max(best_auc, val_auc)
			print(f"ep {ep:>3} | train loss {tr_loss:.4f} auc {tr_auc:.4f} | "
			      f"val loss {val_loss:.4f} auc {val_auc:.4f} | p(hit|feas)={mh:.3f} p(hit|infeas)={mm:.3f}", flush=True)
	print(f"lp_features={bool(a.lp_features)}  BEST val AUC = {best_auc:.4f}")
	if a.out:
		torch.save(model.state_dict(), a.out)


if __name__ == '__main__':
	main()
