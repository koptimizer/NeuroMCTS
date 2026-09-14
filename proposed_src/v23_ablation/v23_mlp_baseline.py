"""Is the graph structure doing any work? An MLP and a linear model on the same features.

MarginalNet consumes six scalars per node (three per variable, three per constraint) and
propagates them over the bipartite graph. With a feature set that small, a per-variable
model that ignores the graph entirely is a serious baseline: if it matches, the message
passing is decoration and the contribution should be stated as feature engineering.

Each baseline sees exactly the variable-side features MarginalNet sees, optionally
concatenated with constraint-side aggregates over that variable's incident rows, which is
the information one round of message passing would deliver. Same targets, same loss, same
train/test split, so the only difference is whether structure is used.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].iterdir())
                 if p.is_dir() and not p.name.startswith('.')]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import LPneuroBLS_v7 as m
from v15_train_marginal import MarginalNet, featurize

torch.set_num_threads(1)


def tabular_features(rec, agg):
	"""Per-variable feature rows. With agg=True, append aggregates of the incident
	constraints -- the information a single message-passing round would carry."""
	A = np.array(rec['A'], dtype=np.int64)
	b = np.array(rec['b'], dtype=np.int64)
	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if not ok:
		return None, None
	x_lp = x_lp.astype(np.float64)
	mm, n = A.shape
	col_deg = (A != 0).sum(axis=0) / max(1, mm)
	frac = np.abs(x_lp - np.round(x_lp))
	X = [x_lp, col_deg, frac]
	if agg:
		r = np.maximum(A.sum(axis=1), 1)
		tight, size, resid = b / r, r / max(1, n), (b - A.dot(x_lp)) / r
		deg = np.maximum((A != 0).sum(axis=0), 1)
		for v in (tight, size, resid):                      # mean and max over incident rows
			X.append(A.T.dot(v) / deg)
			X.append(np.array([v[A[:, j] == 1].max() if (A[:, j] == 1).any() else 0.0
			                    for j in range(n)]))
	return np.stack(X, -1).astype(np.float32), np.array(rec['marginals'], dtype=np.float32)


class MLP(nn.Module):
	"""Per-variable MLP with no access to graph structure beyond the given features."""

	def __init__(self, d_in, hidden=64, layers=3):
		super().__init__()
		L, d = [], d_in
		for _ in range(layers):
			L += [nn.Linear(d, hidden), nn.ReLU()]
			d = hidden
		L += [nn.Linear(d, 1)]
		self.net = nn.Sequential(*L)

	def forward(self, x):
		return self.net(x).squeeze(-1)


def run_tabular(tr, te, d_in, kind, epochs, lr, seed):
	torch.manual_seed(seed)
	model = nn.Linear(d_in, 1) if kind == 'linear' else MLP(d_in)
	fwd = (lambda x: model(x).squeeze(-1)) if kind == 'linear' else model
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	for _ in range(epochs):
		for i in np.random.permutation(len(tr)):
			X, y = tr[i]
			loss = F.binary_cross_entropy_with_logits(fwd(X), y)
			opt.zero_grad(); loss.backward(); opt.step()
	acc, l1 = [], []
	with torch.no_grad():
		for X, y in te:
			p = torch.sigmoid(fwd(X)).numpy()
			yt = y.numpy()
			acc.append(((p >= 0.5) == (yt >= 0.5)).mean())
			l1.append(np.abs(p - yt).mean())
	return float(np.mean(acc)), float(np.mean(l1)), sum(p.numel() for p in model.parameters())


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--train', required=True)
	ap.add_argument('--test', required=True)
	ap.add_argument('--n_train', type=int, default=2000)
	ap.add_argument('--n_test', type=int, default=800)
	ap.add_argument('--epochs', type=int, default=20)
	ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--gnn_ckpt', required=True)
	ap.add_argument('--seed', type=int, default=0)
	a = ap.parse_args()

	tr_recs = json.load(open(a.train))[:a.n_train]
	te_recs = json.load(open(a.test))[:a.n_test]
	print(f"train={len(tr_recs)} test={len(te_recs)} states", flush=True)

	res = {}
	for kind, agg in [('linear', False), ('mlp', False), ('mlp', True)]:
		tr = [(torch.tensor(X), torch.tensor(y))
		       for X, y in (tabular_features(r, agg) for r in tr_recs) if X is not None]
		te = [(torch.tensor(X), torch.tensor(y))
		       for X, y in (tabular_features(r, agg) for r in te_recs) if X is not None]
		acc, l1, np_ = run_tabular(tr, te, tr[0][0].shape[1], kind, a.epochs, a.lr, a.seed)
		name = f"{kind}{' + constraint aggregates' if agg else ''} ({tr[0][0].shape[1]} feat)"
		res[name] = (acc, l1, np_)
		print(f"  {name:44s} acc={100*acc:5.1f}%  L1={l1:.4f}  params={np_:,}", flush=True)

	device = torch.device('cpu')
	gnn = MarginalNet()
	gnn.load_state_dict(torch.load(a.gnn_ckpt, map_location=device, weights_only=False)['model_state_dict'])
	gnn.eval()
	acc, l1 = [], []
	with torch.no_grad():
		for r in te_recs:
			f = featurize(r, device)
			if f is None:
				continue
			p = torch.sigmoid(gnn(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy()
			yt = np.array(r['marginals'])
			acc.append(((p >= 0.5) == (yt >= 0.5)).mean())
			l1.append(np.abs(p - yt).mean())
	npar = sum(p.numel() for p in gnn.parameters())
	print(f"  {'MarginalNet (graph)':44s} acc={100*np.mean(acc):5.1f}%  "
	      f"L1={np.mean(l1):.4f}  params={npar:,}", flush=True)
	ceil = np.mean([np.maximum(np.array(r['marginals']), 1 - np.array(r['marginals'])).mean()
	                 for r in te_recs])
	print(f"  {'Bayes ceiling':44s} acc={100*ceil:5.1f}%")


if __name__ == '__main__':
	main()
