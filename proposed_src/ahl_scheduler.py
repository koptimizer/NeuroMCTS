#!/usr/bin/env python3
"""
Learned try-scheduler for the AHL lattice stage.

A discrete-time hazard model predicts P(hit at try t | not hit before, features)
from instance-static features (LP fractionality, size) and per-try dynamic
features (min vector norm, zero-tail count, Gram-Schmidt log-norm slope). At
inference the pipeline stops retrying once the expected marginal value of one
more try falls below its cost, and routes hard instances onward early. The
model is size-agnostic by construction (all features are size-normalized), so
it can be trained on small sizes and applied to larger unseen ones.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEAT_STATIC = ['lp_frac', 'log_n', 'aspect']
FEAT_DYN = ['min_norm_ratio', 'zero_tail_frac', 'gs_slope']
N_FEAT = len(FEAT_STATIC) + len(FEAT_DYN)


def static_feats(rec):
	n = rec['n']
	return np.array([rec['lp_frac'], np.log(n) / 5.0, rec['m'] / rec['n']], dtype=np.float32)


def dyn_feats(tr, n):
	# size-normalized: norm by sqrt(n+1) (the target solution norm), tail by n
	return np.array([tr['min_norm'] / np.sqrt(n + 1.0),
		tr['n_zero_tail'] / max(1, n), tr['gs_slope']], dtype=np.float32)


def build_dataset(path):
	# One training row per (instance, try t) that was actually executed:
	# features known BEFORE try t's outcome -> label hit(t)
	X, Y = [], []
	for line in open(path):
		rec = json.loads(line)
		sf = static_feats(rec)
		hit = rec['hit_try']
		trace = rec['trace']
		for t, tr in enumerate(trace):
			# features available before attempting try t: static + previous try's
			# dynamic trace (or zeros at t=0)
			prev = dyn_feats(trace[t - 1], rec['n']) if t > 0 else np.zeros(len(FEAT_DYN), np.float32)
			X.append(np.concatenate([sf, prev]))
			Y.append(1.0 if (hit is not None and hit == t + 1) else 0.0)
			if hit is not None and hit == t + 1:
				break
	return np.array(X, np.float32), np.array(Y, np.float32)


class Hazard(nn.Module):
	# Three-line MLP hazard head over normalized features
	def __init__(self, d=N_FEAT, h=32):
		super().__init__()
		self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

	def forward(self, x):
		return self.net(x).squeeze(-1)


def train(path, out, epochs=200, val_frac=0.2, seed=0):
	X, Y = build_dataset(path)
	rng = np.random.default_rng(seed)
	idx = rng.permutation(len(X))
	X, Y = X[idx], Y[idx]
	nv = int(len(X) * val_frac)
	Xtr, Ytr, Xv, Yv = X[nv:], Y[nv:], X[:nv], Y[:nv]
	mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
	Xtr = (Xtr - mu) / sd
	Xv = (Xv - mu) / sd
	model = Hazard()
	opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
	xt, yt = torch.tensor(Xtr), torch.tensor(Ytr)
	xv, yv = torch.tensor(Xv), torch.tensor(Yv)
	pos = Ytr.mean()
	pw = torch.tensor((1 - pos) / max(pos, 1e-3))
	best = 1e9
	for ep in range(epochs):
		model.train()
		opt.zero_grad()
		loss = F.binary_cross_entropy_with_logits(model(xt), yt, pos_weight=pw)
		loss.backward()
		opt.step()
		if ep % 20 == 0 or ep == epochs - 1:
			model.eval()
			with torch.no_grad():
				vl = F.binary_cross_entropy_with_logits(model(xv), yv, pos_weight=pw).item()
				pv = torch.sigmoid(model(xv)).numpy()
				# AUC-free calibration check: mean predicted hit-prob on hit vs miss rows
				mh = pv[Yv == 1].mean() if (Yv == 1).any() else 0
				mm = pv[Yv == 0].mean() if (Yv == 0).any() else 0
			print(f"ep {ep} | train {loss.item():.4f} | val {vl:.4f} | p(hit|hit)={mh:.3f} p(hit|miss)={mm:.3f}")
			if vl < best:
				best = vl
				torch.save({'state': model.state_dict(), 'mu': mu, 'sd': sd}, out)
	print(f"saved {out} (best val {best:.4f})")


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--labels', required=True)
	ap.add_argument('--out', default='./ahl_hazard.pt')
	ap.add_argument('--epochs', type=int, default=200)
	a = ap.parse_args()
	train(a.labels, a.out, a.epochs)
