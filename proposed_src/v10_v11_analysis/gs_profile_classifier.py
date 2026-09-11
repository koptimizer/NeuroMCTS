#!/usr/bin/env python3
"""
Sequence-model feasibility classifier over the raw BKZ Gram-Schmidt log-norm
profile (collect_gs_profile.py), instead of hand-picked summary statistics
(min/mean/std, which reached only AUC 0.588 in prior work). The profile is a
fixed-length ordered numeric sequence from an already symmetry-broken (BKZ
-reduced) representation, sidestepping the WL-indistinguishability limit
that blocks GNNs on the raw (A, b) bipartite graph.

Usage:
  python gs_profile_classifier.py --data ../../runs/gs_profile_20x50.jsonl --epochs 60
  python gs_profile_classifier.py --data ../../runs/gs_profile_20x50.jsonl --sanity
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProfileSeqClassifier(nn.Module):
	def __init__(self, seq_len=20, hidden_dim=32, num_layers=2):
		super().__init__()
		self.input_proj = nn.Linear(1, hidden_dim)
		self.pos_emb = nn.Embedding(seq_len, hidden_dim)
		self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
		                     batch_first=True, bidirectional=True)
		self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

	def forward(self, profile):
		B, L = profile.shape
		pos = torch.arange(L, device=profile.device)
		x = self.input_proj(profile.unsqueeze(-1)) + self.pos_emb(pos).unsqueeze(0)
		out, _ = self.lstm(x)
		pooled = out.mean(dim=1)
		return self.classifier(pooled).squeeze(-1)


def auc(y, x):
	from scipy.stats import rankdata
	pos = x[y == 1]; neg = x[y == 0]
	r = rankdata(np.concatenate([pos, neg]))
	return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run_epoch(model, opt, data, train_mode, batch_size, device):
	model.train(train_mode)
	idx = np.random.permutation(len(data)) if train_mode else np.arange(len(data))
	total_loss = 0.0
	all_p, all_y = [], []
	for s in range(0, len(idx), batch_size):
		chunk_idx = idx[s:s + batch_size]
		profile = torch.tensor(np.stack([data[0][i] for i in chunk_idx]), dtype=torch.float, device=device)
		y = torch.tensor(data[1][chunk_idx], dtype=torch.float, device=device)
		logit = model(profile)
		loss = F.binary_cross_entropy_with_logits(logit, y)
		if train_mode:
			opt.zero_grad(); loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
			opt.step()
		total_loss += loss.item() * len(chunk_idx)
		all_p.append(torch.sigmoid(logit).detach().cpu().numpy())
		all_y.append(y.cpu().numpy())
	p = np.concatenate(all_p); yy = np.concatenate(all_y)
	return total_loss / len(data[0]), auc(yy, p), p, yy


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data', required=True)
	ap.add_argument('--epochs', type=int, default=60)
	ap.add_argument('--hidden_dim', type=int, default=32)
	ap.add_argument('--num_layers', type=int, default=2)
	ap.add_argument('--batch_size', type=int, default=64)
	ap.add_argument('--val_frac', type=float, default=0.2)
	ap.add_argument('--sanity', action='store_true',
	                 help='Replace label with a trivial proxy (profile[0] > median) to verify the pipeline can learn')
	a = ap.parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	recs = [json.loads(l) for l in open(a.data)]
	profiles = np.array([r['profile'] for r in recs], dtype=np.float32)
	seq_len = profiles.shape[1]
	print(f"loaded {len(recs)} instances, profile length {seq_len}, sanity={a.sanity}")

	if a.sanity:
		y_all = (profiles[:, 0] > np.median(profiles[:, 0])).astype(np.float32)
		print(f"sanity proxy label balance: {y_all.mean():.3f}")
		pair_ids = np.arange(len(recs))
	else:
		y_all = np.array([1.0 if r['gt_feasible'] else 0.0 for r in recs], dtype=np.float32)
		pair_ids = np.array([r['pair_id'] for r in recs])

	rng = np.random.default_rng(0)
	uniq = sorted(set(pair_ids.tolist()))
	rng.shuffle(uniq)
	nval = int(len(uniq) * a.val_frac)
	val_ids = set(uniq[:nval])
	is_val = np.array([pid in val_ids for pid in pair_ids])
	train = (profiles[~is_val], y_all[~is_val])
	val = (profiles[is_val], y_all[is_val])
	print(f"train {len(train[0])} / val {len(val[0])}")

	model = ProfileSeqClassifier(seq_len, a.hidden_dim, a.num_layers).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

	best_auc = 0.0
	for ep in range(1, a.epochs + 1):
		tr_loss, tr_auc, _, _ = run_epoch(model, opt, train, True, a.batch_size, device)
		if ep % 5 == 0 or ep == a.epochs:
			with torch.no_grad():
				val_loss, val_auc, pv, yv = run_epoch(model, opt, val, False, a.batch_size, device)
			best_auc = max(best_auc, val_auc)
			print(f"ep {ep:>3} | train loss {tr_loss:.4f} auc {tr_auc:.4f} | "
			      f"val loss {val_loss:.4f} auc {val_auc:.4f}", flush=True)
	print(f"sanity={a.sanity}  BEST val AUC = {best_auc:.4f}")


if __name__ == '__main__':
	main()
