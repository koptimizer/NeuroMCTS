# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

#!/usr/bin/env python3
"""
Final, fairest test of the aggregate-probe infeasibility-signal hypothesis:
combine the frozen pretrained GNN's pooled graph embedding (which sees full
matrix structure -- degrees, sparsity pattern -- that scalar stats cannot)
with the 7 aggregate probe features from collect_feas_signal.py, and check
whether a classifier on the combination beats the scalar-only classifier.

Usage:
  python eval_feas_classifier.py --labels ../../runs/feas_signal_20x50.jsonl \
      --checkpoint ../../runs/.../checkpoint_pretrained.pt
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from LPneuroBLS_v7 import BipartiteGNN, FastBinaryEnv, solve_lp_relaxation

FEATS = ['lp_frac', 'pump_res_min', 'pump_res_mean', 'pump_res_std',
         'ahl_min_norm_min', 'ahl_min_norm_mean', 'ahl_min_norm_std']


def embed(model, device, A, b):
	env = FastBinaryEnv(A, b).propagate_constraints().apply_probing()
	x_lp, _ = solve_lp_relaxation(A, b)
	row_deg = np.maximum(np.sum(A != 0, axis=1), 1)
	x_lp_check = (b - A.dot(x_lp)) / row_deg
	col_deg = (A != 0).sum(axis=0) / max(1, A.shape[0])
	xv, xc, cx = env.get_tensor_state(device)
	xlpv = torch.tensor(np.stack([x_lp, col_deg], axis=-1), dtype=torch.float, device=device)
	xlpc = torch.tensor(x_lp_check, dtype=torch.float, device=device)
	rows, cols = np.where(A == 1)
	ev = torch.tensor(np.array([cols, rows]), dtype=torch.long, device=device)
	ec = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device)
	with torch.no_grad():
		h_var_emb = model.var_emb(xv)
		h_check_emb = model.check_emb(xc)
		h_var = model.var_feat_proj(torch.cat([h_var_emb, xlpv], dim=-1))
		h_check = model.check_feat_proj(torch.cat([h_check_emb, xlpc.unsqueeze(-1), cx], dim=-1))
		c_var = torch.zeros_like(h_var); c_check = torch.zeros_like(h_check)
		h_var_i, h_check_i = h_var, h_check
		for _ in range(model.num_layers):
			msg_v = model.mlp_v2c(h_var)
			a2c = F.elu(model.gat_v2c((msg_v, h_check), ev)).contiguous()
			h_check, c_check = model.lstm_check(a2c, (h_check, c_check)); h_check = model.ln_check(h_check)
			msg_c = model.mlp_c2v(h_check)
			c2v = F.elu(model.gat_c2v((msg_c, h_var), ec)).contiguous()
			h_var, c_var = model.lstm_var(c2v, (h_var, c_var)); h_var = model.ln_var(h_var)
		h_var = h_var + h_var_i; h_check = h_check + h_check_i
		pool = torch.cat([h_var.mean(0, keepdim=True), h_check.mean(0, keepdim=True)], dim=1)
	return pool.squeeze(0).cpu().numpy()


def auc(y, x):
	from scipy.stats import rankdata
	pos = x[y == 1]; neg = x[y == 0]
	r = rankdata(np.concatenate([pos, neg]))
	return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--labels', required=True)
	ap.add_argument('--checkpoint', required=True)
	ap.add_argument('--hidden_dim', type=int, default=256)
	ap.add_argument('--num_layers', type=int, default=8)
	ap.add_argument('--max_n', type=int, default=3000)
	a = ap.parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = BipartiteGNN(hidden_dim=a.hidden_dim, num_layers=a.num_layers).to(device)
	ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
	sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
	model.load_state_dict(sd)
	model.eval()

	recs = [json.loads(l) for l in open(a.labels)]
	recs = [r for r in recs if r.get('ahl_min_norm_min') is not None]
	rng0 = np.random.default_rng(1)
	rng0.shuffle(recs)
	recs = recs[:a.max_n]
	print(f"embedding {len(recs)} residual instances ...")
	embeds, scalars, labels = [], [], []
	for i, r in enumerate(recs):
		d = json.load(open(r['path']))
		A = np.array(d['A'], dtype=np.int32); b = np.array(d['b'], dtype=np.int32)
		e = embed(model, device, A, b)
		embeds.append(e)
		scalars.append([r[f] for f in FEATS])
		labels.append(1.0 if r['gt_feasible'] else 0.0)
		if (i + 1) % 500 == 0:
			print(f"  {i+1}/{len(recs)}", flush=True)
	E = np.array(embeds, dtype=np.float32)
	S = np.array(scalars, dtype=np.float32)
	Y = np.array(labels, dtype=np.float32)

	rng = np.random.default_rng(0)
	idx = rng.permutation(len(Y))
	E, S, Y = E[idx], S[idx], Y[idx]
	nval = int(len(Y) * 0.25)

	def split_norm(M):
		tr, v = M[nval:], M[:nval]
		mu, sd_ = tr.mean(0), tr.std(0) + 1e-6
		return (tr - mu) / sd_, (v - mu) / sd_

	Str, Sv = split_norm(S)
	Etr, Ev = split_norm(E)
	Ytr, Yv = Y[nval:], Y[:nval]

	def train_eval(Xtr, Xv, tag, epochs=300, hidden=32):
		d = Xtr.shape[1]
		mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
		opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=3e-3)
		xt, yt = torch.tensor(Xtr), torch.tensor(Ytr)
		for _ in range(epochs):
			opt.zero_grad()
			loss = F.binary_cross_entropy_with_logits(mlp(xt).squeeze(-1), yt)
			loss.backward(); opt.step()
		with torch.no_grad():
			pv = torch.sigmoid(mlp(torch.tensor(Xv))).numpy().squeeze(-1)
		a_ = auc(Yv, pv)
		print(f"{tag:>28}: held-out AUC = {a_:.4f}  (dim={d})")
		return a_

	train_eval(Str, Sv, 'scalar probe feats only')
	train_eval(Etr, Ev, 'GNN pooled embedding only')
	train_eval(np.concatenate([Str, Etr], 1), np.concatenate([Sv, Ev], 1), 'embedding + scalar combined')


if __name__ == '__main__':
	main()
