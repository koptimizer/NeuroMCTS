"""v15 controlled experiment: hard label (planted x*) vs soft label (true marginal).

Identical architecture, identical instances, identical optimizer and schedule. The only
difference is the per-variable training target:
  hard : BCE against x*_j          -- what every previous cycle in this project used
  soft : BCE against p_j = P(x_j = 1 | A, b), computed by exact enumeration of S

Hypothesis under test: the ~11 failed learning attempts were partly fitting label noise,
because x* is one uniform draw from a solution set of median size 29 and a Bayes-optimal
predictor matches it only 66% of the time. If so, the soft-label model should move toward
the measured 66.0% / 91.7% (all-variable / top-3) ceiling while the hard-label model
stalls near the measured 56.6% / 75.0%.

Both arms are evaluated on the same held-out instances against BOTH targets, so the
comparison cannot be won merely by matching the metric a model was trained on.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import LPneuroBLS_v7 as m

torch.set_num_threads(1)


class MarginalNet(nn.Module):
	"""Bipartite variable/constraint GNN emitting one logit per variable.
	Deliberately small and single-headed: the target here is the marginal itself,
	not the multi-head policy/value stack the retired MCTS stages needed."""

	def __init__(self, hidden=64, layers=4):
		super().__init__()
		self.layers = layers
		self.var_in = nn.Linear(3, hidden)
		self.con_in = nn.Linear(3, hidden)
		self.gat_v2c = GATConv((hidden, hidden), hidden, add_self_loops=False)
		self.gat_c2v = GATConv((hidden, hidden), hidden, add_self_loops=False)
		self.ln_v = nn.LayerNorm(hidden)
		self.ln_c = nn.LayerNorm(hidden)
		self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

	def forward(self, xv, xc, ev2c, ec2v):
		hv = self.var_in(xv)
		hc = self.con_in(xc)
		for _ in range(self.layers):
			hc = self.ln_c(F.elu(self.gat_v2c((hv, hc), ev2c)) + hc)
			hv = self.ln_v(F.elu(self.gat_c2v((hc, hv), ec2v)) + hv)
		return self.head(hv).squeeze(-1)


def featurize(rec, device):
	# same input features for both arms; only the label differs
	A = np.array(rec['A'], dtype=np.int64)
	b = np.array(rec['b'], dtype=np.int64)
	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if not ok:
		return None
	x_lp = x_lp.astype(np.float64)
	mm, n = A.shape
	col_deg = (A != 0).sum(axis=0) / max(1, mm)
	frac = np.abs(x_lp - np.round(x_lp))
	row_sum = np.maximum(A.sum(axis=1), 1)
	xv = torch.tensor(np.stack([x_lp, col_deg, frac], -1), dtype=torch.float, device=device)
	xc = torch.tensor(np.stack([b / row_sum, row_sum / max(1, n),
	                             (b - A.dot(x_lp)) / row_sum], -1), dtype=torch.float, device=device)
	rows, cols = np.where(A == 1)
	ev2c = torch.tensor(np.array([cols, rows]), dtype=torch.long, device=device)
	ec2v = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device)
	return dict(xv=xv, xc=xc, ev2c=ev2c, ec2v=ec2v,
	             hard=torch.tensor(rec['x'], dtype=torch.float, device=device),
	             soft=torch.tensor(rec['marginals'], dtype=torch.float, device=device))


def evaluate(model, data):
	# reported against both targets so neither arm is scored only on its own objective
	model.eval()
	acc_hard, acc_soft, top3, l1 = [], [], [], []
	with torch.no_grad():
		for f in data:
			p = torch.sigmoid(model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).cpu().numpy()
			hard = f['hard'].cpu().numpy()
			soft = f['soft'].cpu().numpy()
			pred = (p >= 0.5).astype(np.float64)
			acc_hard.append((pred == hard).mean())
			acc_soft.append((pred == (soft >= 0.5)).mean())
			order = np.argsort(-np.abs(p - 0.5))[:3]
			top3.append((pred[order] == hard[order]).mean())
			l1.append(np.abs(p - soft).mean())
	model.train()
	return (float(np.mean(acc_hard)), float(np.mean(acc_soft)),
	         float(np.mean(top3)), float(np.mean(l1)))


def train_arm(target, train_data, val_data, epochs, lr, seed, log_every):
	torch.manual_seed(seed)
	model = MarginalNet()
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	hist = []
	for ep in range(1, epochs + 1):
		order = np.random.permutation(len(train_data))
		tot = 0.0
		for i in order:
			f = train_data[i]
			logits = model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])
			loss = F.binary_cross_entropy_with_logits(logits, f[target])
			opt.zero_grad(); loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
			tot += loss.item()
		if ep % log_every == 0 or ep == epochs:
			ah, as_, t3, l1 = evaluate(model, val_data)
			hist.append(dict(epoch=ep, loss=tot / len(train_data), acc_vs_planted=ah,
			                  acc_vs_marginal=as_, top3_vs_planted=t3, l1_to_marginal=l1))
			print(f"  [{target}] ep{ep:>3} loss={tot/len(train_data):.4f}  "
			      f"acc(vs planted)={100*ah:.1f}%  acc(vs marginal)={100*as_:.1f}%  "
			      f"top3={100*t3:.1f}%  L1={l1:.4f}", flush=True)
	return model, hist


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--labels', required=True)
	ap.add_argument('--n_train', type=int, default=1500)
	ap.add_argument('--n_val', type=int, default=300)
	ap.add_argument('--epochs', type=int, default=30)
	ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--log_every', type=int, default=5)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	device = torch.device('cpu')
	recs = json.load(open(a.labels))
	print(f"loaded {len(recs)} labeled instances", flush=True)
	rng = np.random.default_rng(a.seed)
	rng.shuffle(recs)

	t0 = time.time()
	feats = []
	for r in recs[:a.n_train + a.n_val]:
		f = featurize(r, device)
		if f is not None:
			feats.append(f)
	print(f"featurized {len(feats)} in {time.time()-t0:.0f}s", flush=True)
	train_data, val_data = feats[:a.n_train], feats[a.n_train:a.n_train + a.n_val]
	print(f"train={len(train_data)}  val={len(val_data)}\n", flush=True)

	out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
	results = {}
	for target in ('hard', 'soft'):
		print(f"=== training on {target} labels ===", flush=True)
		model, hist = train_arm(target, train_data, val_data, a.epochs, a.lr, a.seed, a.log_every)
		torch.save({'model_state_dict': model.state_dict()}, out_dir / f'marginal_{target}.pt')
		results[target] = hist

	print("\n" + "=" * 92)
	print("v15: does the training target explain the learning failure?")
	print("=" * 92)
	print(f"{'arm':<28}{'acc vs planted':>16}{'acc vs marginal':>17}{'top-3':>9}{'L1 to marginal':>17}")
	for target in ('hard', 'soft'):
		h = results[target][-1]
		print(f"{'trained on '+target+' labels':<28}{100*h['acc_vs_planted']:>15.1f}%"
		      f"{100*h['acc_vs_marginal']:>16.1f}%{100*h['top3_vs_planted']:>8.1f}%{h['l1_to_marginal']:>17.4f}")
	print(f"{'measured Bayes ceiling':<28}{66.0:>15.1f}%{100.0:>16.1f}%{91.7:>8.1f}%{0.0:>17.4f}")
	print(f"{'existing GNN (best.pth)':<28}{56.6:>15.1f}%{'-':>16}{75.0:>8.1f}%{'-':>17}")
	json.dump(results, open(out_dir / 'v15_compare.json', 'w'), indent=2)


if __name__ == '__main__':
	main()
