"""v13 training: REINFORCE over AHL column permutations.

The policy is a small GNN scoring each variable; a permutation is sampled from those
scores via Plackett-Luce, handed to one AHL/BKZ attempt, and rewarded 1/0 by whether
that attempt recovered a verified 0/1 solution. Because per-permutation success rates
are low (measured 3-40% at 20x50/block10, 0-7% at 40x100/block20), each instance is
rolled out several times per update and a leave-one-out baseline is subtracted to keep
gradient variance manageable.

Deliberately NOT reusing LPneuroBLS_v7's BipartiteGNN wholesale: that network's heads
(selection/assign/value/feasibility) were built for the MCTS/DFS roles this cycle
retires. Only its message-passing shape is mirrored, with a single per-variable score
head, so nothing about the retired stages leaks into the new objective.

Usage:
  python v13_train_perm.py --sizes 10x25 20x50 --deadline_hours 8 --out ../../runs/v13
"""
# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from v13_pipeline import ahl_try_with_perm, sample_perm, perm_logprob

SIZE_BLOCK = {'10x25': 8, '20x50': 10, '40x100': 20, '60x150': 15}
TRAIN_DIRS = {
	'10x25': '../../instances/train_instances_hard_10x25_10000',
	'20x50': '../../instances/train_instances_hard_20x50_10000',
	'40x100': '../../instances/train_instances_hard_40x100_10000',
	'60x150': '../../instances/train_instances_hard_60x150_1500',
}


class PermPolicy(nn.Module):
	"""Bipartite variable/constraint GNN emitting one score per variable.
	Scores parameterize a Plackett-Luce distribution over column permutations.
	Message passing mirrors the project's existing encoder; the heads do not."""

	def __init__(self, hidden=64, layers=3):
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


def instance_features(A, b, x_lp, device):
	# variable: LP value, column degree, LP fractionality
	# constraint: b/rowsum (tightness), rowsum (scaled), LP residual
	mm, n = A.shape
	col_deg = (A != 0).sum(axis=0) / max(1, mm)
	frac = np.abs(x_lp - np.round(x_lp))
	xv = np.stack([x_lp, col_deg, frac], axis=-1)
	row_sum = np.maximum(A.sum(axis=1), 1)
	tight = b / row_sum
	resid = (b - A.dot(x_lp)) / row_sum
	xc = np.stack([tight, row_sum / max(1, n), resid], axis=-1)
	rows, cols = np.where(A == 1)
	ev2c = torch.tensor(np.array([cols, rows]), dtype=torch.long, device=device)
	ec2v = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device)
	return (torch.tensor(xv, dtype=torch.float, device=device),
	        torch.tensor(xc, dtype=torch.float, device=device), ev2c, ec2v)


def load_pool(size, limit):
	import LPneuroBLS_v7 as m
	pool = []
	files = sorted(Path(TRAIN_DIRS[size]).glob('*.json'))
	for f in files:
		if len(pool) >= limit:
			break
		d = json.load(open(f))
		if not d.get('feasible', True):
			continue
		A = np.array(d['A'], dtype=np.int64)
		b = np.array(d['b'], dtype=np.int64)
		x_lp, lp_ok = m.solve_lp_relaxation(A, b)
		if not lp_ok:
			continue
		pool.append((A, b, x_lp.astype(np.float64)))
	return pool


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sizes', nargs='+', default=['10x25', '20x50'])
	ap.add_argument('--pool_per_size', type=int, default=300)
	ap.add_argument('--rollouts', type=int, default=8, help='permutation samples per instance per update')
	ap.add_argument('--lr', type=float, default=3e-4)
	ap.add_argument('--temperature', type=float, default=1.0)
	ap.add_argument('--entropy_coef', type=float, default=0.01)
	ap.add_argument('--deadline_hours', type=float, required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--log_every', type=int, default=25)
	ap.add_argument('--ckpt_every', type=int, default=200)
	a = ap.parse_args()

	out_dir = Path(a.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	device = torch.device(a.device)

	pools = {}
	for sz in a.sizes:
		pools[sz] = load_pool(sz, a.pool_per_size)
		print(f"{sz}: loaded {len(pools[sz])} feasible instances (block={SIZE_BLOCK[sz]})", flush=True)

	model = PermPolicy().to(device)
	opt = torch.optim.Adam(model.parameters(), lr=a.lr)
	rng = np.random.default_rng(0)

	deadline = time.time() + a.deadline_hours * 3600
	step = 0
	hist = []
	recent_rand, recent_pol = [], []
	t0 = time.time()

	while time.time() < deadline:
		sz = a.sizes[int(rng.integers(len(a.sizes)))]
		block = SIZE_BLOCK[sz]
		A, b, x_lp = pools[sz][int(rng.integers(len(pools[sz])))]
		xv, xc, ev2c, ec2v = instance_features(A, b, x_lp, device)

		scores = model(xv, xc, ev2c, ec2v)
		scores_np = scores.detach().cpu().numpy()

		perms, rewards = [], []
		for _ in range(a.rollouts):
			perm = sample_perm(scores_np, rng, a.temperature)
			sol = ahl_try_with_perm(A, b, perm, block)
			perms.append(perm)
			rewards.append(1.0 if sol is not None else 0.0)
		rewards = np.array(rewards, dtype=np.float64)

		# uniform-random control on the same instance, same number of tries: the only
		# comparison that matters is policy-vs-random at equal budget
		rand_hits = 0
		for _ in range(a.rollouts):
			p = rng.permutation(A.shape[1])
			if ahl_try_with_perm(A, b, p, block) is not None:
				rand_hits += 1
		recent_rand.append(rand_hits / a.rollouts)
		recent_pol.append(rewards.mean())

		if rewards.std() > 0:
			# leave-one-out baseline keeps the gradient unbiased with low variance
			loo = (rewards.sum() - rewards) / max(1, len(rewards) - 1)
			adv = torch.tensor(rewards - loo, dtype=torch.float, device=device)
			logps = torch.stack([perm_logprob(scores, torch.tensor(p, device=device), a.temperature)
			                      for p in perms])
			pg_loss = -(adv * logps).mean()
			logp_all = F.log_softmax(scores, dim=0)
			ent = -(logp_all.exp() * logp_all).sum()
			loss = pg_loss - a.entropy_coef * ent
			opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()

		step += 1
		if step % a.log_every == 0:
			pol = float(np.mean(recent_pol[-a.log_every:]))
			rnd = float(np.mean(recent_rand[-a.log_every:]))
			el = (time.time() - t0) / 3600
			print(f"[step {step}] elapsed={el:.2f}h  policy_hit={pol:.4f}  random_hit={rnd:.4f}  "
			      f"delta={pol-rnd:+.4f}", flush=True)
			hist.append(dict(step=step, policy=pol, random=rnd, elapsed_h=el))
			with open(out_dir / 'train_hist.json', 'w') as f:
				json.dump(hist, f, indent=2)
		if step % a.ckpt_every == 0:
			torch.save({'model_state_dict': model.state_dict(), 'step': step},
			            out_dir / f'perm_policy_step{step}.pt')

	torch.save({'model_state_dict': model.state_dict(), 'step': step}, out_dir / 'perm_policy_final.pt')
	pol = float(np.mean(recent_pol)) if recent_pol else 0.0
	rnd = float(np.mean(recent_rand)) if recent_rand else 0.0
	print(f"DONE: {step} steps in {(time.time()-t0)/3600:.2f}h  "
	      f"overall policy={pol:.4f} random={rnd:.4f} delta={pol-rnd:+.4f}", flush=True)


if __name__ == '__main__':
	main()
