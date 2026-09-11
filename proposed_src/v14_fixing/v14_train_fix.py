"""v14 training: REINFORCE over sequential variable fixings.

Unlike v13's permutation policy (a stateless bandit that failed), this is a genuine
k-step MDP: after each fix the instance shrinks, its LP relaxation is re-solved, and
the next decision is conditioned on the new state. The objective is also different in
kind from v6/v7's supervised assign head -- policy gradient optimizes P(all k fixings
jointly correct), which is what the retry loop actually consumes, whereas supervised
cross-entropy optimizes per-variable accuracy. At the measured ~73% per-variable
accuracy those two objectives have very different optima.

Reward (see docs/version.md v14):
  +1        AHL solves the reduced instance
  -reject_p a fix made the reduced instance provably unsatisfiable (free/cheap to
            detect, so this is the dense part of the signal)
   0        the fixing was plausible but AHL still failed

The planted solution is deliberately NOT used as the reward target: these instances
are multi-solution, so "differs from x_gt" does not imply "wrong".
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

import LPneuroBLS_v7 as m
from v14_core import (reduce_instance, bound_reject, lp_reject, ahl_on_reduced,
                       reconstruct, gnn_marginals)

# These graphs are tiny (hundreds of edges); multi-threaded BLAS spends more time
# synchronizing than computing. Measured: 11.3 ms/forward at 8 threads vs 0.9 ms at 1.
torch.set_num_threads(1)

SIZE_BLOCK = {'10x25': 8, '20x50': 10, '40x100': 20}
SIZE_K = {'10x25': 3, '20x50': 5, '40x100': 8}
TRAIN_DIRS = {
	'10x25': '../../instances/train_instances_hard_10x25_10000',
	'20x50': '../../instances/train_instances_hard_20x50_10000',
	'40x100': '../../instances/train_instances_hard_40x100_10000',
}


class FixPolicy(nn.Module):
	"""Bipartite GNN scoring each (variable, value) pair of the current reduced instance.
	One softmax over 2*n_remaining actions defines the next fix.
	Re-invoked after every fix, so the policy sees the shrunken instance each step."""

	def __init__(self, hidden=64, layers=3):
		super().__init__()
		self.layers = layers
		self.var_in = nn.Linear(3, hidden)
		self.con_in = nn.Linear(3, hidden)
		self.gat_v2c = GATConv((hidden, hidden), hidden, add_self_loops=False)
		self.gat_c2v = GATConv((hidden, hidden), hidden, add_self_loops=False)
		self.ln_v = nn.LayerNorm(hidden)
		self.ln_c = nn.LayerNorm(hidden)
		self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

	def forward(self, xv, xc, ev2c, ec2v):
		hv = self.var_in(xv)
		hc = self.con_in(xc)
		for _ in range(self.layers):
			hc = self.ln_c(F.elu(self.gat_v2c((hv, hc), ev2c)) + hc)
			hv = self.ln_v(F.elu(self.gat_c2v((hc, hv), ec2v)) + hv)
		return self.head(hv)


def state_features(A, b, device):
	# features of the current (possibly reduced) instance; LP is re-solved each step
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
	return xv, xc, ev2c, ec2v


def rollout(model, A, b, k, device, rng, greedy=False, collect=True, temperature=1.0):
	"""One k-step fixing episode on the ORIGINAL index space.
	Returns (fixed_idx, fixed_vals, logps, broke_early)."""
	n = A.shape[1]
	alive = np.arange(n)          # original indices still unfixed
	A_cur, b_cur = A.copy(), b.copy()
	fixed_idx, fixed_vals, logps = [], [], []
	for _ in range(k):
		if A_cur.shape[1] == 0:
			break
		feat = state_features(A_cur, b_cur, device)
		if feat is None:
			return fixed_idx, fixed_vals, logps, True
		logits = model(*feat).reshape(-1)       # (n_cur * 2,)
		logp_all = F.log_softmax(logits, dim=0)
		if greedy:
			a = int(torch.argmax(logp_all).item())
		else:
			# exploration temperature is separate from the policy itself: the
			# distilled distribution must stay peaked to match a2_gnn quality
			probs = F.softmax(logits / max(temperature, 1e-6), dim=0)
			a = int(torch.multinomial(probs, 1).item())
		if collect:
			logps.append(logp_all[a])
		local_var, val = a // 2, a % 2
		fixed_idx.append(int(alive[local_var]))
		fixed_vals.append(int(val))
		keep = np.setdiff1d(np.arange(A_cur.shape[1]), [local_var])
		b_cur = b_cur - A_cur[:, local_var] * val
		A_cur = A_cur[:, keep]
		alive = alive[keep]
		if bound_reject(A_cur, b_cur):
			return fixed_idx, fixed_vals, logps, True
	return fixed_idx, fixed_vals, logps, False


def policy_propose(model, A, b, k, rng, device, greedy=False):
	# inference-time hook used by v14_bench's a3_policy arm
	idx, vals, _, _ = rollout(model, A, b, k, device, rng, greedy=greedy, collect=False)
	return np.array(idx, dtype=np.int64), np.array(vals, dtype=np.int64)


def load_pool(size, limit):
	pool = []
	for f in sorted(Path(TRAIN_DIRS[size]).glob('*.json')):
		if len(pool) >= limit:
			break
		d = json.load(open(f))
		if not d.get('feasible', True):
			continue
		pool.append((np.array(d['A'], dtype=np.int64), np.array(d['b'], dtype=np.int64)))
	return pool


def distill_target(p1, device, beta=12.0):
	# the a2_gnn behavior as an action distribution. beta sharpens variable choice:
	# a target merely proportional to confidence is far too flat, and sampling from
	# it picks middling variables instead of the confident ones a2_gnn commits to
	# (observed: flat target left 40x100 at a 0.000 solve rate even after distilling)
	conf = np.abs(p1 - 0.5)
	logits = beta * conf[:, None] + np.log(np.stack([1.0 - p1, p1], axis=-1) + 1e-9)
	flat = logits.reshape(-1)
	flat = flat - flat.max()
	tgt = np.exp(flat)
	return torch.tensor(tgt / tgt.sum(), dtype=torch.float, device=device)


def warm_start(model, pools, sizes, ref_ckpt, device, steps, lr):
	"""Cold-start fix: a randomly initialized policy gets k fixings jointly right
	with probability 2^-k (0.4% at k=8), so REINFORCE never observes a success and
	has no gradient. Distilling the pretrained assign head first puts the policy at
	the measured a2_gnn operating point, from which RL has something to improve on."""
	ref = m.BipartiteGNN(hidden_dim=256, num_layers=8).to(device)
	st = torch.load(ref_ckpt, map_location=device, weights_only=False)
	ref.load_state_dict(st['model_state_dict'] if 'model_state_dict' in st else st)
	ref.eval()
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	rng = np.random.default_rng(1)
	t0 = time.time()
	for i in range(steps):
		sz = sizes[int(rng.integers(len(sizes)))]
		A, b = pools[sz][int(rng.integers(len(pools[sz])))]
		x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
		if not ok:
			continue
		p1 = gnn_marginals(ref, A, b, x_lp.astype(np.float64), device)
		feat = state_features(A, b, device)
		if feat is None:
			continue
		logp = F.log_softmax(model(*feat).reshape(-1), dim=0)
		loss = F.kl_div(logp, distill_target(p1, device), reduction='batchmean')
		opt.zero_grad(); loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		opt.step()
		if (i + 1) % 200 == 0:
			print(f"  [warm-start {i+1}/{steps}] kl={loss.item():.5f} ({time.time()-t0:.0f}s)", flush=True)
	return model


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sizes', nargs='+', default=['20x50', '40x100'])
	ap.add_argument('--pool_per_size', type=int, default=300)
	ap.add_argument('--rollouts', type=int, default=6)
	ap.add_argument('--train_tries', type=int, default=3)
	ap.add_argument('--reject_penalty', type=float, default=0.3)
	ap.add_argument('--lr', type=float, default=3e-4)
	ap.add_argument('--entropy_coef', type=float, default=0.02)
	ap.add_argument('--temperature', type=float, default=0.5)
	ap.add_argument('--deadline_hours', type=float, required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--log_every', type=int, default=25)
	ap.add_argument('--ckpt_every', type=int, default=250)
	ap.add_argument('--warm_steps', type=int, default=3000)
	ap.add_argument('--warm_lr', type=float, default=1e-3)
	ap.add_argument('--ref_ckpt', default='../../runs/best.pth')
	a = ap.parse_args()

	out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
	device = torch.device(a.device)
	pools = {sz: load_pool(sz, a.pool_per_size) for sz in a.sizes}
	for sz in a.sizes:
		print(f"{sz}: {len(pools[sz])} feasible instances, k={SIZE_K[sz]}, block={SIZE_BLOCK[sz]}", flush=True)

	model = FixPolicy().to(device)
	if a.warm_steps > 0:
		print(f"=== warm-start: distilling {a.ref_ckpt} for {a.warm_steps} steps ===", flush=True)
		warm_start(model, pools, a.sizes, a.ref_ckpt, device, a.warm_steps, a.warm_lr)
		torch.save({'model_state_dict': model.state_dict(), 'step': 0},
		            out_dir / 'fix_policy_warmstart.pt')
	opt = torch.optim.Adam(model.parameters(), lr=a.lr)
	rng = np.random.default_rng(0)
	deadline = time.time() + a.deadline_hours * 3600
	step = 0
	hist = []
	recent = {sz: [] for sz in a.sizes}
	t0 = time.time()

	while time.time() < deadline:
		sz = a.sizes[int(rng.integers(len(a.sizes)))]
		k, block = SIZE_K[sz], SIZE_BLOCK[sz]
		A, b = pools[sz][int(rng.integers(len(pools[sz])))]

		rewards, all_logps = [], []
		for _ in range(a.rollouts):
			idx, vals, logps, broke = rollout(model, A, b, k, device, rng, temperature=a.temperature)
			if broke or len(idx) == 0:
				r = -a.reject_penalty
			else:
				A_red, b_red, keep = reduce_instance(A, b, np.array(idx), np.array(vals))
				if lp_reject(A_red, b_red):
					r = -a.reject_penalty
				else:
					x_red = ahl_on_reduced(A_red, b_red, block, a.train_tries, rng)
					r = 1.0 if x_red is not None else 0.0
			rewards.append(r)
			all_logps.append(torch.stack(logps).sum() if logps else None)
		rewards = np.array(rewards, dtype=np.float64)
		recent[sz].append(float((rewards > 0).mean()))

		valid = [i for i, lp in enumerate(all_logps) if lp is not None]
		if len(valid) >= 2 and rewards[valid].std() > 0:
			rv = rewards[valid]
			loo = (rv.sum() - rv) / (len(rv) - 1)
			adv = torch.tensor(rv - loo, dtype=torch.float, device=device)
			lps = torch.stack([all_logps[i] for i in valid])
			loss = -(adv * lps).mean()
			opt.zero_grad(); loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()

		step += 1
		if step % a.log_every == 0:
			msg = '  '.join(f"{sz}_solve={np.mean(recent[sz][-200:]):.3f}" for sz in a.sizes if recent[sz])
			el = (time.time() - t0) / 3600
			print(f"[step {step}] elapsed={el:.2f}h  {msg}", flush=True)
			hist.append(dict(step=step, elapsed_h=el,
			                  **{sz: (float(np.mean(recent[sz][-200:])) if recent[sz] else None) for sz in a.sizes}))
			json.dump(hist, open(out_dir / 'train_hist.json', 'w'), indent=2)
		if step % a.ckpt_every == 0:
			torch.save({'model_state_dict': model.state_dict(), 'step': step},
			            out_dir / f'fix_policy_step{step}.pt')

	torch.save({'model_state_dict': model.state_dict(), 'step': step}, out_dir / 'fix_policy_final.pt')
	print(f"DONE: {step} steps in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == '__main__':
	main()
