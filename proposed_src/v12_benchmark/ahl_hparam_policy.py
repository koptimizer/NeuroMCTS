"""Train a small MLP to predict, per instance, which AHL/BKZ block size solves it.
Policy is evaluated offline against the swept ground truth (held-out split) --
no re-running AHL needed for evaluation.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

BLOCK_GRID = [6, 8, 10, 15, 20, 25, 30, 40]
SAFE_BLOCKS = {
	'10x25': [6, 8, 10, 15, 20, 25, 30, 40],
	'20x50': [6, 8, 10, 15, 20, 25, 30, 40],
	'40x100': [6, 8, 10, 15, 20, 25, 30],
	'60x150': [6, 8, 10, 15, 20],
	'80x200': [6, 8, 10, 15],
}
# per-size max-safe-block heuristic, from the pre-sweep timing calibration
SIZE_MAX_SAFE = {'10x25': 40, '20x50': 40, '40x100': 30, '60x150': 20, '80x200': 15}


def load_records(path):
	with open(path) as f:
		return json.load(f)


def to_arrays(records):
	feats, labels, mask, times, sizes = [], [], [], [], []
	for r in records:
		ft = r['feat']
		feats.append([ft['m'], ft['n'], ft['m'] / ft['n'], ft['density'], ft['vertex_spread'], ft['b_tightness']])
		lab = np.zeros(len(BLOCK_GRID), dtype=np.float32)
		msk = np.zeros(len(BLOCK_GRID), dtype=np.float32)
		tim = np.full(len(BLOCK_GRID), np.nan, dtype=np.float32)
		for bi, block in enumerate(BLOCK_GRID):
			key = str(block)
			if key in r['results']:
				msk[bi] = 1.0
				lab[bi] = float(r['results'][key]['solved'])
				tim[bi] = r['results'][key]['time']
		labels.append(lab)
		mask.append(msk)
		times.append(tim)
		sizes.append(r['size'])
	return np.array(feats, dtype=np.float32), np.array(labels), np.array(mask), np.array(times), sizes


class Policy(nn.Module):
	def __init__(self, in_dim, n_blocks, hidden=64):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, n_blocks),
		)

	def forward(self, x):
		return self.net(x)


def train(feats, labels, mask, epochs, lr, seed):
	torch.manual_seed(seed)
	mu, sd = feats.mean(0), feats.std(0) + 1e-6
	feats_n = (feats - mu) / sd
	x = torch.tensor(feats_n, dtype=torch.float32)
	y = torch.tensor(labels, dtype=torch.float32)
	m = torch.tensor(mask, dtype=torch.float32)

	model = Policy(x.shape[1], len(BLOCK_GRID))
	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
	lossf = nn.BCEWithLogitsLoss(reduction='none')
	for ep in range(epochs):
		opt.zero_grad()
		logit = model(x)
		loss = (lossf(logit, y) * m).sum() / m.sum()
		loss.backward()
		opt.step()
		if (ep + 1) % 50 == 0:
			print(f"epoch {ep+1}/{epochs}  loss={loss.item():.4f}")
	return model, mu, sd


def policy_choose_block(model, mu, sd, feat_row, size):
	x = torch.tensor(((feat_row - mu) / sd)[None, :], dtype=torch.float32)
	with torch.no_grad():
		prob = torch.sigmoid(model(x))[0].numpy()
	candidates = SAFE_BLOCKS[size]
	best_block, best_p = None, -1
	for bi, block in enumerate(BLOCK_GRID):
		if block not in candidates:
			continue
		p = prob[bi]
		if p > best_p + 1e-6 or (abs(p - best_p) <= 1e-6 and (best_block is None or block < best_block)):
			best_block, best_p = block, p
	return best_block


def best_fixed_block(records, idxs, labels, size):
	block_grid_idx = {b: bi for bi, b in enumerate(BLOCK_GRID)}
	best_block, best_rate = None, -1
	for block in SAFE_BLOCKS[size]:
		bi = block_grid_idx[block]
		rows = [i for i in idxs if records[i]['size'] == size]
		if not rows:
			continue
		rate = np.mean([labels[i, bi] for i in rows])
		if rate > best_rate:
			best_block, best_rate = block, rate
	return best_block


def evaluate(records, idxs, model, mu, sd, feats, times, labels, fixed_block_by_size):
	sizes_seen = {}
	block_grid_idx = {b: bi for bi, b in enumerate(BLOCK_GRID)}
	for i in idxs:
		size = records[i]['size']

		policy_block = policy_choose_block(model, mu, sd, feats[i], size)
		default_block = SIZE_MAX_SAFE[size]
		if default_block not in SAFE_BLOCKS[size]:
			default_block = max(SAFE_BLOCKS[size])
		fixed_block = fixed_block_by_size[size]

		pi, di, fi = block_grid_idx[policy_block], block_grid_idx[default_block], block_grid_idx[fixed_block]
		p_solved, p_time = bool(labels[i, pi]), times[i, pi]
		d_solved, d_time = bool(labels[i, di]), times[i, di]
		f_solved, f_time = bool(labels[i, fi]), times[i, fi]

		s = sizes_seen.setdefault(size, dict(n=0, policy_solved=0, default_solved=0, fixed_solved=0,
		                                      policy_time=[], default_time=[], fixed_time=[]))
		s['n'] += 1
		s['policy_solved'] += int(p_solved)
		s['default_solved'] += int(d_solved)
		s['fixed_solved'] += int(f_solved)
		if p_solved:
			s['policy_time'].append(p_time)
		if d_solved:
			s['default_time'].append(d_time)
		if f_solved:
			s['fixed_time'].append(f_time)
	return sizes_seen


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sweep', nargs='+', default=['../../runs/ahl_hparam/sweep.json'])
	ap.add_argument('--epochs', type=int, default=300)
	ap.add_argument('--lr', type=float, default=1e-2)
	ap.add_argument('--val_frac', type=float, default=0.25)
	ap.add_argument('--seed', type=int, default=0)
	a = ap.parse_args()

	# merge sweep files by instance path; a later file's per-block results override/extend
	# an earlier file's (used to fold a higher-tries boost sweep into the base sweep)
	merged = {}
	for path in a.sweep:
		for r in load_records(path):
			key = r['path']
			if key not in merged:
				merged[key] = r
			else:
				merged[key]['results'].update(r['results'])
	records = list(merged.values())

	feats, labels, mask, times, sizes = to_arrays(records)
	n = len(records)
	rng = np.random.default_rng(a.seed)

	train_idx, val_idx = [], []
	by_size = {}
	for i, s in enumerate(sizes):
		by_size.setdefault(s, []).append(i)
	for s, idxs in by_size.items():
		idxs = np.array(idxs)
		rng.shuffle(idxs)
		n_val = max(1, int(len(idxs) * a.val_frac))
		val_idx += idxs[:n_val].tolist()
		train_idx += idxs[n_val:].tolist()
	print(f"train={len(train_idx)} val={len(val_idx)}")

	model, mu, sd = train(feats[train_idx], labels[train_idx], mask[train_idx], a.epochs, a.lr, a.seed)

	fixed_block_by_size = {}
	for size in SAFE_BLOCKS:
		fixed_block_by_size[size] = best_fixed_block(records, train_idx, labels, size)
	print("best single fixed block per size (chosen on train split):", fixed_block_by_size)

	print("\n=== held-out evaluation: learned per-instance policy vs two fixed-block baselines ===")
	print("  default   = current pipeline default (largest block still fast/safe at this size)")
	print("  best-fixed= single block that solved most often on TRAIN split (no per-instance choice)")
	res = evaluate(records, val_idx, model, mu, sd, feats, times, labels, fixed_block_by_size)
	for size in ['10x25', '20x50', '40x100', '60x150', '80x200']:
		if size not in res:
			continue
		d = res[size]
		p_avg = np.mean(d['policy_time']) if d['policy_time'] else float('nan')
		def_avg = np.mean(d['default_time']) if d['default_time'] else float('nan')
		f_avg = np.mean(d['fixed_time']) if d['fixed_time'] else float('nan')
		print(f"{size:8s} n={d['n']:3d}  "
		      f"policy={d['policy_solved']}/{d['n']}({p_avg:.3f}s)  "
		      f"default(b={SIZE_MAX_SAFE[size]})={d['default_solved']}/{d['n']}({def_avg:.3f}s)  "
		      f"best-fixed(b={fixed_block_by_size[size]})={d['fixed_solved']}/{d['n']}({f_avg:.3f}s)")

	torch.save({'state_dict': model.state_dict(), 'mu': mu.tolist(), 'sd': sd.tolist()}, '../../runs/ahl_hparam/policy.pt')
	print("saved ../../runs/ahl_hparam/policy.pt")


if __name__ == '__main__':
	main()
