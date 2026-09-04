"""Train a small MLP to predict, per instance, which CP-SAT solver config resolves it fastest.
Policy is evaluated offline against the swept ground truth (held-out split) -- no re-solving needed."""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

CONFIG_NAMES = ['default', 'linearization0', 'linearization2', 'portfolio_search', 'lp_search', 'no_symmetry']


def load_records(path):
	with open(path) as f:
		return json.load(f)


def to_arrays(records):
	feats, labels, times, sizes, gt_feas = [], [], [], [], []
	for r in records:
		ft = r['feat']
		feats.append([ft['m'], ft['n'], ft['m'] / ft['n'], ft['density'], ft['vertex_spread'], ft['b_tightness']])
		lab = np.zeros(len(CONFIG_NAMES), dtype=np.float32)
		tim = np.full(len(CONFIG_NAMES), np.nan, dtype=np.float32)
		for ci, name in enumerate(CONFIG_NAMES):
			res = r['results'][name]
			lab[ci] = float(res['solved'])
			tim[ci] = res['time']
		labels.append(lab)
		times.append(tim)
		sizes.append(r['size'])
		gt_feas.append(r['gt_feasible'])
	return np.array(feats, dtype=np.float32), np.array(labels), np.array(times), sizes, gt_feas


class Policy(nn.Module):
	def __init__(self, in_dim, n_configs, hidden=64):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, n_configs),
		)

	def forward(self, x):
		return self.net(x)


def train(feats, labels, epochs, lr, seed):
	torch.manual_seed(seed)
	mu, sd = feats.mean(0), feats.std(0) + 1e-6
	feats_n = (feats - mu) / sd
	x = torch.tensor(feats_n, dtype=torch.float32)
	y = torch.tensor(labels, dtype=torch.float32)

	model = Policy(x.shape[1], len(CONFIG_NAMES))
	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
	lossf = nn.BCEWithLogitsLoss()
	for ep in range(epochs):
		opt.zero_grad()
		logit = model(x)
		loss = lossf(logit, y)
		loss.backward()
		opt.step()
		if (ep + 1) % 50 == 0:
			print(f"epoch {ep+1}/{epochs}  loss={loss.item():.4f}")
	return model, mu, sd


def policy_choose_config(model, mu, sd, feat_row):
	x = torch.tensor(((feat_row - mu) / sd)[None, :], dtype=torch.float32)
	with torch.no_grad():
		prob = torch.sigmoid(model(x))[0].numpy()
	ci = int(np.argmax(prob))
	return CONFIG_NAMES[ci]


def best_fixed_config(records, idxs, labels, size):
	best_name, best_rate = None, -1
	for ci, name in enumerate(CONFIG_NAMES):
		rows = [i for i in idxs if records[i]['size'] == size]
		if not rows:
			continue
		rate = np.mean([labels[i, ci] for i in rows])
		if rate > best_rate:
			best_name, best_rate = name, rate
	return best_name


def evaluate(records, idxs, model, mu, sd, feats, times, labels, fixed_by_size):
	name_to_idx = {n: i for i, n in enumerate(CONFIG_NAMES)}
	sizes_seen = {}
	for i in idxs:
		size = records[i]['size']
		policy_name = policy_choose_config(model, mu, sd, feats[i])
		default_name = 'default'
		fixed_name = fixed_by_size[size]

		pi, di, fi = name_to_idx[policy_name], name_to_idx[default_name], name_to_idx[fixed_name]
		p_solved, p_time = bool(labels[i, pi]), times[i, pi]
		d_solved, d_time = bool(labels[i, di]), times[i, di]
		f_solved, f_time = bool(labels[i, fi]), times[i, fi]

		s = sizes_seen.setdefault(size, dict(n=0, policy_solved=0, default_solved=0, fixed_solved=0,
		                                      policy_time=[], default_time=[], fixed_time=[]))
		s['n'] += 1
		s['policy_solved'] += int(p_solved)
		s['default_solved'] += int(d_solved)
		s['fixed_solved'] += int(f_solved)
		s['policy_time'].append(p_time)
		s['default_time'].append(d_time)
		s['fixed_time'].append(f_time)
	return sizes_seen


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sweep', default='../runs/cpsat_hparam/sweep.json')
	ap.add_argument('--epochs', type=int, default=300)
	ap.add_argument('--lr', type=float, default=1e-2)
	ap.add_argument('--val_frac', type=float, default=0.25)
	ap.add_argument('--seed', type=int, default=0)
	a = ap.parse_args()

	records = load_records(a.sweep)
	feats, labels, times, sizes, gt_feas = to_arrays(records)
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

	model, mu, sd = train(feats[train_idx], labels[train_idx], a.epochs, a.lr, a.seed)

	fixed_by_size = {}
	for size in by_size:
		fixed_by_size[size] = best_fixed_config(records, train_idx, labels, size)
	print("best single fixed config per size (train split):", fixed_by_size)

	print("\n=== held-out evaluation: learned per-instance policy vs default vs best-fixed-config ===")
	res = evaluate(records, val_idx, model, mu, sd, feats, times, labels, fixed_by_size)
	for size in ['40x100', '60x150', '80x200']:
		if size not in res:
			continue
		d = res[size]
		p_avg = np.mean(d['policy_time'])
		def_avg = np.mean(d['default_time'])
		f_avg = np.mean(d['fixed_time'])
		print(f"{size:8s} n={d['n']:3d}  "
		      f"policy={d['policy_solved']}/{d['n']}({p_avg:.2f}s)  "
		      f"default={d['default_solved']}/{d['n']}({def_avg:.2f}s)  "
		      f"best-fixed({fixed_by_size[size]})={d['fixed_solved']}/{d['n']}({f_avg:.2f}s)")

	torch.save({'state_dict': model.state_dict(), 'mu': mu.tolist(), 'sd': sd.tolist()}, '../runs/cpsat_hparam/policy.pt')
	print("saved ../runs/cpsat_hparam/policy.pt")


if __name__ == '__main__':
	main()
