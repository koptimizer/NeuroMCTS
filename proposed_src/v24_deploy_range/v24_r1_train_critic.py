"""v24 R1: train the value head to predict log subtree cost; report held-out correlation.

Gate for R2: the critic is the baseline of the actor's advantage, so if it cannot rank decision
costs on held-out states (Spearman below 0.5) the advantage is noise and fine-tuning is not
attempted. A size-only baseline (log cost regressed on log n_free) is reported alongside so the
critic's value beyond 'bigger states are costlier' is visible.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

from v24_rl_core import load_pv
from v23_warmstart_cost import features

torch.set_num_threads(1)


def feats(recs):
	dev = torch.device('cpu')
	out = []
	for r in recs:
		A = np.array(r['A'], np.int64); b = np.array(r['b'], np.int64); x = np.array(r['x_lp'], np.float64)
		out.append((features(A, b, x, dev), math.log(r['cost']), r['n_free']))
	return out


def evaluate(net, data):
	net.eval()
	pred, true, nf = [], [], []
	with torch.no_grad():
		for f, y, n in data:
			pred.append(float(net(*f)[2])); true.append(y); nf.append(n)
	net.train()
	pred, true, nf = map(np.array, (pred, true, nf))
	# size-only baseline: linear fit of log cost on log n_free (fit on the same data, generous to it)
	X = np.stack([np.log(nf), np.ones_like(nf, dtype=float)], 1)
	coef = np.linalg.lstsq(X, true, rcond=None)[0]; base = X @ coef
	return dict(spearman=spearmanr(pred, true).correlation, pearson=pearsonr(pred, true)[0],
	            rmse=float(np.sqrt(np.mean((pred - true) ** 2))),
	            size_only_spearman=spearmanr(base, true).correlation,
	            size_only_rmse=float(np.sqrt(np.mean((base - true) ** 2))), pred=pred, true=true, nf=nf)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--train', required=True)
	ap.add_argument('--test', required=True)
	ap.add_argument('--ckpt', default='../../runs/v22/model_n25/conditional.pt')
	ap.add_argument('--epochs', type=int, default=15)
	ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--unfreeze', action='store_true', help='also fine-tune the trunk (default: value head only)')
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	net = load_pv(a.ckpt)
	tr = feats(json.load(open(a.train))); te = feats(json.load(open(a.test)))
	print(f"train={len(tr)} test={len(te)} decisions", flush=True)
	params = list(net.val.parameters()) + (list(net.base.parameters()) if a.unfreeze else [])
	opt = torch.optim.Adam(params, lr=a.lr)
	torch.manual_seed(0); rng = np.random.default_rng(0)
	net.train()
	for ep in range(1, a.epochs + 1):
		tot = 0.0
		for i in rng.permutation(len(tr)):
			f, y, _ = tr[i]
			loss = F.mse_loss(net(*f)[2], torch.tensor(y))
			opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
		ev = evaluate(net, te)
		print(f"  ep{ep:>3} train mse={tot/len(tr):.3f} | test spearman={ev['spearman']:.3f} pearson={ev['pearson']:.3f} rmse={ev['rmse']:.3f} "
		      f"| size-only spearman={ev['size_only_spearman']:.3f} rmse={ev['size_only_rmse']:.3f}", flush=True)
	ev = evaluate(net, te)
	print("\nby n_free of the decision node (test):")
	print(f"{'n_free':>8}{'n':>7}{'spearman':>10}{'size-only':>11}")
	for lo, hi in [(5, 15), (15, 25), (25, 35), (35, 45), (45, 56)]:
		sel = (ev['nf'] >= lo) & (ev['nf'] < hi)
		if sel.sum() > 20:
			print(f"{f'{lo}-{hi-1}':>8}{sel.sum():>7}{spearmanr(ev['pred'][sel], ev['true'][sel]).correlation:>10.3f}{spearmanr(np.log(ev['nf'][sel]), ev['true'][sel]).correlation:>11.3f}")
	Path(a.out).mkdir(parents=True, exist_ok=True)
	torch.save({'model_state_dict': net.state_dict()}, Path(a.out) / 'pv.pt')
	json.dump({k: float(v) for k, v in ev.items() if k in ('spearman', 'pearson', 'rmse', 'size_only_spearman', 'size_only_rmse')},
	          open(Path(a.out) / 'critic_eval.json', 'w'), indent=1)
	print(f"saved: {a.out}/pv.pt")


if __name__ == '__main__':
	main()
