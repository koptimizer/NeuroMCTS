"""v19: accuracy per millisecond -- the axis v15-v18 never measured.

Everything so far compared learning and classical reasoning on ACCURACY alone, and on
that axis classical probing wins or ties (v18: LP-probe union recovers 82.8-92.0% of
FORCED variables vs the model's 87.9-90.4%). But probing pays one solve PER VARIABLE,
while the network answers every variable in a single forward pass. Inside a search tree
visiting thousands of nodes, cost is not a footnote -- it decides what is usable at all.

This is the learn2branch situation: strong branching is the accurate expert and is too
slow to run at every node, so a network is trained to imitate it and delivers most of the
benefit at a fraction of the cost. If the same holds here, learning's role is amortizing
expensive reasoning rather than supplying information classical methods cannot reach.

Measured per node, single-threaded, on identical states:
  wall-clock for a full-instance verdict (all variables answered)
  recall/accuracy on the ground-truth FORCED set
  the ratio of the two
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
import numpy as np
import torch

import LPneuroBLS_v7 as m
from v15_train_marginal import MarginalNet, featurize
from v18_probing_recall import prop_probe, lp_probe

torch.set_num_threads(1)
EPS = 1e-9


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--test', required=True)
	ap.add_argument('--ckpt', default='../../runs/v17/conditional.pt')
	ap.add_argument('--depths', type=int, nargs='+', default=[5, 6, 7, 8])
	ap.add_argument('--per_depth', type=int, default=50)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	device = torch.device('cpu')
	model = MarginalNet()
	st = torch.load(a.ckpt, map_location=device, weights_only=False)
	model.load_state_dict(st['model_state_dict'])
	model.eval()

	recs = json.load(open(a.test))
	print("=" * 104)
	print("v19: cost vs accuracy per node (single thread, whole-instance verdict)")
	print("=" * 104)
	print(f"{'depth':>6}{'n_free':>8}{'':>3}"
	      f"{'model ms':>10}{'model acc':>11}{'':>3}"
	      f"{'LP-probe ms':>13}{'LP-probe acc':>14}{'':>3}"
	      f"{'speedup':>9}{'acc delta':>11}")
	table = {}
	for d in a.depths:
		sel = [r for r in recs if r['depth'] == d][:a.per_depth]
		if not sel:
			continue
		t_model, t_lp, a_model, a_lp, nfree = [], [], [], [], []
		for r in sel:
			A = np.array(r['A'], dtype=np.int64)
			b = np.array(r['b'], dtype=np.int64)
			x_gt = np.array(r['x'], dtype=np.int64)
			p_true = np.array(r['marginals'])
			forced = (p_true < EPS) | (p_true > 1 - EPS)
			if forced.sum() == 0:
				continue
			forced_val = (p_true > 0.5).astype(np.int64)
			f = featurize(r, device)
			if f is None:
				continue
			nfree.append(A.shape[1])

			# the network answers every variable in one pass
			t0 = time.perf_counter()
			with torch.no_grad():
				p = torch.sigmoid(model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy()
			t_model.append(time.perf_counter() - t0)
			a_model.append(((p >= 0.5).astype(np.int64)[forced] == x_gt[forced]).mean())

			# probing pays one LP solve per variable to reach the same coverage
			t0 = time.perf_counter()
			det = np.zeros(A.shape[1], dtype=bool)
			for j in range(A.shape[1]):
				det[j] = lp_probe(A, b, j, 1 - forced_val[j])
			t_lp.append(time.perf_counter() - t0)
			# probing is only ever right where it fires; elsewhere it abstains
			a_lp.append((det & forced).sum() / forced.sum())

		row = dict(n_vars=float(np.mean(nfree)),
		            model_ms=1000 * float(np.mean(t_model)), model_acc=float(np.mean(a_model)),
		            lp_ms=1000 * float(np.mean(t_lp)), lp_acc=float(np.mean(a_lp)))
		row['speedup'] = row['lp_ms'] / max(1e-9, row['model_ms'])
		table[d] = row
		print(f"{d:>6}{row['n_vars']:>8.1f}{'':>3}"
		      f"{row['model_ms']:>10.2f}{100*row['model_acc']:>10.1f}%{'':>3}"
		      f"{row['lp_ms']:>13.2f}{100*row['lp_acc']:>13.1f}%{'':>3}"
		      f"{row['speedup']:>8.1f}x{100*(row['model_acc']-row['lp_acc']):>10.1f}%")

	print("\nmodel acc  : fraction of FORCED variables the network's argmax gets right")
	print("LP-probe acc: fraction of FORCED variables probing actually detects (it abstains elsewhere)")
	json.dump(table, open(a.out, 'w'), indent=2)
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
