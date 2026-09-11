"""v18: decompose the depth 6-7 gap that neither LP nor the learned model closes.

At those depths ~2 solutions survive, so every free variable falls into one of two kinds:
  FORCED : all surviving solutions agree on it (p_j is exactly 0 or 1). Logically
           determined by the prefix, so a perfect reasoner gets it right.
  FREE   : surviving solutions disagree (p_j strictly between 0 and 1). Genuinely
           undetermined; nobody can beat max(p_j, 1-p_j) on it.

The ceiling is therefore (fraction forced) + (fraction free) x (their own max marginal),
and any shortfall against it must come from missing FORCED variables. This script
measures exactly that split for each predictor, so the 12%p gap can be attributed rather
than guessed at:

  low accuracy on FORCED -> reasoning failure; the information is derivable and missed
  low accuracy on FREE   -> expected and irreducible, not a defect

Calibration is also reported: a model that is confidently wrong on FREE variables is
failing in a different (and more fixable) way than one that is simply uncertain.
"""
# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import numpy as np
import torch

import LPneuroBLS_v7 as m
from v15_train_marginal import MarginalNet, featurize

torch.set_num_threads(1)
EPS = 1e-9


def propagation_forced(A, b, n):
	# which variables plain constraint propagation can pin down at this node
	env = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
	env.propagate_constraints().apply_probing()
	return env.assignment != -1, env.assignment


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--test', required=True)
	ap.add_argument('--ckpt', default='../../runs/v17/conditional.pt')
	ap.add_argument('--depths', type=int, nargs='+', default=[5, 6, 7, 8])
	ap.add_argument('--limit', type=int, default=3000)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	device = torch.device('cpu')
	model = MarginalNet()
	st = torch.load(a.ckpt, map_location=device, weights_only=False)
	model.load_state_dict(st['model_state_dict'])
	model.eval()

	recs = [r for r in json.load(open(a.test))[:a.limit] if r['depth'] in a.depths]
	print(f"diagnosing {len(recs)} states at depths {a.depths}\n", flush=True)

	agg = {d: dict(frac_forced=[], acc_forced_model=[], acc_free_model=[],
	                acc_forced_lp=[], lp_integral_recall=[], prop_recall=[],
	                conf_forced=[], conf_free=[]) for d in a.depths}

	with torch.no_grad():
		for r in recs:
			f = featurize(r, device)
			if f is None:
				continue
			A = np.array(r['A'], dtype=np.int64)
			b = np.array(r['b'], dtype=np.int64)
			x_gt = np.array(r['x'], dtype=np.int64)
			p_true = np.array(r['marginals'])
			d = r['depth']

			forced = (p_true < EPS) | (p_true > 1 - EPS)
			free = ~forced
			if forced.sum() == 0:
				continue

			p = torch.sigmoid(model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy()
			pred = (p >= 0.5).astype(np.int64)
			x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))

			e = agg[d]
			e['frac_forced'].append(forced.mean())
			e['acc_forced_model'].append((pred[forced] == x_gt[forced]).mean())
			if free.sum():
				e['acc_free_model'].append((pred[free] == x_gt[free]).mean())
				e['conf_free'].append(np.abs(p[free] - 0.5).mean())
			e['conf_forced'].append(np.abs(p[forced] - 0.5).mean())

			if ok:
				lp_pred = (np.clip(x_lp, 0, 1) >= 0.5).astype(np.int64)
				e['acc_forced_lp'].append((lp_pred[forced] == x_gt[forced]).mean())
				lp_int = (x_lp < 1e-6) | (x_lp > 1 - 1e-6)
				e['lp_integral_recall'].append((lp_int & forced).sum() / max(1, forced.sum()))

			pmask, _ = propagation_forced(A, b, A.shape[1])
			e['prop_recall'].append((pmask & forced).sum() / max(1, forced.sum()))

	print("=" * 100)
	print("v18: where the depth 6-7 gap lives (FORCED = determined by the prefix, FREE = genuinely 50/50)")
	print("=" * 100)
	print(f"{'depth':>6}{'%FORCED':>10}{'model acc':>12}{'LP acc':>9}"
	      f"{'  |':>4}{'prop detects':>14}{'LP integral':>13}{'  |':>4}{'model acc':>11}{'model conf':>12}")
	print(f"{'':>6}{'of vars':>10}{'on FORCED':>12}{'on FORCED':>9}{'':>4}{'of FORCED':>14}"
	      f"{'of FORCED':>13}{'':>4}{'on FREE':>11}{'on FREE':>12}")
	print("-" * 100)
	table = {}
	for d in a.depths:
		e = agg[d]
		if not e['frac_forced']:
			continue
		row = {k: float(np.mean(v)) for k, v in e.items() if v}
		table[d] = row
		print(f"{d:>6}{100*row['frac_forced']:>9.1f}%{100*row['acc_forced_model']:>11.1f}%"
		      f"{100*row['acc_forced_lp']:>8.1f}%{'':>4}{100*row['prop_recall']:>13.1f}%"
		      f"{100*row['lp_integral_recall']:>12.1f}%{'':>4}"
		      f"{100*row['acc_free_model']:>10.1f}%{row['conf_free']:>12.3f}")
	print("\nreading: 'prop detects' and 'LP integral' are recall on FORCED variables --")
	print("how many of the logically determined variables each classical method even flags.")
	json.dump(table, open(a.out, 'w'), indent=2)
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
