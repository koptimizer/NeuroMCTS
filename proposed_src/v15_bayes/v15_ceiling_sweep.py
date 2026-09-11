"""Does a learned predictor follow the Bayes ceiling when the ceiling is raised?

Holding n=25 and raising m shrinks |S| and lifts the ceiling from ~70% to 100%. This
trains the same model at each m and plots accuracy against the ceiling it faces:

  tracks the ceiling -> solution multiplicity was the binding constraint, and this
                        project's 11 negative results were largely a property of the
                        instance family rather than of learning per se
  stays flat         -> the information is there (ceiling 100% at m=20) but a bounded
                        forward pass cannot extract it: a computational wall

Baselines (LP relaxation, the existing multi-head best.pth) are scored on the same
instances so the comparison is not just model-vs-ceiling.
"""
# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
from pathlib import Path
import numpy as np
import torch

import LPneuroBLS_v7 as m
from v14_core import gnn_marginals
from v15_train_marginal import MarginalNet, featurize, train_arm

torch.set_num_threads(1)


def ceiling_of(recs):
	return float(np.mean([np.maximum(np.array(r['marginals']), 1 - np.array(r['marginals'])).mean()
	                       for r in recs]))


def eval_preds(p, x_gt, p_true):
	pred = (p >= 0.5).astype(np.int64)
	out = {'all': float((pred == x_gt).mean()),
	        'l1': float(np.abs(p - p_true).mean())}
	for k in (3, 5):
		o = np.argsort(-np.abs(p - 0.5))[:k]
		out[f'top{k}'] = float((pred[o] == x_gt[o]).mean())
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sets', nargs='+', required=True, help='m:path pairs, e.g. 10:../runs/v15/a.json')
	ap.add_argument('--n_train', type=int, default=600)
	ap.add_argument('--n_val', type=int, default=200)
	ap.add_argument('--epochs', type=int, default=30)
	ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	device = torch.device('cpu')
	old = m.BipartiteGNN(hidden_dim=256, num_layers=8)
	ck = torch.load('../../runs/best.pth', map_location=device, weights_only=False)
	old.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck)
	old.eval()

	table = []
	for spec in a.sets:
		mm, path = spec.split(':', 1)
		recs = json.load(open(path))
		rng = np.random.default_rng(a.seed)
		rng.shuffle(recs)
		recs = recs[:a.n_train + a.n_val]
		feats = [f for f in (featurize(r, device) for r in recs) if f is not None]
		keep = recs[:len(feats)]
		tr, va = feats[:a.n_train], feats[a.n_train:]
		va_recs = keep[a.n_train:]
		ns = np.array([r['n_sols'] for r in keep])
		ceil = ceiling_of(va_recs)
		print(f"\n=== m={mm} (n=25)  |S| median={np.median(ns):.0f}  "
		      f"unique={100*np.mean(ns==1):.0f}%  ceiling={100*ceil:.1f}% ===", flush=True)

		model, _ = train_arm('hard', tr, va, a.epochs, a.lr, a.seed, a.epochs)

		agg = {k: {'all': [], 'top3': [], 'top5': [], 'l1': []} for k in ('v15', 'old_gnn', 'lp')}
		model.eval()
		with torch.no_grad():
			for f, r in zip(va, va_recs):
				A = np.array(r['A'], dtype=np.int64)
				b = np.array(r['b'], dtype=np.int64)
				x_gt = np.array(r['x'], dtype=np.int64)
				p_true = np.array(r['marginals'])
				x_lp, _ = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
				preds = {
					'v15': torch.sigmoid(model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy(),
					'old_gnn': gnn_marginals(old, A, b, x_lp.astype(np.float64), device),
					'lp': np.clip(x_lp, 0, 1),
				}
				for key, p in preds.items():
					e = eval_preds(p, x_gt, p_true)
					for kk in agg[key]:
						agg[key][kk].append(e[kk])

		row = dict(m=int(mm), n_sols_median=float(np.median(ns)),
		            unique_frac=float(np.mean(ns == 1)), ceiling=ceil,
		            **{f'{key}_{kk}': float(np.mean(v)) for key, d in agg.items() for kk, v in d.items()})
		table.append(row)
		torch.save({'model_state_dict': model.state_dict()}, f'../../runs/v15/marginal_m{mm}.pt')

	print("\n" + "=" * 96)
	print("v15 ceiling sweep (n=25, m varied): does accuracy follow the ceiling?")
	print("=" * 96)
	print(f"{'m':>3}{'|S|med':>8}{'unique':>8}{'ceiling':>10}{'v15':>9}{'gap':>8}"
	      f"{'old GNN':>10}{'LP':>8}{'v15 top3':>10}{'ceil top3':>11}")
	for r in table:
		print(f"{r['m']:>3}{r['n_sols_median']:>8.0f}{100*r['unique_frac']:>7.0f}%"
		      f"{100*r['ceiling']:>9.1f}%{100*r['v15_all']:>8.1f}%"
		      f"{100*(r['ceiling']-r['v15_all']):>7.1f}%{100*r['old_gnn_all']:>9.1f}%"
		      f"{100*r['lp_all']:>7.1f}%{100*r['v15_top3']:>9.1f}%{'':>11}")
	json.dump(table, open(a.out, 'w'), indent=2)
	print(f"\nsaved: {a.out}")


if __name__ == '__main__':
	main()
