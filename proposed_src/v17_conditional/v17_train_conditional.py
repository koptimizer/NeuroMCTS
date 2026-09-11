"""v17: train and evaluate the conditional (in-tree) marginal predictor.

Trains the v15 architecture on states drawn from inside the search tree rather than only
at the root, then reports accuracy against the conditional Bayes ceiling BROKEN DOWN BY
DEPTH. The depth breakdown is the point: v16 located the useful window at depth 6-10, so
a single averaged number would hide whether the model actually captures it.

Baselines scored on identical states:
  LP relaxation on the reduced instance  (the classical signal, recomputed in-tree)
  the root-only v15 model                (does training on in-tree states matter?)
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
import torch.nn.functional as F

import LPneuroBLS_v7 as m
from v15_train_marginal import MarginalNet, featurize

torch.set_num_threads(1)


def evaluate_by_depth(model, data, recs, ref_model=None):
	rows = {}
	model.eval()
	with torch.no_grad():
		for f, r in zip(data, recs):
			d = r['depth']
			p = torch.sigmoid(model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy()
			x_gt = np.array(r['x'], dtype=np.int64)
			p_true = np.array(r['marginals'])
			A = np.array(r['A'], dtype=np.int64)
			b = np.array(r['b'], dtype=np.int64)
			x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
			entry = rows.setdefault(d, {'ceil': [], 'model': [], 'lp': [], 'ref': [], 'l1': []})
			entry['ceil'].append(np.maximum(p_true, 1 - p_true).mean())
			entry['model'].append(((p >= 0.5).astype(np.int64) == x_gt).mean())
			entry['l1'].append(np.abs(p - p_true).mean())
			if ok:
				entry['lp'].append(((np.clip(x_lp, 0, 1) >= 0.5).astype(np.int64) == x_gt).mean())
			if ref_model is not None:
				pr = torch.sigmoid(ref_model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy()
				entry['ref'].append(((pr >= 0.5).astype(np.int64) == x_gt).mean())
	model.train()
	return rows


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--train', required=True)
	ap.add_argument('--test', required=True)
	ap.add_argument('--n_train', type=int, default=8000)
	ap.add_argument('--n_test', type=int, default=2000)
	ap.add_argument('--epochs', type=int, default=25)
	ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--target', choices=['soft', 'hard'], default='soft')
	ap.add_argument('--ref_ckpt', default='../../runs/v15/marginal_hard.pt')
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	device = torch.device('cpu')
	tr_recs = json.load(open(a.train))[:a.n_train]
	te_recs = json.load(open(a.test))[:a.n_test]
	tr = [f for f in (featurize(r, device) for r in tr_recs) if f is not None]
	te_pairs = [(f, r) for f, r in ((featurize(r, device), r) for r in te_recs) if f is not None]
	te, te_recs = [p[0] for p in te_pairs], [p[1] for p in te_pairs]
	print(f"train={len(tr)} in-tree states, test={len(te)}", flush=True)

	ref = MarginalNet()
	st = torch.load(a.ref_ckpt, map_location=device, weights_only=False)
	ref.load_state_dict(st['model_state_dict'])
	ref.eval()

	model = MarginalNet()
	opt = torch.optim.Adam(model.parameters(), lr=a.lr)
	torch.manual_seed(0)
	for ep in range(1, a.epochs + 1):
		order = np.random.permutation(len(tr))
		tot = 0.0
		for i in order:
			f = tr[i]
			logits = model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])
			loss = F.binary_cross_entropy_with_logits(logits, f[a.target])
			opt.zero_grad(); loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
			tot += loss.item()
		if ep % 5 == 0 or ep == a.epochs:
			print(f"  ep{ep:>3} loss={tot/len(tr):.4f}", flush=True)

	rows = evaluate_by_depth(model, te, te_recs, ref)
	print("\n" + "=" * 88)
	print("v17: in-tree conditional prediction, by depth (n=25 hard family)")
	print("=" * 88)
	print(f"{'depth':>6}{'states':>8}{'ceiling':>10}{'v17 model':>11}{'gap':>8}"
	      f"{'v15 root-only':>15}{'LP in-tree':>12}")
	for d in sorted(rows):
		e = rows[d]
		ce, mo = np.mean(e['ceil']), np.mean(e['model'])
		rf = np.mean(e['ref']) if e['ref'] else float('nan')
		lp = np.mean(e['lp']) if e['lp'] else float('nan')
		print(f"{d:>6}{len(e['ceil']):>8}{100*ce:>9.1f}%{100*mo:>10.1f}%"
		      f"{100*(ce-mo):>7.1f}%{100*rf:>14.1f}%{100*lp:>11.1f}%")
	allc = np.mean([v for e in rows.values() for v in e['ceil']])
	allm = np.mean([v for e in rows.values() for v in e['model']])
	allr = np.mean([v for e in rows.values() for v in e['ref']])
	alll = np.mean([v for e in rows.values() for v in e['lp']])
	print("-" * 88)
	print(f"{'ALL':>6}{'':>8}{100*allc:>9.1f}%{100*allm:>10.1f}%{100*(allc-allm):>7.1f}%"
	      f"{100*allr:>14.1f}%{100*alll:>11.1f}%")

	out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
	torch.save({'model_state_dict': model.state_dict()}, out_dir / 'conditional.pt')
	json.dump({str(d): {k: float(np.mean(v)) for k, v in e.items() if v} for d, e in rows.items()},
	           open(out_dir / 'v17_by_depth.json', 'w'), indent=2)
	print(f"\nsaved: {out_dir}/conditional.pt")


if __name__ == '__main__':
	main()
