"""v24: 'where the network helps' table for several checkpoints on the same states.

Same construction as v23_where_analysis.py (prefix of the LP-confidence order fixed to the
planted solution on the 21x60 evaluation instances, exact conditional marginals where the
enumeration completes), extended to all 30 instances, a 60 s enumeration limit, and a list of
checkpoints scored on identical states so that the comparison between them is paired.
"""
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
from v15_train_marginal import MarginalNet, featurize
from v15_bayes_limit import enumerate_solutions

torch.set_num_threads(1)


def load(ckpt):
	net = MarginalNet()
	net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
	net.eval()
	return net


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', default='../../instances/v22test_21x60')
	ap.add_argument('--ckpts', nargs='+', required=True)
	ap.add_argument('--names', nargs='+', required=True)
	ap.add_argument('--sizes', type=int, nargs='+', default=[20, 25, 30, 40, 50, 60])
	ap.add_argument('--time_limit', type=float, default=60.0)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	nets = [load(c) for c in a.ckpts]
	dev = torch.device('cpu')
	rows = []
	for f in sorted(Path(a.data_dir).glob('*.json')):
		d = json.load(open(f))
		A0 = np.array(d['A'], np.int64); b0 = np.array(d['b'], np.int64); x0 = np.array(d['x'], np.int64)
		x_lp0, ok = m.solve_lp_relaxation(A0.astype(np.int32), b0.astype(np.int32))
		if not ok:
			continue
		order = np.argsort(-np.abs(x_lp0 - 0.5))
		n = A0.shape[1]
		for nf in a.sizes:
			dep = n - nf
			fx, keep = order[:dep], order[dep:]
			Ar = A0[:, keep]; br = b0 - A0[:, fx].dot(x0[fx])
			live = Ar.sum(1) > 0
			Ar, br = Ar[live], br[live]
			S, cok = enumerate_solutions(Ar, br, 200000, a.time_limit)
			if not cok or len(S) == 0:
				continue
			p_true = S.mean(0)
			forced = (p_true < 1e-9) | (p_true > 1 - 1e-9)
			if forced.sum() == 0:
				continue
			fv = (p_true > 0.5).astype(np.int64)
			rec = dict(A=Ar.tolist(), b=br.tolist(), x=S[0].tolist(), marginals=p_true.tolist())
			fe = featurize(rec, dev)
			if fe is None:
				continue
			x_lp = np.clip(fe['xv'][:, 0].numpy(), 0, 1)
			row = dict(inst=f.name, n_free=int(nf), m_rows=int(live.sum()), n_sols=int(len(S)),
			           n_forced=int(forced.sum()),
			           lp_forced=float(((x_lp >= 0.5).astype(np.int64)[forced] == fv[forced]).mean()))
			for name, net in zip(a.names, nets):
				with torch.no_grad():
					pm = torch.sigmoid(net(fe['xv'], fe['xc'], fe['ev2c'], fe['ec2v'])).numpy()
				row[f'{name}_forced'] = float(((pm >= 0.5).astype(np.int64)[forced] == fv[forced]).mean())
				row[f'{name}_corr'] = float(np.corrcoef(pm, x_lp)[0, 1])
				row[f'{name}_l1'] = float(np.abs(pm - p_true).mean())
			rows.append(row)
	json.dump(rows, open(a.out, 'w'), indent=1)
	print(f"{'n_free':>7}{'states':>8}{'m/|K|':>7}{'LP':>8}" + ''.join(f"{nm:>9}{'gain':>7}{'corr':>7}" for nm in a.names))
	for nf in a.sizes:
		rs = [r for r in rows if r['n_free'] == nf]
		if not rs:
			continue
		lp = np.mean([r['lp_forced'] for r in rs])
		line = f"{nf:>7}{len(rs):>8}{np.mean([r['m_rows']/nf for r in rs]):>7.2f}{100*lp:>7.1f}%"
		for nm in a.names:
			mo = np.mean([r[f'{nm}_forced'] for r in rs])
			line += f"{100*mo:>8.1f}%{100*(mo-lp):>+7.1f}{np.mean([r[f'{nm}_corr'] for r in rs]):>7.3f}"
		print(line)
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
