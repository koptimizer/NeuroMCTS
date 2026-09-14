"""Per-node cost when both methods use a properly engineered LP backend.

The cost comparison in the paper measures LP-probing against a backend that rebuilds a
SCIP model from Python for every probe. Real solvers keep one LP and re-optimize from the
parent basis after a bound change, which is 10-100x cheaper, so that measurement
flatters the learned alternative. This re-measures both sides with a single model whose
variable bounds are changed in place (Gurobi dual simplex, warm started).

Both methods are charged everything they must pay to produce a whole-instance verdict:
  network : one LP for its input features, plus one forward pass
  probing : one LP per free variable
The network's feature LP is warm started too, so neither side is advantaged by the change.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].iterdir())
                 if p.is_dir() and not p.name.startswith('.')]

import numpy as np
import torch
import gurobipy as gp

from v15_train_marginal import MarginalNet

torch.set_num_threads(1)
EPS = 1e-9


class WarmLP:
	"""One LP per instance, re-optimized in place after bound changes.
	Mirrors how a branch-and-bound implementation reuses its relaxation instead of
	rebuilding it, which is the comparison the paper's cost table should have made."""

	def __init__(self, A, b, env):
		self.n = A.shape[1]
		self.mo = gp.Model(env=env)
		self.mo.Params.Method = 1                       # dual simplex: right for bound changes
		self.x = self.mo.addMVar(self.n, lb=0.0, ub=1.0)
		self.mo.addConstr(A.astype(float) @ self.x == b.astype(float))
		self.mo.setObjective(0)

	def relax(self):
		self.mo.optimize()
		if self.mo.Status != gp.GRB.OPTIMAL:
			return None
		return np.array(self.x.X, dtype=np.float64)

	def fix_infeasible(self, j, val):
		"""True if forcing x_j = val makes the relaxation infeasible."""
		lo, up = self.x[j].LB, self.x[j].UB
		self.x[j].LB = self.x[j].UB = float(val)
		self.mo.optimize()
		bad = self.mo.Status != gp.GRB.OPTIMAL
		self.x[j].LB, self.x[j].UB = lo, up
		return bad


def features(A, b, x_lp, device):
	mm, n = A.shape
	col_deg = (A != 0).sum(axis=0) / max(1, mm)
	frac = np.abs(x_lp - np.round(x_lp))
	r = np.maximum(A.sum(axis=1), 1)
	xv = torch.tensor(np.stack([x_lp, col_deg, frac], -1), dtype=torch.float, device=device)
	xc = torch.tensor(np.stack([b / r, r / max(1, n), (b - A.dot(x_lp)) / r], -1),
	                   dtype=torch.float, device=device)
	rows, cols = np.where(A == 1)
	return (xv, xc,
	         torch.tensor(np.array([cols, rows]), dtype=torch.long, device=device),
	         torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--test', required=True)
	ap.add_argument('--ckpt', required=True)
	ap.add_argument('--depths', type=int, nargs='+', default=[5, 6, 7, 8])
	ap.add_argument('--per_depth', type=int, default=40)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()
	device = torch.device('cpu')
	model = MarginalNet()
	model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=False)['model_state_dict'])
	model.eval()
	recs = json.load(open(a.test))

	print("=" * 96)
	print("Per-node cost with a warm-started LP backend (both methods)")
	print("=" * 96)
	print(f"{'depth':>6}{'n_free':>8}{'network ms':>12}{'net acc':>10}"
	      f"{'probe ms':>11}{'probe recall':>14}{'ratio':>8}")
	table = {}
	for d in a.depths:
		sel = [r for r in recs if r['depth'] == d][:a.per_depth]
		tn, tp, an, ap_, nf = [], [], [], [], []
		for r in sel:
			A = np.array(r['A'], dtype=np.int64)
			b = np.array(r['b'], dtype=np.int64)
			x_gt = np.array(r['x'], dtype=np.int64)
			pt = np.array(r['marginals'])
			forced = (pt < EPS) | (pt > 1 - EPS)
			if forced.sum() == 0:
				continue
			fv = (pt > 0.5).astype(np.int64)

			lp = WarmLP(A, b, env)
			if lp.relax() is None:
				continue
			nf.append(A.shape[1])

			# network: one (warm) feature LP + one forward pass
			t0 = time.perf_counter()
			x_lp = lp.relax()
			with torch.no_grad():
				p = torch.sigmoid(model(*features(A, b, x_lp, device))).numpy()
			tn.append(time.perf_counter() - t0)
			an.append(((p >= 0.5).astype(np.int64)[forced] == x_gt[forced]).mean())

			# probing: one (warm) LP per free variable
			t0 = time.perf_counter()
			det = np.array([lp.fix_infeasible(j, 1 - fv[j]) for j in range(A.shape[1])])
			tp.append(time.perf_counter() - t0)
			ap_.append((det & forced).sum() / forced.sum())
			lp.mo.dispose()

		row = dict(n_free=float(np.mean(nf)), net_ms=1000 * float(np.mean(tn)),
		            net_acc=float(np.mean(an)), probe_ms=1000 * float(np.mean(tp)),
		            probe_recall=float(np.mean(ap_)))
		row['ratio'] = row['probe_ms'] / max(1e-9, row['net_ms'])
		table[d] = row
		print(f"{d:>6}{row['n_free']:>8.1f}{row['net_ms']:>12.2f}{100*row['net_acc']:>9.1f}%"
		      f"{row['probe_ms']:>11.2f}{100*row['probe_recall']:>13.1f}%{row['ratio']:>7.1f}x")
	json.dump(table, open(a.out, 'w'), indent=1)
	print(f"\nsaved: {a.out}")


if __name__ == '__main__':
	main()
