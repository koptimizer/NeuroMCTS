"""v15: measure the Bayes limit on per-variable prediction for this instance family.

Every learning attempt in this project (11 of them) has tried to predict something
about x* from (A, b). None beat ~73% per-variable accuracy. This script asks whether
that is a modeling failure or an information-theoretic ceiling, by computing the exact
posterior instead of estimating it.

Because the generator plants one x* and publishes b = A x*, any x in the solution set
S = {x : Ax = b} would have produced the same b. The posterior over "which x was
planted" is therefore uniform on S, and the true marginal is
    p_j = P(x_j = 1 | A, b) = |{x in S : x_j = 1}| / |S|.
A Bayes-optimal predictor answers argmax and is right max(p_j, 1-p_j) of the time, so
    bayes_acc = mean_j max(p_j, 1 - p_j)
is the ceiling NO model can exceed on this distribution.

S is enumerated exactly with CP-SAT (capped; capped instances are reported separately
and excluded from the ceiling statistics, since a truncated S biases the marginals).

Readout:
  GNN accuracy ~= bayes_acc  -> the ceiling is the barrier; stop trying to learn this
  GNN accuracy <<  bayes_acc  -> signal exists and our models are failing to extract it
"""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from ortools.sat.python import cp_model


class SolutionCollector(cp_model.CpSolverSolutionCallback):
	"""Collects every 0/1 solution CP-SAT enumerates, up to a cap.
	Stopping at the cap is recorded so truncated instances can be excluded.
	Storing full vectors is fine at these sizes (n <= 35)."""

	def __init__(self, xv, cap):
		super().__init__()
		self.xv = xv
		self.cap = cap
		self.sols = []
		self.hit_cap = False

	def on_solution_callback(self):
		self.sols.append([self.Value(v) for v in self.xv])
		if len(self.sols) >= self.cap:
			self.hit_cap = True
			self.StopSearch()


def enumerate_solutions(A, b, cap, time_limit):
	mm, n = A.shape
	model = cp_model.CpModel()
	xv = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(mm):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(xv[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.enumerate_all_solutions = True
	solver.parameters.num_search_workers = 1
	solver.parameters.max_time_in_seconds = time_limit
	col = SolutionCollector(xv, cap)
	status = solver.Solve(model, col)
	# Only OPTIMAL means the enumeration ran to exhaustion (INFEASIBLE means there was
	# nothing to enumerate). FEASIBLE here means the search was cut off by the time limit
	# with solutions already collected -- accepting it would silently return a truncated
	# S and therefore biased marginals, which is exactly what the labels must not contain.
	complete = (not col.hit_cap) and status in (cp_model.OPTIMAL, cp_model.INFEASIBLE)
	return np.array(col.sols, dtype=np.int64), complete


def probe_one(task):
	idx, path, cap, time_limit = task
	d = json.load(open(path))
	if not d.get('feasible', True) or d.get('x') is None:
		return idx, None
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	x_gt = np.array(d['x'], dtype=np.int64)
	t0 = time.time()
	S, complete = enumerate_solutions(A, b, cap, time_limit)
	if len(S) == 0:
		return idx, None
	p = S.mean(axis=0)                                   # true marginals
	bayes_acc = float(np.maximum(p, 1 - p).mean())       # ceiling on per-variable accuracy
	bayes_pred = (p >= 0.5).astype(np.int64)
	acc_vs_planted = float((bayes_pred == x_gt).mean())  # ceiling measured against the planted x
	# how much of the ceiling survives when restricted to the most confident variables
	conf_order = np.argsort(-np.abs(p - 0.5))
	top = {k: float(np.maximum(p, 1 - p)[conf_order[:k]].mean()) for k in (3, 5, 8, 10) if k <= len(p)}
	return idx, dict(instance=Path(path).name, n=int(A.shape[1]), n_sols=int(len(S)),
	                  complete=bool(complete), bayes_acc=bayes_acc,
	                  acc_vs_planted=acc_vs_planted, top_conf=top,
	                  marginals=p.tolist(), time=time.time() - t0)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--limit', type=int, default=24)
	ap.add_argument('--cap', type=int, default=200000)
	ap.add_argument('--time_limit', type=float, default=120.0)
	ap.add_argument('--jobs', type=int, default=8)
	ap.add_argument('--out_json', required=True)
	a = ap.parse_args()

	files = sorted(Path(a.data_dir).glob('*.json'))[:a.limit]
	tasks = [(i, str(f), a.cap, a.time_limit) for i, f in enumerate(files)]
	print(f"enumerating solution sets for {len(tasks)} instances (cap={a.cap})", flush=True)

	out = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		for idx, rec in pool.imap_unordered(probe_one, tasks, chunksize=1):
			if rec is None:
				continue
			out.append(rec)
			tag = '' if rec['complete'] else '  [CAPPED - excluded]'
			print(f"  [{len(out)}] {rec['instance']} |S|={rec['n_sols']} "
			      f"bayes_acc={100*rec['bayes_acc']:.1f}%{tag}  ({time.time()-t0:.0f}s)", flush=True)

	full = [r for r in out if r['complete']]
	print(f"\n=== Bayes ceiling ({len(full)} instances with complete enumeration, "
	      f"{len(out)-len(full)} capped and excluded) ===")
	if full:
		ns = np.array([r['n_sols'] for r in full])
		ba = np.array([r['bayes_acc'] for r in full])
		ap_ = np.array([r['acc_vs_planted'] for r in full])
		print(f"  |S| (solution count) : median={np.median(ns):.0f}  min={ns.min()}  max={ns.max()}")
		print(f"  Bayes per-variable accuracy ceiling : {100*ba.mean():.1f}%  (min {100*ba.min():.1f}%, max {100*ba.max():.1f}%)")
		print(f"  Bayes prediction vs the planted x   : {100*ap_.mean():.1f}%")
		for k in (3, 5, 8, 10):
			vals = [r['top_conf'][str(k)] if str(k) in r['top_conf'] else r['top_conf'].get(k)
			         for r in full if (str(k) in r['top_conf'] or k in r['top_conf'])]
			vals = [v for v in vals if v is not None]
			if vals:
				print(f"  ceiling on the {k:>2} most confident variables : {100*np.mean(vals):.1f}%")
	json.dump(out, open(a.out_json, 'w'))
	print(f"saved: {a.out_json}")


if __name__ == '__main__':
	main()
