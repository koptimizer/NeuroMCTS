#!/usr/bin/env python3
"""
CP-SAT parameter sweep (algorithm configuration): tests whether solver-level
settings, chosen without touching the model's correctness, change solve rate
or speed at 60x150 -- the one size where plain CP-SAT already struggles
(cpsat_hint_experiment.py: only 2/12 feasible solved in 30s). A later step
could replace the fixed --config here with a learned per-instance selector;
this script first checks whether any single config meaningfully beats the
default, which is a prerequisite for that to be worth learning at all.

Usage:
  python cpsat_param_sweep.py --data_dir ../../instances/sub60x150_18 \
      --time_limit 30 --config linearization2
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from ortools.sat.python import cp_model


CONFIGS = {
	'default': {},
	'linearization0': {'linearization_level': 0},
	'linearization2': {'linearization_level': 2},
	'portfolio_search': {'search_branching': cp_model.PORTFOLIO_SEARCH},
	'lp_search': {'search_branching': cp_model.LP_SEARCH},
	'no_symmetry': {'symmetry_level': 0},
}


def load_data(data_dir):
	files = sorted(Path(data_dir).glob("*.json"))
	data = []
	for f in files:
		d = json.load(open(f))
		A = np.array(d["A"], dtype=np.int32)
		b = np.array(d["b"], dtype=np.int32)
		gt_feasible = bool(d.get("feasible", True))
		data.append((A, b, gt_feasible))
	return data


def solve(A, b, time_limit_sec, param_overrides):
	m, n = A.shape
	model = cp_model.CpModel()
	x_vars = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(x_vars[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.log_search_progress = False
	solver.parameters.max_time_in_seconds = time_limit_sec
	for k, v in param_overrides.items():
		setattr(solver.parameters, k, v)
	status = solver.Solve(model)
	if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		sol = np.array([solver.Value(x_vars[j]) for j in range(n)], dtype=np.int8)
		return True, sol, status
	elif status == cp_model.INFEASIBLE:
		return False, None, status
	return True, None, status


def status_name(status):
	return {cp_model.OPTIMAL: 'OPTIMAL', cp_model.FEASIBLE: 'FEASIBLE',
	        cp_model.INFEASIBLE: 'INFEASIBLE', cp_model.UNKNOWN: 'UNKNOWN'}.get(status, str(status))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--time_limit', type=float, default=30.0)
	ap.add_argument('--config', required=True, choices=list(CONFIGS.keys()))
	a = ap.parse_args()
	data = load_data(a.data_dir)
	overrides = CONFIGS[a.config]
	print(f"config={a.config} overrides={overrides}  {len(data)} instances")

	tp = tn = fp = fn = sol_correct = 0
	times = []
	for idx, (A, b, gt_feasible) in enumerate(data):
		t0 = time.perf_counter()
		pred_feasible, pred_sol, status = solve(A, b, a.time_limit, overrides)
		dt = time.perf_counter() - t0
		times.append(dt)
		if gt_feasible and pred_feasible: tp += 1
		elif not gt_feasible and not pred_feasible: tn += 1
		elif not gt_feasible and pred_feasible: fp += 1
		else: fn += 1
		if gt_feasible and pred_sol is not None and np.all(A.dot(pred_sol.astype(np.int32)) == b):
			sol_correct += 1
		print(f"    [{idx+1}/{len(data)}] gt_feas={gt_feasible} pred_feas={pred_feasible} "
		      f"status={status_name(status)} t={dt:.2f}s", flush=True)

	gt_f = tp + fn; gt_i = tn + fp
	acc = (tp + tn) / len(data) * 100
	sol_acc = sol_correct / gt_f * 100 if gt_f else 0.0
	print(f"\n=== config={a.config} ===")
	print(f"  feasibility acc {acc:.2f}%  (TP {tp}/{gt_f} TN {tn}/{gt_i} FP {fp}/{gt_i} FN {fn}/{gt_f})")
	print(f"  sol acc {sol_acc:.2f}%  ({sol_correct}/{gt_f})")
	print(f"  avg time {np.mean(times):.3f}s  median {np.median(times):.3f}s  "
	      f"max {np.max(times):.3f}s  total {np.sum(times):.1f}s")


if __name__ == '__main__':
	main()
