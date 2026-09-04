#!/usr/bin/env python3
"""
Tests whether CP-SAT itself can be strengthened rather than replaced, at the
one size (60x150) where our own pipeline collapses (6.67% solved) and where
classical baselines have never been run at all. Two conditions on the same
instances: plain CP-SAT (the baseline_dns/CPSAT_ORtools_dns.py recipe), and
CP-SAT warm-started via AddHint() with the final rounded point from our
kernel pump (proposed_src/collect_paired_signal.py's kernel_pump_once),
regardless of whether the pump itself converged -- a "near-miss" point is
still a legitimate hint. Since CP-SAT remains the exact solver in both
conditions, correctness is unaffected either way; only wall-clock time can
change.

Usage:
  python cpsat_hint_experiment.py --data_dir ../instances/sub60x150_18 --time_limit 60
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from ortools.sat.python import cp_model

import sys
sys.path.insert(0, '.')
from LPneuroBLS_v7 import solve_lp_relaxation


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


def pump_hint(A, b, max_iter=200):
	# same alternating rounding + null-space projection as kernel_pump_once,
	# but returns the final rounded point regardless of convergence
	x_lp, _ = solve_lp_relaxation(A, b)
	A_pinv = np.linalg.pinv(A.astype(np.float64))
	Af = A.astype(np.float64); bf = b.astype(np.float64)
	x = np.clip(x_lp.astype(np.float64), 0.0, 1.0)
	prev = None
	for _ in range(max_iter):
		x_r = np.round(np.clip(x, 0.0, 1.0))
		if np.all(np.abs(Af.dot(x_r) - bf) < 1e-6):
			return x_r.astype(np.int32)
		if prev is not None and np.array_equal(x_r, prev):
			break
		prev = x_r.copy()
		x = x_r - A_pinv.dot(Af.dot(x_r) - bf)
	return np.clip(np.round(x), 0, 1).astype(np.int32)


def solve(A, b, time_limit_sec, hint=None):
	m, n = A.shape
	model = cp_model.CpModel()
	x_vars = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(x_vars[j] for j in idx) == int(b[i]))
	if hint is not None:
		for j, v in enumerate(hint):
			model.AddHint(x_vars[j], int(v))
	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.log_search_progress = False
	solver.parameters.max_time_in_seconds = time_limit_sec
	status = solver.Solve(model)
	if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		sol = np.array([solver.Value(x_vars[j]) for j in range(n)], dtype=np.int8)
		return True, sol, status
	elif status == cp_model.INFEASIBLE:
		return False, None, status
	return True, None, status


def run_condition(data, time_limit, use_hint):
	tp = tn = fp = fn = sol_correct = 0
	times = []
	for idx, (A, b, gt_feasible) in enumerate(data):
		hint = pump_hint(A, b) if use_hint else None
		t0 = time.perf_counter()
		pred_feasible, pred_sol, status = solve(A, b, time_limit, hint)
		dt = time.perf_counter() - t0
		times.append(dt)
		if gt_feasible and pred_feasible: tp += 1
		elif not gt_feasible and not pred_feasible: tn += 1
		elif not gt_feasible and pred_feasible: fp += 1
		else: fn += 1
		if gt_feasible and pred_sol is not None and np.all(A.dot(pred_sol.astype(np.int32)) == b):
			sol_correct += 1
		print(f"    [{idx+1}/{len(data)}] gt_feas={gt_feasible} pred_feas={pred_feasible} "
		      f"status={solver_status_name(status)} t={dt:.2f}s", flush=True)
	gt_f = tp + fn; gt_i = tn + fp
	return dict(tp=tp, tn=tn, fp=fp, fn=fn, sol_correct=sol_correct, gt_f=gt_f, gt_i=gt_i,
	            times=times, total=len(data))


def solver_status_name(status):
	return {cp_model.OPTIMAL: 'OPTIMAL', cp_model.FEASIBLE: 'FEASIBLE',
	        cp_model.INFEASIBLE: 'INFEASIBLE', cp_model.UNKNOWN: 'UNKNOWN'}.get(status, str(status))


def print_summary(label, r):
	acc = (r['tp'] + r['tn']) / r['total'] * 100
	sol_acc = r['sol_correct'] / r['gt_f'] * 100 if r['gt_f'] else 0.0
	print(f"\n=== {label} ===")
	print(f"  feasibility acc {acc:.2f}%  (TP {r['tp']}/{r['gt_f']} TN {r['tn']}/{r['gt_i']} "
	      f"FP {r['fp']}/{r['gt_i']} FN {r['fn']}/{r['gt_f']})")
	print(f"  sol acc {sol_acc:.2f}%  ({r['sol_correct']}/{r['gt_f']})")
	print(f"  avg time {np.mean(r['times']):.3f}s  median {np.median(r['times']):.3f}s  "
	      f"max {np.max(r['times']):.3f}s  total {np.sum(r['times']):.1f}s")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--time_limit', type=float, default=60.0)
	a = ap.parse_args()
	data = load_data(a.data_dir)
	print(f"loaded {len(data)} instances from {a.data_dir}")

	print("\n--- condition A: plain CP-SAT ---")
	rA = run_condition(data, a.time_limit, use_hint=False)
	print("\n--- condition B: CP-SAT + pump hint ---")
	rB = run_condition(data, a.time_limit, use_hint=True)

	print_summary("A: plain CP-SAT", rA)
	print_summary("B: CP-SAT + pump hint", rB)


if __name__ == '__main__':
	main()
