"""
Pure CP-SAT baseline for the crossover experiment, with honest tri-state
reporting (feasible / infeasible / unresolved-on-timeout) instead of the
stock baseline_dns/CPSAT_ORtools_dns.py convention of guessing "feasible"
when the solver times out. Guessing would bias CP-SAT's own accuracy
upward at exactly the large sizes where the crossover question is decided,
so it is not used here.
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from ortools.sat.python import cp_model


def load_data(data_dir):
	files = sorted(Path(data_dir).glob("*.json"))
	data = []
	for f in files:
		with open(f) as fp:
			d = json.load(fp)
		A = np.array(d["A"], dtype=np.int32)
		b = np.array(d["b"], dtype=np.int32)
		gt_feasible = bool(d.get("feasible", True))
		data.append((f.name, A, b, gt_feasible))
	return data


def solve(A, b, time_limit_sec):
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
	status = solver.Solve(model)
	if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		sol = np.array([solver.Value(x_vars[j]) for j in range(n)], dtype=np.int8)
		return 'feasible', sol
	if status == cp_model.INFEASIBLE:
		return 'infeasible', None
	return 'unresolved', None


def run(data_dir, time_limit, out_json):
	data = load_data(data_dir)
	total = len(data)
	if total == 0:
		raise SystemExit(f"No JSON files found in {data_dir}")

	tp = tn = fp = fn = sol_correct = unresolved = 0
	gt_f_total = gt_i_total = 0
	times = []
	per_instance = []

	for name, A, b, gt_feasible in data:
		t0 = time.perf_counter()
		status, sol = solve(A, b, time_limit)
		elapsed = time.perf_counter() - t0
		times.append(elapsed)

		gt_f_total += int(gt_feasible)
		gt_i_total += int(not gt_feasible)

		pred_feasible = None if status == 'unresolved' else (status == 'feasible')
		if pred_feasible is None:
			unresolved += 1
		elif gt_feasible and pred_feasible:
			tp += 1
		elif not gt_feasible and not pred_feasible:
			tn += 1
		elif not gt_feasible and pred_feasible:
			fp += 1
		else:
			fn += 1

		sol_ok = False
		if gt_feasible and sol is not None and np.all(A.dot(sol.astype(np.int64)) == b):
			sol_correct += 1
			sol_ok = True

		per_instance.append({
			'instance': name, 'gt_feasible': gt_feasible, 'status': status,
			'pred_feasible': pred_feasible, 'sol_correct': sol_ok, 'time_sec': round(elapsed, 4),
		})

	resolved = total - unresolved
	gt_f, gt_i = gt_f_total, gt_i_total
	summary = {
		'data_dir': data_dir, 'time_limit': time_limit, 'total': total,
		'gt_feasible': gt_f, 'gt_infeasible': gt_i, 'unresolved': unresolved,
		'resolved': resolved,
		'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
		'feasibility_acc_resolved': (tp + tn) / resolved * 100 if resolved else 0.0,
		'sol_correct': sol_correct, 'sol_acc': sol_correct / gt_f * 100 if gt_f else 0.0,
		'avg_time': float(np.mean(times)), 'std_time': float(np.std(times)),
		'total_time': float(np.sum(times)),
	}

	w = 62
	print("\n" + "=" * w)
	print(f"Result Summary: CP-SAT pure (honest unresolved), cap={time_limit}s")
	print(f"  dir: {data_dir}")
	print("=" * w)
	print(f"  Total Instances   : {total:>6}")
	print(f"  GT Feasible       : {gt_f:>6}  ({100*gt_f/total:.1f}%)")
	print(f"  GT Infeasible     : {gt_i:>6}  ({100*gt_i/total:.1f}%)")
	print(f"  Unresolved        : {unresolved:>6}  ({100*unresolved/total:.1f}%)")
	print("-" * w)
	print(f"  Feasibility Acc (resolved) : {tp+tn} / {resolved}  ({summary['feasibility_acc_resolved']:.2f}%)")
	print(f"      TP {tp}  TN {tn}  FP {fp}  FN {fn}")
	print(f"  Solution Acc (GT feasible) : {sol_correct} / {gt_f}  ({summary['sol_acc']:.2f}%)")
	print("-" * w)
	print(f"  Avg Time / Inst   : {summary['avg_time']:.4f} sec")
	print(f"  Total Time        : {summary['total_time']:.4f} sec")
	print("=" * w)

	if out_json:
		with open(out_json, 'w') as f:
			json.dump({'summary': summary, 'per_instance': per_instance}, f, indent=2)
		print(f"Saved: {out_json}")


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--time_limit', type=float, default=30.0)
	ap.add_argument('--out_json', type=str, default=None)
	a = ap.parse_args()
	run(a.data_dir, a.time_limit, a.out_json)
