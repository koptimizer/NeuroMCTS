"""Parallel-across-instances runner for the honest tri-state exact-solver baselines
(Gurobi MILP / SCIP MILP / CP-SAT), needed because a 900s cap x 30 instances run
sequentially could take up to 7.5 hours per (solver, size) combo in the worst case."""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np


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


def solve_one(task):
	idx, solver, name, A, b, gt_feasible, time_limit = task
	if solver == 'gurobi':
		import crossover_gurobi_pure as m
		import gurobipy as gp
		env = gp.Env(empty=True)
		env.setParam("OutputFlag", 0)
		env.start()
		t0 = time.perf_counter()
		status, sol = m.solve(A, b, env, time_limit)
		elapsed = time.perf_counter() - t0
	elif solver == 'scip':
		import crossover_scip_pure as m
		t0 = time.perf_counter()
		status, sol = m.solve(A, b, time_limit)
		elapsed = time.perf_counter() - t0
	elif solver == 'cpsat':
		import crossover_cpsat_pure as m
		t0 = time.perf_counter()
		status, sol = m.solve(A, b, time_limit)
		elapsed = time.perf_counter() - t0
	else:
		raise ValueError(solver)

	pred_feasible = None if status == 'unresolved' else (status == 'feasible')
	sol_ok = False
	if gt_feasible and sol is not None and np.all(A.dot(np.asarray(sol, dtype=np.int64)) == b):
		sol_ok = True
	return idx, {
		'instance': name, 'gt_feasible': gt_feasible, 'status': status,
		'pred_feasible': pred_feasible, 'sol_correct': sol_ok, 'time_sec': round(elapsed, 4),
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--solver', required=True, choices=['gurobi', 'scip', 'cpsat'])
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--time_limit', type=float, default=900.0)
	ap.add_argument('--jobs', type=int, default=8)
	ap.add_argument('--out_json', required=True)
	a = ap.parse_args()

	data = load_data(a.data_dir)
	tasks = [(i, a.solver, name, A, b, gt_f, a.time_limit) for i, (name, A, b, gt_f) in enumerate(data)]
	print(f"[{a.solver}] {len(tasks)} instances, cap={a.time_limit}s, jobs={a.jobs}  |  {a.data_dir}", flush=True)

	t0 = time.time()
	per_instance = [None] * len(tasks)
	with Pool(a.jobs) as pool:
		done = 0
		for idx, rec in pool.imap_unordered(solve_one, tasks, chunksize=1):
			per_instance[idx] = rec
			done += 1
			print(f"  [{a.solver}] {done}/{len(tasks)}  ({time.time()-t0:.0f}s)  "
			      f"{rec['instance']}: {rec['status']} ({rec['time_sec']:.1f}s)", flush=True)

	tp = tn = fp = fn = sol_correct = unresolved = 0
	gt_f_total = gt_i_total = 0
	times = []
	for r in per_instance:
		gt_f_total += int(r['gt_feasible'])
		gt_i_total += int(not r['gt_feasible'])
		times.append(r['time_sec'])
		if r['pred_feasible'] is None:
			unresolved += 1
		elif r['gt_feasible'] and r['pred_feasible']:
			tp += 1
		elif not r['gt_feasible'] and not r['pred_feasible']:
			tn += 1
		elif not r['gt_feasible'] and r['pred_feasible']:
			fp += 1
		else:
			fn += 1
		if r['sol_correct']:
			sol_correct += 1

	resolved = len(per_instance) - unresolved
	summary = {
		'solver': a.solver, 'data_dir': a.data_dir, 'time_limit': a.time_limit, 'total': len(per_instance),
		'gt_feasible': gt_f_total, 'gt_infeasible': gt_i_total, 'unresolved': unresolved, 'resolved': resolved,
		'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
		'feasibility_acc_resolved': (tp + tn) / resolved * 100 if resolved else 0.0,
		'infeasible_detect_rate': tn / gt_i_total * 100 if gt_i_total else 0.0,
		'sol_correct': sol_correct, 'sol_acc': sol_correct / gt_f_total * 100 if gt_f_total else 0.0,
		'avg_time': float(np.mean(times)), 'total_time_wall': time.time() - t0,
		'total_time_cpu': float(np.sum(times)),
	}
	print(f"\n[{a.solver}] done: {summary}", flush=True)
	with open(a.out_json, 'w') as f:
		json.dump({'summary': summary, 'per_instance': per_instance}, f, indent=2)
	print(f"Saved: {a.out_json}")


if __name__ == '__main__':
	main()
