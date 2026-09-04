"""Sweep CP-SAT solver config x instance to build a labeled dataset for a per-instance config-selection policy.
Only sizes where CP-SAT doesn't already solve everything instantly (10x25/20x50 are degenerate, skipped)."""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from ortools.sat.python import cp_model
from difficulty_index import instance_metrics

CONFIGS = {
	'default': {},
	'linearization0': {'linearization_level': 0},
	'linearization2': {'linearization_level': 2},
	'portfolio_search': {'search_branching': cp_model.PORTFOLIO_SEARCH},
	'lp_search': {'search_branching': cp_model.LP_SEARCH},
	'no_symmetry': {'symmetry_level': 0},
}
POOLS = {
	'40x100': ('../instances/sub_hard_40x100_40', 40),
	'60x150': ('../instances/test_instances_hard_60x150_40', 40),
	'80x200': ('../instances/test_instances_hard_80x200_30', 30),
}
TIME_LIMIT = 20.0


def features(A, b, seed):
	m, n = A.shape
	rng = np.random.default_rng(seed)
	vs, bt, lp_ok = instance_metrics(A, b, rng)
	density = float(A.mean())
	return dict(m=m, n=n, density=density, vertex_spread=vs, b_tightness=bt, lp_feasible=lp_ok)


def solve_one_config(A, b, overrides):
	m, n = A.shape
	model = cp_model.CpModel()
	x_vars = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(x_vars[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.log_search_progress = False
	solver.parameters.max_time_in_seconds = TIME_LIMIT
	for k, v in overrides.items():
		setattr(solver.parameters, k, v)
	t0 = time.time()
	status = solver.Solve(model)
	dt = time.time() - t0
	if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
		sol = np.array([solver.Value(x_vars[j]) for j in range(n)], dtype=np.int64)
		ok = bool(np.all(A.dot(sol) == b))
		return dict(status='feasible', sol_ok=ok, time=dt)
	if status == cp_model.INFEASIBLE:
		return dict(status='infeasible', sol_ok=None, time=dt)
	return dict(status='unresolved', sol_ok=None, time=dt)


def sweep_one(task):
	idx, path, size, seed = task
	with open(path) as f:
		d = json.load(f)
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	gt_feasible = bool(d.get('feasible', True))
	feat = features(A, b, seed)
	results = {}
	for name, overrides in CONFIGS.items():
		r = solve_one_config(A, b, overrides)
		# solved = correctly resolved w.r.t. ground truth (sound solver: a 'feasible'/'infeasible'
		# verdict is always correct when reached, so success only depends on matching gt_feasible)
		if gt_feasible:
			r['solved'] = (r['status'] == 'feasible' and r['sol_ok'])
		else:
			r['solved'] = (r['status'] == 'infeasible')
		results[name] = r
	return idx, dict(path=str(path), size=size, gt_feasible=gt_feasible, feat=feat, results=results)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--jobs', type=int, default=16)
	a = ap.parse_args()

	tasks = []
	idx_ctr, seed = 0, 0
	for size, (dir_path, count) in POOLS.items():
		files = sorted(Path(dir_path).glob('*.json'))[:count]
		print(f"{size}: {len(files)} instances queued", flush=True)
		for fp in files:
			tasks.append((idx_ctr, fp, size, seed))
			idx_ctr += 1
			seed += 1

	t0 = time.time()
	out = [None] * len(tasks)
	with Pool(a.jobs) as pool:
		done = 0
		for idx, rec in pool.imap_unordered(sweep_one, tasks, chunksize=1):
			out[idx] = rec
			done += 1
			print(f"  {done}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)

	with open(a.out, 'w') as f:
		json.dump(out, f)
	print(f"done: {len(tasks)} instances, {time.time()-t0:.0f}s total -> {a.out}")


if __name__ == '__main__':
	main()
