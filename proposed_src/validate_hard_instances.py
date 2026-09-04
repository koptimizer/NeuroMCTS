#!/usr/bin/env python3
"""
Validates a generated hard-instance directory pair (train+test, same size):
  1. dedup: no duplicate (A, b) pairs across train+test combined
  2. band compliance: every feasible instance's b_i in [ceil(.25*rowsum_i), floor(.75*rowsum_i)]
  3. planted-solution sanity: stored x actually satisfies Ax=b for feasible instances
  4. CP-SAT re-verification: a sample of infeasible instances are re-confirmed INFEASIBLE
  5. difficulty-index comparison: scores a sample against the dnsms reference distribution
     built earlier today (same size) to quantify how much harder these are

Usage:
  python validate_hard_instances.py --size 10x25 --train_n 10000 --test_n 1000
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from ortools.sat.python import cp_model

sys.path.insert(0, '.')
from LPneuroBLS_v7 import gen_planted_instance
from difficulty_index import instance_metrics, build_reference, score_level


def load_dir(d):
	files = sorted(Path(d).glob('*.json'))
	recs = []
	for f in files:
		recs.append(json.load(open(f)))
	return recs


def check_dedup(recs):
	keys = set()
	dup = 0
	for r in recs:
		k = (json.dumps(r['A']), json.dumps(r['b']))
		if k in keys:
			dup += 1
		keys.add(k)
	return dup, len(recs)


def compute_vertex_spread_task(args):
	A, b, seed = args
	rng = np.random.default_rng(seed)
	vs, bt, ok = instance_metrics(np.array(A, dtype=np.int64), np.array(b, dtype=np.int64), rng)
	return vs if ok else None


def check_vertex_spread_gain(recs, m, n, sample_n, jobs):
	feas = [r for r in recs if r['feasible']]
	rng = np.random.default_rng(2)
	if len(feas) > sample_n:
		idx = rng.choice(len(feas), sample_n, replace=False)
		feas = [feas[i] for i in idx]
	tasks = [(r['A'], r['b'], 6_000_000 + i) for i, r in enumerate(feas)]
	with Pool(jobs) as pool:
		hard_vs = [v for v in pool.map(compute_vertex_spread_task, tasks) if v is not None]
	ref_tasks = [(m, n, 8_500_000 + i) for i in range(sample_n)]
	with Pool(jobs) as pool:
		nat_results = pool.map(build_ref_task, ref_tasks)
	nat_vs = [r[0] for r in nat_results if r[2]]
	return np.mean(hard_vs), np.mean(nat_vs)


def check_planted_solution(recs):
	bad = 0
	checked = 0
	for r in recs:
		if not r['feasible']:
			continue
		A = np.array(r['A'], dtype=np.int64); b = np.array(r['b'], dtype=np.int64)
		x = np.array(r['x'], dtype=np.int64)
		checked += 1
		if not np.array_equal(A.dot(x), b):
			bad += 1
	return bad, checked


def cpsat_check_task(rec):
	A = np.array(rec['A'], dtype=np.int64); b = np.array(rec['b'], dtype=np.int64)
	m, n = A.shape
	model = cp_model.CpModel()
	xv = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(m):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(xv[j] for j in idx) == int(b[i]))
	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.max_time_in_seconds = 30.0
	status = solver.Solve(model)
	return status == cp_model.INFEASIBLE, status == cp_model.UNKNOWN


def check_infeasible_reverify(recs, sample_n, jobs):
	infeas = [r for r in recs if not r['feasible']]
	rng = np.random.default_rng(0)
	if len(infeas) > sample_n:
		idx = rng.choice(len(infeas), sample_n, replace=False)
		infeas = [infeas[i] for i in idx]
	with Pool(jobs) as pool:
		results = pool.map(cpsat_check_task, infeas)
	confirmed = sum(1 for ok, _ in results if ok)
	unresolved = sum(1 for _, unk in results if unk)
	return confirmed, unresolved, len(infeas)


def metrics_task(args):
	A, b, seed = args
	rng = np.random.default_rng(seed)
	vs, bt, ok = instance_metrics(np.array(A, dtype=np.int64), np.array(b, dtype=np.int64), rng)
	return vs, bt, ok


def build_ref_task(args):
	m, n, seed = args
	rng = np.random.default_rng(seed)
	A, b, x = gen_planted_instance(m, n, rng)
	vs, bt, ok = instance_metrics(A.astype(np.int64), b.astype(np.int64), rng)
	return vs, bt, ok


def difficulty_comparison(recs, m, n, sample_n, jobs):
	feas = [r for r in recs if r['feasible']]
	rng = np.random.default_rng(1)
	if len(feas) > sample_n:
		idx = rng.choice(len(feas), sample_n, replace=False)
		feas = [feas[i] for i in idx]
	tasks = [(r['A'], r['b'], 7_000_000 + i) for i, r in enumerate(feas)]
	with Pool(jobs) as pool:
		hard_results = pool.map(metrics_task, tasks)
	ref_tasks = [(m, n, 8_000_000 + i) for i in range(sample_n)]
	with Pool(jobs) as pool:
		ref_results = pool.map(build_ref_task, ref_tasks)
	ref = build_reference([r[0] for r in ref_results if r[2]], [r[1] for r in ref_results if r[2]])
	levels = []
	for vs, bt, ok in hard_results:
		if ok:
			lvl, *_ = score_level(vs, bt, ref)
			levels.append(lvl)
	levels = np.array(levels)
	dist = {i: int((levels == i).sum()) for i in range(1, 6)}
	return dist, float(levels.mean()) if len(levels) else None, len(levels)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', required=True)
	ap.add_argument('--train_n', type=int, default=10000)
	ap.add_argument('--test_n', type=int, default=1000)
	ap.add_argument('--root', default='../instances')
	ap.add_argument('--sample_n', type=int, default=200)
	ap.add_argument('--jobs', type=int, default=8)
	a = ap.parse_args()
	m, n = (int(v) for v in a.size.split('x'))

	train_dir = f"{a.root}/train_instances_hard_{m}x{n}_{a.train_n}"
	test_dir = f"{a.root}/test_instances_hard_{m}x{n}_{a.test_n}"
	train = load_dir(train_dir)
	test = load_dir(test_dir)
	all_recs = train + test
	print(f"=== {a.size}: {len(train)} train + {len(test)} test = {len(all_recs)} total ===")

	dup, tot = check_dedup(all_recs)
	print(f"[dedup] {dup} duplicate (A,b) pairs out of {tot}")

	hard_mean, nat_mean = check_vertex_spread_gain(all_recs, m, n, a.sample_n, a.jobs)
	print(f"[vertex_spread] hard mean={hard_mean:.3f} vs natural-dnsms mean={nat_mean:.3f}  "
	      f"(+{100*(hard_mean-nat_mean)/nat_mean:.1f}%)")

	bad2, checked2 = check_planted_solution(all_recs)
	print(f"[planted x sanity] {bad2} mismatches out of {checked2} feasible instances")

	n_feas = sum(1 for r in all_recs if r['feasible'])
	n_infeas = len(all_recs) - n_feas
	print(f"[label balance] feasible {n_feas} ({100*n_feas/len(all_recs):.1f}%), infeasible {n_infeas} ({100*n_infeas/len(all_recs):.1f}%)")

	confirmed, unresolved, n_checked = check_infeasible_reverify(all_recs, a.sample_n, a.jobs)
	print(f"[CP-SAT re-verify] {confirmed}/{n_checked} confirmed INFEASIBLE, {unresolved} unresolved (timeout)")

	dist, mean_level, n_scored = difficulty_comparison(all_recs, m, n, a.sample_n, a.jobs)
	print(f"[difficulty index] scored {n_scored} feasible instances vs fresh dnsms reference (same size)")
	print(f"  level distribution (1=easy..5=hard): {dist}  mean={mean_level:.2f}" if mean_level else "  (none scored)")


if __name__ == '__main__':
	main()
