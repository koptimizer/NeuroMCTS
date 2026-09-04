#!/usr/bin/env python3
"""
Builds per-size reference distributions (vertex_spread, b_tightness_mean)
from freshly generated instances, then scores existing instance directories
(instances/*) into the 5-level difficulty index (difficulty_index.py).
"""
import json
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, '.')
from LPneuroBLS_v7 import gen_planted_instance
from difficulty_index import instance_metrics, b_tightness_mean, build_reference, score_level


def gen_ref_task(args):
	m, n, seed = args
	rng = np.random.default_rng(seed)
	A, b, x = gen_planted_instance(m, n, rng)
	vs, bt, ok = instance_metrics(A.astype(np.int64), b.astype(np.int64), rng)
	return vs, bt, ok


def build_ref(m, n, count, jobs=8, seed0=5_000_000):
	tasks = [(m, n, seed0 + i) for i in range(count)]
	with Pool(jobs) as pool:
		results = pool.map(gen_ref_task, tasks)
	vs_list = [r[0] for r in results if r[2]]
	bt_list = [r[1] for r in results if r[2]]
	return build_reference(vs_list, bt_list)


def eval_file_task(args):
	path, seed = args
	d = json.load(open(path))
	A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
	gt_feasible = bool(d.get('feasible', True))
	rng = np.random.default_rng(seed)
	if not gt_feasible:
		return dict(path=str(path), gt_feasible=False)
	vs, bt, ok = instance_metrics(A, b, rng)
	if not ok:
		return dict(path=str(path), gt_feasible=True, lp_feasible=False)
	return dict(path=str(path), gt_feasible=True, lp_feasible=True, vs=vs, bt=bt)


def evaluate_dir(dirpath, ref, m, n, sample_n=300, jobs=8, seed0=9_000_000):
	files = sorted(Path(dirpath).glob('*.json'))
	rng0 = np.random.default_rng(0)
	if len(files) > sample_n:
		idx = rng0.choice(len(files), sample_n, replace=False)
		files = [files[i] for i in sorted(idx)]
	tasks = [(f, seed0 + i) for i, f in enumerate(files)]
	with Pool(jobs) as pool:
		results = pool.map(eval_file_task, tasks)
	n_total = len(results)
	n_infeasible = sum(1 for r in results if not r['gt_feasible'])
	n_lp_infeasible = sum(1 for r in results if r['gt_feasible'] and not r.get('lp_feasible', True))
	scored = [r for r in results if r.get('lp_feasible')]
	levels = []
	for r in scored:
		lvl, score, pv, pb = score_level(r['vs'], r['bt'], ref)
		levels.append(lvl)
	levels = np.array(levels)
	dist = {i: int((levels == i).sum()) for i in range(1, 6)}
	return dict(dirpath=str(dirpath), n_total=n_total, n_infeasible=n_infeasible,
	            n_lp_infeasible=n_lp_infeasible, n_scored=len(scored),
	            level_dist=dist, mean_level=float(levels.mean()) if len(levels) else None)


if __name__ == '__main__':
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument('--ref_count', type=int, default=200)
	ap.add_argument('--sample_n', type=int, default=300)
	ap.add_argument('--jobs', type=int, default=8)
	a = ap.parse_args()

	sizes = [(10, 25), (20, 50), (40, 100), (60, 150)]
	refs = {}
	for m, n in sizes:
		print(f'building reference for {m}x{n} (n={a.ref_count})...', flush=True)
		refs[(m, n)] = build_ref(m, n, a.ref_count, jobs=a.jobs)
		print(f'  vs range [{min(refs[(m,n)]["vs_sorted"]):.3f}, {max(refs[(m,n)]["vs_sorted"]):.3f}]  '
		      f'bt range [{min(refs[(m,n)]["bt_sorted"]):.4f}, {max(refs[(m,n)]["bt_sorted"]):.4f}]', flush=True)

	dirs = [
		('../instances/test_instances_dns_10x25_1000', 10, 25, 'legacy dns 10x25'),
		('../instances/test_instances_dnsms_10x25_1000', 10, 25, 'dnsms 10x25'),
		('../instances/test_instances_dnsms_20x50_1000', 20, 50, 'dnsms 20x50'),
		('../instances/test_instances_dnsms_40x100_1000', 40, 100, 'dnsms 40x100'),
		('../instances/test_instances_dnsms_60x150_40', 60, 150, 'dnsms 60x150'),
	]
	print()
	for dirpath, m, n, label in dirs:
		res = evaluate_dir(dirpath, refs[(m, n)], m, n, sample_n=a.sample_n, jobs=a.jobs)
		print(f'=== {label} ===')
		print(f'  {res["n_total"]} sampled | infeasible: {res["n_infeasible"]} | '
		      f'LP-infeasible (of feasible): {res["n_lp_infeasible"]} | scored: {res["n_scored"]}')
		print(f'  level distribution (1=easiest..5=hardest): {res["level_dist"]}  mean={res["mean_level"]:.2f}' if res['mean_level'] else '  (none scored)')
		print(flush=True)
