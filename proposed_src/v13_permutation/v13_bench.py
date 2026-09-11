"""v13 evaluation on the shared 24:6 benchmark sets, reported in this project's
standard five metrics (Feasible MCC / Solution ACC / Unresolved Rate / Avg / STD time).

Runs the v13 symbolic pipeline twice per instance set under an identical AHL try
budget -- once with uniform-random permutations (the v9 behavior, control) and once
with the learned permutation policy -- so any difference is attributable to the policy
alone rather than to budget or pipeline changes. A hard per-instance wall-clock cap is
enforced by subprocess, matching the convention used for the CP-SAT/SCIP baselines
(timeout => 'unresolved', never a guess).
"""
# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import math
import time
from pathlib import Path
from multiprocessing import Process
import numpy as np


def _worker(mode, size, inst_path, out_path, ckpt, tries, block, dfs_budget):
	import torch
	import LPneuroBLS_v7 as m
	from v13_pipeline import ahl_solve_policy, ahl_try_with_perm
	from v13_train_perm import PermPolicy, instance_features

	with open(inst_path) as f:
		d = json.load(f)
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	gt_feasible = bool(d.get('feasible', True))

	device = torch.device('cpu')
	rng = np.random.default_rng(abs(hash(Path(inst_path).name)) % (2**31))
	t0 = time.time()

	env = m.FastBinaryEnv(A, b)
	env.propagate_constraints().apply_probing()
	sol, pred, method = None, None, None
	if env.is_invalid:
		pred, method = False, 'propagation_infeasible'
	else:
		x_lp, lp_ok = m.solve_lp_relaxation(A, b)
		if not lp_ok:
			pred, method = False, 'lp_hard_rule'
		else:
			A_pinv = np.linalg.pinv(A.astype(np.float64))
			x_pump, pumped, _ = m.kernel_pump_stats(x_lp, A, b, A_pinv, max_iter=300)
			if pumped:
				sol, pred, method = np.asarray(x_pump), True, 'kernel_pump'
			else:
				if mode == 'policy':
					model = PermPolicy()
					state = torch.load(ckpt, map_location=device, weights_only=False)
					model.load_state_dict(state['model_state_dict'])
					model.eval()
					xv, xc, ev2c, ec2v = instance_features(A, b, x_lp.astype(np.float64), device)
					with torch.no_grad():
						scores = model(xv, xc, ev2c, ec2v).cpu().numpy()
					s, _, _ = ahl_solve_policy(A, b, scores, rng, block, tries)
				else:
					s = None
					for t in range(tries):
						p = np.arange(A.shape[1]) if t == 0 else rng.permutation(A.shape[1])
						s = ahl_try_with_perm(A, b, p, block)
						if s is not None:
							break
				if s is not None:
					sol, pred, method = s, True, 'ahl_bkz'
				elif dfs_budget > 0:
					order = np.arange(A.shape[1])
					val_order = np.zeros(A.shape[1], dtype=np.int64)
					ds, status, _ = m.policy_dfs(A, b, env.assignment.copy(), order, val_order,
					                              node_budget=dfs_budget)
					if status == 'solved':
						sol, pred, method = ds, True, 'dfs_solved'
					elif status == 'infeasible':
						pred, method = False, 'dfs_proved_infeasible'
					else:
						pred, method = None, 'unresolved'
				else:
					pred, method = None, 'unresolved'

	elapsed = time.time() - t0
	sol_correct = bool(gt_feasible and sol is not None and
	                    np.all(A.dot(np.asarray(sol, dtype=np.int64)) == b))
	with open(out_path, 'w') as f:
		json.dump({'instance': Path(inst_path).name, 'gt_feasible': gt_feasible,
		            'pred_feasible': pred, 'sol_correct': sol_correct,
		            'method': method, 'time_sec': round(elapsed, 4)}, f)


def run_set(mode, size, data_dir, out_dir, ckpt, tries, block, dfs_budget, cap, jobs):
	files = sorted(Path(data_dir).glob('*.json'))
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	pending, running, done = list(files), [], 0
	t0 = time.time()
	while pending or running:
		while pending and len(running) < jobs:
			fp = pending.pop(0)
			op = out_dir / (fp.stem + '_result.json')
			p = Process(target=_worker, args=(mode, size, str(fp), str(op), ckpt, tries, block, dfs_budget))
			p.start()
			running.append(dict(p=p, fp=fp, op=op, start=time.time()))
		time.sleep(1)
		still = []
		for r in running:
			if r['p'].is_alive():
				if time.time() - r['start'] > cap:
					r['p'].terminate(); r['p'].join(timeout=5)
					if r['p'].is_alive():
						r['p'].kill(); r['p'].join()
					gt = bool(json.load(open(r['fp'])).get('feasible', True))
					with open(r['op'], 'w') as f:
						json.dump({'instance': r['fp'].name, 'gt_feasible': gt,
						            'pred_feasible': None, 'sol_correct': False,
						            'method': 'timeout_killed',
						            'time_sec': round(time.time() - r['start'], 4)}, f)
					done += 1
				else:
					still.append(r)
			else:
				r['p'].join()
				done += 1
		running = still
	print(f"  [{mode} {size}] {done}/{len(files)} done in {time.time()-t0:.0f}s", flush=True)


def metrics(out_dir):
	files = sorted(Path(out_dir).glob('*_result.json'))
	tp = tn = fp = fn = 0
	gt_f = gt_i = sol_ok = unres_f = 0
	times = []
	for f in files:
		r = json.load(open(f))
		gt, pred = r['gt_feasible'], r['pred_feasible']
		times.append(r['time_sec'])
		gt_f += int(gt); gt_i += int(not gt)
		if pred is None:
			unres_f += int(gt)
		elif gt and pred: tp += 1
		elif not gt and not pred: tn += 1
		elif not gt and pred: fp += 1
		else: fn += 1
		sol_ok += int(r['sol_correct'])
	den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
	mcc = (tp*tn - fp*fn)/den if den > 0 else None
	return dict(n=len(files), gt_f=gt_f, gt_i=gt_i, mcc=mcc, sol_ok=sol_ok,
	            unres_f=unres_f, avg=float(np.mean(times)), std=float(np.std(times)))


def fmt(label, mm):
	mcc = f"{mm['mcc']*100:.1f}%" if mm['mcc'] is not None else "0.0% (N/A)"
	return (f"{label:<22} FeasibleMCC={mcc:>14}  "
	        f"SolutionACC={100*mm['sol_ok']/max(1,mm['gt_f']):5.1f}%({mm['sol_ok']}/{mm['gt_f']})  "
	        f"UnresolvedRate={100*mm['unres_f']/max(1,mm['gt_f']):5.1f}%({mm['unres_f']}/{mm['gt_f']})  "
	        f"AvgTime={mm['avg']:8.3f}s  StdTime={mm['std']:8.3f}s")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sizes', nargs='+', required=True)
	ap.add_argument('--ckpt', required=True)
	ap.add_argument('--tries', type=int, default=40)
	ap.add_argument('--dfs_budget', type=int, default=0)
	ap.add_argument('--cap', type=float, default=900.0)
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--out_root', default='../../runs/v13_bench')
	ap.add_argument('--modes', nargs='+', default=['random', 'policy'])
	a = ap.parse_args()

	from v13_train_perm import SIZE_BLOCK
	results = {}
	for sz in a.sizes:
		block = SIZE_BLOCK[sz]
		for mode in a.modes:
			od = f"{a.out_root}/{mode}_{sz}"
			print(f"=== {mode} {sz} (block={block}, tries={a.tries}) ===", flush=True)
			run_set(mode, sz, f"../../instances/bench30_{sz}", od, a.ckpt,
			        a.tries, block, a.dfs_budget, a.cap, a.jobs)
			results[(mode, sz)] = metrics(od)

	print("\n" + "=" * 110)
	print("v13 results (identical AHL try budget; only the permutation source differs)")
	print("=" * 110)
	for sz in a.sizes:
		for mode in a.modes:
			if (mode, sz) in results:
				print(fmt(f"{sz} [{mode}]", results[(mode, sz)]))
		print("-" * 110)


if __name__ == '__main__':
	main()
