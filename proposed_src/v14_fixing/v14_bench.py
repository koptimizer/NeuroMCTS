"""v14 arm comparison on the shared 24:6 benchmark sets, in the project's standard
five metrics (Feasible MCC / Solution ACC / Unresolved Rate / Avg / STD time).

Arms, all given the SAME total AHL-try budget (T*t) so any difference is
attributable to the fixing strategy rather than to compute:
  a0_plain  : no fixing, whole budget on the full instance          (= v9/v13 behavior)
  a1_lp     : fix top-k by LP confidence, retry                     (classical baseline)
  a2_gnn    : fix top-k by the pretrained assign head, retry        (bar the RL must beat)
  a3_policy : fix with the v14 RL policy, retry                     (proposed)
  a4_oracle : fix top-k using the planted solution                  (ceiling, not an opponent)

Because fixing shrinks n, an AHL try in a fixing arm is cheaper than one in a0.
Equal-try parity therefore favors the fixing arms, so wall-clock time is reported
alongside and should be read as the conservative comparison.
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


def _worker(arm, inst_path, out_path, k, T, tries, block, ckpt, policy_ckpt):
	import torch
	import LPneuroBLS_v7 as m
	from v14_core import (solve_with_fixing, solve_plain, lp_confidence_fixing,
	                       gnn_marginals, gnn_fixing)

	d = json.load(open(inst_path))
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	gt_feasible = bool(d.get('feasible', True))
	x_gt = np.array(d['x'], dtype=np.int64) if d.get('x') is not None else None
	device = torch.device('cpu')
	rng = np.random.default_rng(abs(hash(Path(inst_path).name)) % (2 ** 31))
	t0 = time.time()

	sol, pred, method, stats = None, None, None, {}
	env = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
	env.propagate_constraints().apply_probing()
	if env.is_invalid:
		pred, method = False, 'propagation_infeasible'
	else:
		x_lp, lp_ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
		if not lp_ok:
			pred, method = False, 'lp_hard_rule'
		else:
			x_lp = x_lp.astype(np.float64)
			A_pinv = np.linalg.pinv(A.astype(np.float64))
			x_pump, pumped, _ = m.kernel_pump_stats(x_lp, A.astype(np.int32),
			                                         b.astype(np.int32), A_pinv, max_iter=300)
			if pumped:
				sol, pred, method = np.asarray(x_pump, dtype=np.int64), True, 'kernel_pump'
			else:
				if arm == 'a0_plain':
					sol, stats = solve_plain(A, b, T * tries, block, rng)
				else:
					if arm == 'a1_lp':
						def propose(attempt, r):
							return lp_confidence_fixing(A, b, k, r, x_lp, stochastic=(attempt > 0))
					elif arm == 'a2_gnn':
						model = m.BipartiteGNN(hidden_dim=256, num_layers=8).to(device)
						st = torch.load(ckpt, map_location=device, weights_only=False)
						model.load_state_dict(st['model_state_dict'] if 'model_state_dict' in st else st)
						model.eval()
						p1 = gnn_marginals(model, A, b, x_lp, device)
						def propose(attempt, r):
							return gnn_fixing(p1, k, r, stochastic=(attempt > 0))
					elif arm == 'a1b_lp_top':
						from v14_core import topT_joint_fixings
						sel_t, combos_t = topT_joint_fixings(np.clip(x_lp, 0, 1), k, T)
						def propose(attempt, r):
							return sel_t, combos_t[min(attempt, len(combos_t) - 1)]
					elif arm == 'a2b_gnn_top':
						from v14_core import topT_joint_fixings
						model = m.BipartiteGNN(hidden_dim=256, num_layers=8).to(device)
						st = torch.load(ckpt, map_location=device, weights_only=False)
						model.load_state_dict(st['model_state_dict'] if 'model_state_dict' in st else st)
						model.eval()
						p1 = gnn_marginals(model, A, b, x_lp, device)
						sel_t, combos_t = topT_joint_fixings(p1, k, T)
						def propose(attempt, r):
							return sel_t, combos_t[min(attempt, len(combos_t) - 1)]
					elif arm == 'a3_policy':
						from v14_train_fix import FixPolicy, policy_propose
						pol = FixPolicy()
						st = torch.load(policy_ckpt, map_location=device, weights_only=False)
						pol.load_state_dict(st['model_state_dict'])
						pol.eval()
						def propose(attempt, r):
							return policy_propose(pol, A, b, k, r, device, greedy=(attempt == 0))
					elif arm == 'a4_oracle':
						def propose(attempt, r):
							# infeasible instances carry no planted solution; fix nothing
							if x_gt is None:
								return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
							conf = -np.abs(x_lp - np.round(x_lp))
							sel = np.argsort(-conf)[:k]
							return sel, x_gt[sel]
					sol, stats = solve_with_fixing(A, b, propose, k, T, tries, block, rng)
				if sol is not None:
					pred, method = True, 'ahl_fixed' if arm != 'a0_plain' else 'ahl_bkz'
				else:
					pred, method = None, 'unresolved'

	elapsed = time.time() - t0
	sol_correct = bool(gt_feasible and sol is not None and
	                    np.all(A.dot(np.asarray(sol, dtype=np.int64)) == b))
	with open(out_path, 'w') as f:
		json.dump({'instance': Path(inst_path).name, 'gt_feasible': gt_feasible,
		            'pred_feasible': pred, 'sol_correct': sol_correct, 'method': method,
		            'time_sec': round(elapsed, 4), 'stats': stats}, f)


def run_arm(arm, data_dir, out_dir, k, T, tries, block, ckpt, policy_ckpt, cap, jobs):
	files = sorted(Path(data_dir).glob('*.json'))
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	pending, running, done = list(files), [], 0
	t0 = time.time()
	while pending or running:
		while pending and len(running) < jobs:
			fp = pending.pop(0)
			op = out_dir / (fp.stem + '_result.json')
			p = Process(target=_worker, args=(arm, str(fp), str(op), k, T, tries,
			                                   block, ckpt, policy_ckpt))
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
						            'time_sec': round(time.time() - r['start'], 4),
						            'stats': {}}, f)
					done += 1
				else:
					still.append(r)
			else:
				r['p'].join(); done += 1
		running = still
	print(f"  [{arm}] {done}/{len(files)} in {time.time()-t0:.0f}s", flush=True)


def metrics(out_dir):
	files = sorted(Path(out_dir).glob('*_result.json'))
	tp = tn = fp = fn = 0
	gt_f = gt_i = sol_ok = unres_f = 0
	times = []
	solved_set = set()
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
		if r['sol_correct']:
			sol_ok += 1
			solved_set.add(r['instance'])
	den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
	return dict(n=len(files), gt_f=gt_f, gt_i=gt_i,
	             mcc=((tp * tn - fp * fn) / den if den > 0 else None),
	             sol_ok=sol_ok, unres_f=unres_f, solved=solved_set,
	             avg=float(np.mean(times)), std=float(np.std(times)))


def fmt(label, mm):
	mcc = f"{mm['mcc']*100:.1f}%" if mm['mcc'] is not None else "0.0% (N/A)"
	return (f"{label:<20} MCC={mcc:>13}  "
	         f"SolACC={100*mm['sol_ok']/max(1,mm['gt_f']):5.1f}%({mm['sol_ok']}/{mm['gt_f']})  "
	         f"Unres={100*mm['unres_f']/max(1,mm['gt_f']):5.1f}%  "
	         f"Avg={mm['avg']:7.3f}s  Std={mm['std']:7.3f}s")


def mcnemar(a, b):
	# paired exact test on which instances each arm solved
	only_a = len(a['solved'] - b['solved'])
	only_b = len(b['solved'] - a['solved'])
	n = only_a + only_b
	if n == 0:
		return only_a, only_b, 1.0
	p = 2 * sum(math.comb(n, i) for i in range(max(only_a, only_b), n + 1)) / 2 ** n
	return only_a, only_b, min(p, 1.0)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', required=True)
	ap.add_argument('--block', type=int, required=True)
	ap.add_argument('--k', type=int, required=True)
	ap.add_argument('--T', type=int, default=10)
	ap.add_argument('--tries', type=int, default=4)
	ap.add_argument('--arms', nargs='+',
	                 default=['a0_plain', 'a1_lp', 'a2_gnn', 'a4_oracle'])
	ap.add_argument('--ckpt', default='../../runs/best.pth')
	ap.add_argument('--policy_ckpt', default=None)
	ap.add_argument('--cap', type=float, default=900.0)
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--out_root', default='../../runs/v14_bench')
	a = ap.parse_args()

	res = {}
	for arm in a.arms:
		od = f"{a.out_root}/{a.size}_{arm}"
		print(f"=== {arm} {a.size} (k={a.k}, T={a.T}, tries={a.tries}, budget={a.T*a.tries}) ===", flush=True)
		run_arm(arm, f"../../instances/bench30_{a.size}", od, a.k, a.T, a.tries,
		        a.block, a.ckpt, a.policy_ckpt, a.cap, a.jobs)
		res[arm] = metrics(od)

	print("\n" + "=" * 104)
	print(f"v14 {a.size}  (k={a.k}, T={a.T} attempts x {a.tries} tries = {a.T*a.tries} AHL tries per arm)")
	print("=" * 104)
	for arm in a.arms:
		print(fmt(arm, res[arm]))
	if 'a0_plain' in res:
		print("-" * 104)
		for arm in a.arms:
			if arm == 'a0_plain':
				continue
			oa, ob, p = mcnemar(res[arm], res['a0_plain'])
			print(f"  {arm} vs a0_plain : {arm}만 {oa}개, a0만 {ob}개, McNemar p={p:.3f}")
	if 'a2_gnn' in res and 'a3_policy' in res:
		oa, ob, p = mcnemar(res['a3_policy'], res['a2_gnn'])
		print(f"  a3_policy vs a2_gnn : policy만 {oa}개, gnn만 {ob}개, McNemar p={p:.3f}")


if __name__ == '__main__':
	main()
