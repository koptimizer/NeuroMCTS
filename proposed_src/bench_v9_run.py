"""Benchmark runner for the pure v9 pipeline (AHL/BKZ + GNN-MCTS + policy DFS, no CP-SAT)
under a hard 900s per-instance wall-clock cap, with per-size parameter budgets chosen from
prior block-optimization findings (lattice-enum-progress memory + this session's AHL sweep)
now that a much larger time budget is available than earlier (mostly 20-30s cap) runs.

A hard OS-level per-instance timeout is required (not signal.alarm) because BKZ.reduction
is a blocking fpylll/C call that a Python-level alarm cannot reliably interrupt -- confirmed
earlier this session. Each instance runs in its own subprocess; if it does not finish within
the cap, the process is killed and the instance is recorded as 'unresolved', matching the
tri-state convention used for the CP-SAT/SCIP baselines.
"""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Process

CHECKPOINTS = {
	'10x25': '../runs/260806-2249_LP_Hybrid_v7/checkpoint_hard_finetuned_10x25.pt',
	'20x50': '../runs/260806-2258_LP_Hybrid_v7/checkpoint_hard_finetuned_20x50.pt',
	'40x100': '../runs/260710-2026_LP_Hybrid_v7/checkpoint_pretrained.pt',
	'60x150': '../runs/260710-2026_LP_Hybrid_v7/checkpoint_pretrained.pt',
}

# Per-size parameter budgets. AHL block chosen from the accuracy-vs-block curve
# (lattice-enum-progress: 20x50->block10 @78.25%, 40x100->block20 @66.67%) where
# available; 60x150 has no such curve (AHL near-useless there per this session's
# sweep) so uses the fastest-safe block from timing calibration and leans on DFS.
# ahl_tries and dfs_budget (base; scaled by n/25 inside infer_instance) are sized
# so the estimated worst case comfortably fits under the 900s cap -- verified by
# calibration before the full run, see --calibrate.
PARAMS = {
	'10x25':  dict(ahl_block=8,  ahl_tries=15, num_simulations=100, mp_scale=8.0, pump_iters=200, dfs_budget=20000),
	'20x50':  dict(ahl_block=10, ahl_tries=50, num_simulations=150, mp_scale=8.0, pump_iters=200, dfs_budget=700000),
	'40x100': dict(ahl_block=20, ahl_tries=40, num_simulations=150, mp_scale=8.0, pump_iters=300, dfs_budget=1500000),
	'60x150': dict(ahl_block=15, ahl_tries=40, num_simulations=150, mp_scale=8.0, pump_iters=300, dfs_budget=70000),
}


def _worker(size, inst_path, out_path, checkpoint=None):
	import numpy as np
	import torch
	import LPneuroBLS_v7 as m
	from LPneuroBLS_v9_infer_pure import prepare_lp_tensors, matrix_to_graph, infer_instance

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = m.BipartiteGNN(hidden_dim=256, num_layers=8).to(device)
	ckpt = torch.load(checkpoint or CHECKPOINTS[size], map_location=device, weights_only=False)
	sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
	model.load_state_dict(sd)
	model.eval()
	p = PARAMS[size]
	mcts = m.MCTS(model, device, num_simulations=p['num_simulations'], c_puct=1.5, training=False)

	with open(inst_path) as f:
		d = json.load(f)
	A = np.array(d['A'], dtype=np.int32)
	b = np.array(d['b'], dtype=np.int32)
	gt_feasible = bool(d.get('feasible', True))

	t0 = time.time()
	x_lp_var, x_lp_check, x_lp, lp_ok = prepare_lp_tensors(A, b, device)
	A_pinv = np.linalg.pinv(A.astype(np.float64))
	ev2c, ec2v = matrix_to_graph(A, device)
	mp_iters = min(48, int(np.ceil(p['mp_scale'] * A.shape[1] / 25.0)))
	mcts.mp_iters = mp_iters

	assignment, solved, pred_feasible, feas_prob, traj, is_invalid, method = infer_instance(
		model, mcts, A, b, x_lp_var, x_lp_check, x_lp, A_pinv, device,
		ev2c=ev2c, ec2v=ec2v, deterministic=True, feas_threshold=0.5,
		lp_is_feasible=lp_ok, pump_iters=p['pump_iters'], dfs_budget=p['dfs_budget'],
		mp_iters=mp_iters, ahl_block=p['ahl_block'], ahl_tries=p['ahl_tries'],
	)
	elapsed = time.time() - t0

	sol_correct = False
	if gt_feasible and solved:
		arr = np.maximum(np.array(assignment, dtype=np.int64), 0)
		sol_correct = bool(np.all(A.dot(arr) == b))

	result = {
		'instance': Path(inst_path).name, 'gt_feasible': gt_feasible,
		'pred_feasible': (None if pred_feasible is None else bool(pred_feasible)),
		'status': ('unresolved' if pred_feasible is None else ('feasible' if pred_feasible else 'infeasible')),
		'sol_correct': sol_correct, 'method': method, 'time_sec': round(elapsed, 4),
	}
	with open(out_path, 'w') as f:
		json.dump(result, f)


def run_one(size, inst_path, out_path, cap, checkpoint=None):
	proc = Process(target=_worker, args=(size, str(inst_path), str(out_path), checkpoint))
	t0 = time.time()
	proc.start()
	proc.join(timeout=cap)
	if proc.is_alive():
		proc.terminate()
		proc.join(timeout=10)
		if proc.is_alive():
			proc.kill()
			proc.join()
		elapsed = time.time() - t0
		with open(inst_path) as f:
			gt_feasible = bool(json.load(f).get('feasible', True))
		result = {
			'instance': Path(inst_path).name, 'gt_feasible': gt_feasible,
			'pred_feasible': None, 'status': 'unresolved',
			'sol_correct': False, 'method': 'timeout_killed', 'time_sec': round(elapsed, 4),
		}
		with open(out_path, 'w') as f:
			json.dump(result, f)
		return result, True
	with open(out_path) as f:
		result = json.load(f)
	return result, False


def _gt_feasible(inst_path):
	with open(inst_path) as f:
		return bool(json.load(f).get('feasible', True))


def run_parallel(size, files, out_dir, cap, jobs, checkpoint=None):
	pending = list(files)
	running = []  # list of dict(proc, fp, out_path, start)
	done_count = 0
	t0 = time.time()
	while pending or running:
		while pending and len(running) < jobs:
			fp = pending.pop(0)
			out_path = out_dir / (fp.stem + '_result.json')
			proc = Process(target=_worker, args=(size, str(fp), str(out_path), checkpoint))
			proc.start()
			running.append(dict(proc=proc, fp=fp, out_path=out_path, start=time.time()))
		time.sleep(2)
		still_running = []
		for r in running:
			if r['proc'].is_alive():
				if time.time() - r['start'] > cap:
					r['proc'].terminate()
					r['proc'].join(timeout=10)
					if r['proc'].is_alive():
						r['proc'].kill()
						r['proc'].join()
					elapsed = time.time() - r['start']
					result = {
						'instance': r['fp'].name, 'gt_feasible': _gt_feasible(r['fp']),
						'pred_feasible': None, 'status': 'unresolved',
						'sol_correct': False, 'method': 'timeout_killed', 'time_sec': round(elapsed, 4),
					}
					with open(r['out_path'], 'w') as f:
						json.dump(result, f)
					done_count += 1
					print(f"  [{done_count}/{len(files)}] ({time.time()-t0:.0f}s) {r['fp'].name}: "
					      f"unresolved [KILLED] ({elapsed:.1f}s)", flush=True)
				else:
					still_running.append(r)
			else:
				r['proc'].join()
				with open(r['out_path']) as f:
					result = json.load(f)
				done_count += 1
				print(f"  [{done_count}/{len(files)}] ({time.time()-t0:.0f}s) {r['fp'].name}: "
				      f"{result['status']} ({result['time_sec']:.1f}s)", flush=True)
		running = still_running
	print(f"[v9 {size}] done: {len(files)} instances, {time.time()-t0:.0f}s total", flush=True)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', required=True, choices=list(PARAMS.keys()))
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--cap', type=float, default=900.0)
	ap.add_argument('--out_dir', required=True)
	ap.add_argument('--limit', type=int, default=None)
	ap.add_argument('--jobs', type=int, default=1)
	ap.add_argument('--checkpoint', type=str, default=None,
	                 help='Override the per-size default checkpoint (e.g. a multisize fine-tune)')
	a = ap.parse_args()

	out_dir = Path(a.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	files = sorted(Path(a.data_dir).glob('*.json'))
	if a.limit:
		files = files[:a.limit]

	ckpt_used = a.checkpoint or CHECKPOINTS[a.size]
	print(f"[v9 {a.size}] {len(files)} instances, cap={a.cap}s, jobs={a.jobs}, checkpoint={ckpt_used}, params={PARAMS[a.size]}", flush=True)

	if a.jobs <= 1:
		t0 = time.time()
		for i, fp in enumerate(files):
			out_path = out_dir / (fp.stem + '_result.json')
			result, killed = run_one(a.size, fp, out_path, a.cap, a.checkpoint)
			tag = ' [KILLED]' if killed else ''
			print(f"  [{i+1}/{len(files)}] ({time.time()-t0:.0f}s) {fp.name}: "
			      f"{result['status']}{tag} ({result['time_sec']:.1f}s)", flush=True)
		print(f"[v9 {a.size}] done: {len(files)} instances, {time.time()-t0:.0f}s total", flush=True)
	else:
		run_parallel(a.size, files, out_dir, a.cap, a.jobs, a.checkpoint)


if __name__ == '__main__':
	main()
