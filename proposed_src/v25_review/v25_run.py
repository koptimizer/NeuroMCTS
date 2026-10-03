"""v25 runner: one instance set x several arms x one seed offset, lattice stage off, each instance
in its own process under a wall-clock budget. Writes runs/v25/<tag>_<arm>/inst-XXXX.json and
runs/v25/summary_<tag>.json (per arm: solved, median nodes, median time, refuted share).

Example:
  python3 v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S0 --time_limit 600 \
      --arms lp model pv lpvar_s1val s1var_lpval pvvar_lpval maxsd domwdeg impact lp_probe --seed_offset 0
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse, json, os, time, zlib
from pathlib import Path
from multiprocessing import Process
import numpy as np
import torch
import gurobipy as gp

torch.set_num_threads(1)


def _worker(arm, inst_path, out_path, ckpt_s1, ckpt_pv, time_limit, seed_off, enum_limit, fast=False):
	from v25_search import GUIDES, NEEDS, search
	from v15_train_marginal import MarginalNet
	from v24_rl_core import load_pv
	device = torch.device('cpu')
	model = None
	if fast and NEEDS.get(arm) in ('s1', 'pv') and torch.load(ckpt_s1, map_location='cpu', weights_only=False).get('arch') is None:      # A1: dense TorchScript inference path, identical weights
		from v25_dense import load_dense
		model = load_dense(ckpt_s1, None, script=True) if NEEDS[arm] == 's1' else load_dense(*ckpt_pv.split(','), script=True)
		model.dense = True
	elif NEEDS.get(arm) == 's1':
		ck = torch.load(ckpt_s1, map_location=device, weights_only=False)
		if ck.get('arch') == 'nsnet':                      # v26 E1: NSNet-style marginal network, same interface
			from v26_nsnet import NSNetMarginal
			model = NSNetMarginal()
		else:
			model = MarginalNet()
		model.load_state_dict(ck['model_state_dict']); model.eval()
	elif NEEDS.get(arm) == 'pv':
		p1, p2 = ckpt_pv.split(',')
		ck = torch.load(p2, map_location=device, weights_only=False)
		if ck.get('mode', 'posterior') != 'posterior':          # P3 variant (lp / random policy logit): evaluate with its own formula
			from v25_r2_variants import build
			model = build(p1, ck['mode']); model.load_state_dict(ck['model_state_dict']); model.eval()
		else:
			model = load_pv(p1, p2)
	d = json.load(open(inst_path))
	A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
	seed = (zlib.crc32(Path(inst_path).name.encode()) + seed_off * 1_000_003) % (2 ** 31)
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
	r = search(A, b, GUIDES[arm], model, time_limit, seed, env, ctx=dict(device=device, enum_limit=enum_limit))
	r.update(instance=Path(inst_path).name, arm=arm, seed_offset=seed_off)
	json.dump(r, open(out_path, 'w'))


def run_arm(arm, data_dir, out_dir, ckpt_s1, ckpt_pv, time_limit, jobs, seed_off, enum_limit, fast=False):
	os.makedirs(out_dir, exist_ok=True)
	files = sorted(Path(data_dir).glob('*.json')); todo = [f for f in files if not (Path(out_dir) / (f.stem + '.json')).exists()]
	running = []; t0 = time.time()
	while todo or running:
		while todo and len(running) < jobs:
			f = todo.pop(0); op = Path(out_dir) / (f.stem + '.json')
			p = Process(target=_worker, args=(arm, str(f), str(op), ckpt_s1, ckpt_pv, time_limit, seed_off, enum_limit, fast)); p.start()
			running.append((p, f, op, time.time()))
		time.sleep(0.5)
		still = []
		for p, f, op, ts in running:
			if p.is_alive():
				if time.time() - ts > time_limit + 60:
					p.terminate(); p.join()
					json.dump(dict(instance=f.name, arm=arm, seed_offset=seed_off, solved=False, nodes=-1, sec=time_limit, path_len=0, dead_first=[], n_refuted=0, refuted_nodes=0, killed=True), open(op, 'w'))
				else:
					still.append((p, f, op, ts))
			else:
				if not op.exists():
					json.dump(dict(instance=f.name, arm=arm, seed_offset=seed_off, solved=False, nodes=-1, sec=time_limit, path_len=0, dead_first=[], n_refuted=0, refuted_nodes=0, crashed=True), open(op, 'w'))
		running = still
	rs = [json.load(open(Path(out_dir) / (f.stem + '.json'))) for f in files]
	sol = [r for r in rs if r['solved']]
	summ = dict(arm=arm, n=len(rs), solved=len(sol), median_nodes=float(np.median([r['nodes'] for r in sol])) if sol else None,
	            median_sec=float(np.median([r['sec'] for r in sol])) if sol else None,
	            refuted_share=float(np.median([r['refuted_nodes'] / max(r['nodes'], 1) for r in sol])) if sol else None,
	            wall=round(time.time() - t0, 1))
	print(f"  [{arm}] solved {summ['solved']}/{summ['n']}  median nodes {summ['median_nodes']}  median sec {summ['median_sec']}  refuted share {summ['refuted_share']}  ({summ['wall']}s)", flush=True)
	return summ


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--tag', required=True)
	ap.add_argument('--time_limit', type=float, required=True)
	ap.add_argument('--arms', nargs='+', required=True)
	ap.add_argument('--ckpt_s1', default='../../runs/v22/model_n25/conditional.pt')
	ap.add_argument('--ckpt_pv', default='../../runs/v22/model_n25/conditional.pt,../../runs/v24/r2/pv_best.pt')
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--seed_offset', type=int, default=0)
	ap.add_argument('--enum_limit', type=float, default=60.0)
	ap.add_argument('--out_root', default='../../runs/v25')
	ap.add_argument('--fast', action='store_true', help='dense TorchScript inference for the network arms (A1)')
	a = ap.parse_args()
	os.makedirs(a.out_root, exist_ok=True)
	sp = f"{a.out_root}/summary_{a.tag}.json"
	res = json.load(open(sp)) if os.path.exists(sp) else {}
	for arm in a.arms:
		key = arm + ('_fast' if a.fast else '')
		print(f"=== {a.tag} {key} (budget {a.time_limit}s, seed {a.seed_offset}) ===", flush=True)
		res[key] = run_arm(arm, a.data_dir, f"{a.out_root}/{a.tag}_{key}", a.ckpt_s1, a.ckpt_pv, a.time_limit, a.jobs, a.seed_offset, a.enum_limit, a.fast)
		json.dump(res, open(sp, 'w'), indent=1)


if __name__ == '__main__':
	main()
