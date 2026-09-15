"""Cascade evaluation (guidance arms x {lattice off, lattice on}) with the warm-started search.

Identical protocol to v22_cascade_search.py -- same instances, budgets, seeding, hard
per-instance timeout, summary and McNemar test -- with one substitution: the search loop
is v23_warm_search.search, which keeps a single relaxation per instance and re-optimizes
it from the parent basis instead of rebuilding a SCIP model at every node. Every arm goes
through the same loop, so the change affects all of them equally; it exists so that the
per-node cost the paper attributes to model rebuilding is removed from the deployed
system rather than merely measured in isolation.

Adds a `random` arm for the cascade question: whether the residual left by the lattice
stage is easy enough that unguided branching closes it too.
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
import zlib
from pathlib import Path
from multiprocessing import Process
import numpy as np
import torch

import LPneuroBLS_v7 as m
import gurobipy as gp
from v23_warm_search import GUIDES, search

torch.set_num_threads(1)


def ahl_stage(A, b, block, tries, seed):
	# returns (solution|None, seconds spent)
	from v13_pipeline import ahl_try_with_perm
	rng = np.random.default_rng(seed)
	n = A.shape[1]
	t0 = time.time()
	for t in range(tries):
		perm = np.arange(n) if t == 0 else rng.permutation(n)
		sol = ahl_try_with_perm(A, b, perm, block)
		if sol is not None:
			return sol, time.time() - t0
	return None, time.time() - t0


def _worker(arm, ahl_on, inst_path, out_path, ckpt, time_limit, ahl_share, block, tries, seed_off=0):
	from v15_train_marginal import MarginalNet
	device = torch.device('cpu')
	model = None
	if arm == 'model':
		model = MarginalNet()
		st = torch.load(ckpt, map_location=device, weights_only=False)
		model.load_state_dict(st['model_state_dict'])
		model.eval()

	d = json.load(open(inst_path))
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	gt = bool(d.get('feasible', True))
	# zlib.crc32 rather than hash(): Python randomizes string hashing per process
	# (PYTHONHASHSEED), so hash() would silently reseed every run and make results
	# irreproducible. seed_off selects an independent repetition of the same experiment.
	seed = (zlib.crc32(Path(inst_path).name.encode()) + seed_off * 1_000_003) % (2 ** 31)

	solved_by, ahl_sec = None, 0.0
	if ahl_on:
		sol, ahl_sec = ahl_stage(A, b, block, tries, seed)
		if sol is not None:
			solved_by = 'ahl'
	budget = time_limit - ahl_sec if ahl_on else time_limit
	if solved_by is None and budget > 0:
		genv = gp.Env(empty=True)
		genv.setParam('OutputFlag', 0)
		genv.start()
		ok, nodes, el = search(A, b, GUIDES[arm], model, device, budget, seed, genv)
		genv.dispose()
		if ok:
			solved_by = 'search'
	else:
		nodes, el = 0, 0.0

	json.dump(dict(instance=Path(inst_path).name, gt_feasible=gt,
	                solved=bool(solved_by is not None), solved_by=solved_by,
	                nodes=int(nodes), ahl_sec=round(ahl_sec, 3),
	                search_sec=round(el, 3), total_sec=round(ahl_sec + el, 3)),
	           open(out_path, 'w'))


def run_arm(arm, ahl_on, data_dir, out_dir, ckpt, time_limit, ahl_share, block, tries, jobs, seed_off=0):
	files = sorted(Path(data_dir).glob('*.json'))
	out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
	pending, running, done = list(files), [], 0
	t0 = time.time()
	while pending or running:
		while pending and len(running) < jobs:
			fp = pending.pop(0)
			op = out_dir / (fp.stem + '_r.json')
			p = Process(target=_worker, args=(arm, ahl_on, str(fp), str(op), ckpt,
			                                   time_limit, ahl_share, block, tries, seed_off))
			p.start(); running.append(dict(p=p, fp=fp, op=op, s=time.time()))
		time.sleep(1)
		still = []
		for r in running:
			if r['p'].is_alive():
				if time.time() - r['s'] > time_limit + 120:
					r['p'].kill(); r['p'].join()
					json.dump(dict(instance=r['fp'].name, gt_feasible=True, solved=False,
					                solved_by=None, nodes=-1, ahl_sec=0.0,
					                search_sec=time_limit, total_sec=time_limit), open(r['op'], 'w'))
					done += 1
				else:
					still.append(r)
			else:
				r['p'].join(); done += 1
		running = still
	print(f"    [{arm} ahl={'on' if ahl_on else 'off'}] {done}/{len(files)} in {time.time()-t0:.0f}s", flush=True)


def summarize(out_dir):
	rows = [json.load(open(f)) for f in sorted(Path(out_dir).glob('*_r.json'))]
	feas = [r for r in rows if r['gt_feasible']]
	sol = [r for r in feas if r['solved']]
	by_ahl = [r for r in sol if r['solved_by'] == 'ahl']
	by_search = [r for r in sol if r['solved_by'] == 'search']
	resid = [r for r in feas if r['solved_by'] != 'ahl']
	return dict(n_feas=len(feas), n_solved=len(sol), rate=len(sol) / max(1, len(feas)),
	             n_ahl=len(by_ahl), n_search=len(by_search),
	             n_residual=len(resid),
	             search_rate_on_residual=len(by_search) / max(1, len(resid)),
	             med_total=float(np.median([r['total_sec'] for r in sol])) if sol else float('nan'),
	             solved_set=set(r['instance'] for r in sol))


def mcnemar(a, b):
	oa = len(a['solved_set'] - b['solved_set'])
	ob = len(b['solved_set'] - a['solved_set'])
	n = oa + ob
	p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(max(oa, ob), n + 1)) / 2 ** n)
	return oa, ob, p


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--tag', required=True)
	ap.add_argument('--time_limit', type=float, required=True)
	ap.add_argument('--block', type=int, required=True)
	ap.add_argument('--tries', type=int, default=10)
	ap.add_argument('--arms', nargs='+', default=['lp', 'model'])
	ap.add_argument('--ckpt', required=True)
	ap.add_argument('--ahl', nargs='+', default=['off', 'on'])
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--out_root', default='../../runs/v23')
	ap.add_argument('--seed_offset', type=int, default=0)
	a = ap.parse_args()

	res = {}
	for mode in a.ahl:
		on = (mode == 'on')
		for arm in a.arms:
			od = f"{a.out_root}/{a.tag}_ahl{mode}_{arm}"
			print(f"  === {a.tag} ahl={mode} {arm} ===", flush=True)
			run_arm(arm, on, a.data_dir, od, a.ckpt, a.time_limit, 0.5, a.block, a.tries, a.jobs, a.seed_offset)
			res[(mode, arm)] = summarize(od)

	print("\n" + "=" * 92)
	print(f"v23-warm {a.tag}  (budget {a.time_limit}s, AHL block={a.block} tries={a.tries})")
	print("=" * 92)
	print(f"{'AHL':>5}{'arm':>8}{'solve rate':>14}{'by AHL':>9}{'by search':>11}"
	      f"{'residual':>10}{'search on resid':>17}{'med sec':>10}")
	for mode in a.ahl:
		for arm in a.arms:
			r = res[(mode, arm)]
			print(f"{mode:>5}{arm:>8}{100*r['rate']:>12.1f}% {r['n_ahl']:>8}{r['n_search']:>11}"
			      f"{r['n_residual']:>10}{100*r['search_rate_on_residual']:>16.1f}%{r['med_total']:>10.2f}")
	for mode in a.ahl:
		if ('model' in a.arms) and ('lp' in a.arms):
			oa, ob, p = mcnemar(res[(mode, 'model')], res[(mode, 'lp')])
			print(f"  [ahl={mode}] model만 {oa}개, lp만 {ob}개, McNemar p={p:.3f}")
	out = {f"{k[0]}_{k[1]}": {kk: (list(vv) if isinstance(vv, set) else vv)
	                            for kk, vv in v.items()} for k, v in res.items()}
	json.dump(out, open(f"{a.out_root}/summary_{a.tag}.json", 'w'), indent=2)


if __name__ == '__main__':
	main()
