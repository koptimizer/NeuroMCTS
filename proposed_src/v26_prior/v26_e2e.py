"""v26 E4: end-to-end comparison on the evaluation sets — SCIP with all defaults (presolve, heuristics,
conflict analysis, reliability branching), SCIP with our Stage-1+2 rule as the branching plugin
(presolve off so variables keep their identity; heuristics and conflict analysis on), CP-SAT
(single worker), and lattice reduction with bounded enumeration (BKZ-40, 30 permutations, 300 s
enumeration cap). First feasible solution stops every method; every solution is verified.
Usage: v26_e2e.py --data_dir ... --tag G70 --time_limit 1200 --methods scip_default scip_pv cpsat lattice
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, glob, json, os, time, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def run_scip(A, b, tl, plugin, ckpt_s1, ckpt_pv, seed):
	import torch; torch.set_num_threads(1)
	from pyscipopt import Model, quicksum
	from v26_scip_branch import BLSBranch, NET
	mm, n = A.shape
	md = Model('bls'); md.hideOutput()
	x = [md.addVar(name=f'x_{j}', vtype='B') for j in range(n)]
	for i in range(mm):
		md.addCons(quicksum(int(A[i, j]) * x[j] for j in np.flatnonzero(A[i])) == int(b[i]))
	md.setObjective(0.0, 'minimize'); md.setRealParam('limits/time', tl); md.setIntParam('limits/solutions', 1)
	md.setIntParam('randomization/randomseedshift', seed % 100000)
	if plugin:
		from pyscipopt import SCIP_PARAMSETTING
		from v25_dense import load_dense
		if 'net' not in NET: NET['net'] = load_dense(ckpt_s1, ckpt_pv, script=True)
		md.setPresolve(SCIP_PARAMSETTING.OFF); md.setSeparating(SCIP_PARAMSETTING.OFF)
		md.setIntParam('nodeselection/dfs/stdpriority', 1000000); md.setIntParam('nodeselection/dfs/memsavepriority', 1000000)
		md.includeBranchrule(BLSBranch('pv', A, b, np.random.default_rng(seed)), 'bls', 'BLS learned branching', priority=1000000, maxdepth=-1, maxbounddist=1.0)
	t0 = time.time(); md.optimize(); sec = time.time() - t0
	ok = md.getNSols() > 0
	if ok:
		sol = md.getBestSol(); xs = np.array([round(md.getSolVal(sol, v)) for v in x], dtype=np.int64); ok = bool(np.array_equal(A.dot(xs), b))
	return ok, sec, dict(nodes=int(md.getNNodes()), status=md.getStatus())


def run_cpsat(A, b, tl, seed):
	from ortools.sat.python import cp_model
	mm, n = A.shape
	md = cp_model.CpModel(); x = [md.NewBoolVar(f'x{j}') for j in range(n)]
	for i in range(mm):
		md.Add(sum(x[j] for j in np.flatnonzero(A[i])) == int(b[i]))
	sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl; sv.parameters.num_workers = 1; sv.parameters.random_seed = seed % 100000
	t0 = time.time(); st = sv.Solve(md); sec = time.time() - t0
	ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
	if ok:
		xs = np.array([sv.Value(v) for v in x], dtype=np.int64); ok = bool(np.array_equal(A.dot(xs), b))
	return ok, sec, dict(status=sv.StatusName(st))


def run_lattice(f, tl):
	from v25_lattice_strong import one as lat
	r = lat((f, 40, 30, min(300.0, tl), 0))
	return r['how'] in ('bkz', 'enum'), r['sec_total'], dict(how=r['how'], sec_bkz=r['sec_bkz'])


def one(task):
	f, method, tl, seed_off, ckpt_s1, ckpt_pv = task
	d = json.load(open(f)); A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
	seed = (zlib.crc32(_Path(f).name.encode()) + seed_off * 1_000_003) % (2 ** 31)
	if method == 'scip_default': ok, sec, extra = run_scip(A, b, tl, False, ckpt_s1, ckpt_pv, seed)
	elif method == 'scip_pv': ok, sec, extra = run_scip(A, b, tl, True, ckpt_s1, ckpt_pv, seed)
	elif method == 'cpsat': ok, sec, extra = run_cpsat(A, b, tl, seed)
	else: ok, sec, extra = run_lattice(f, tl)
	return dict(instance=_Path(f).name, method=method, solved=bool(ok), sec=round(float(sec), 3), seed_offset=seed_off, **extra)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--time_limit', type=float, required=True)
	ap.add_argument('--methods', nargs='+', default=['scip_default', 'scip_pv', 'cpsat', 'lattice']); ap.add_argument('--seed_offset', type=int, default=0)
	ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--ckpt_s1', default='../../runs/v22/model_n25/conditional.pt'); ap.add_argument('--ckpt_pv', default='../../runs/v24/r2/pv_best.pt')
	ap.add_argument('--out_root', default='../../runs/v26')
	a = ap.parse_args()
	fs = sorted(glob.glob(a.data_dir + '/*.json'))
	for mth in a.methods:
		out = f'{a.out_root}/E2E_{a.tag}_S{a.seed_offset}_{mth}'; os.makedirs(out, exist_ok=True); t0 = time.time()
		with ProcessPoolExecutor(a.jobs) as ex:                 # not a Pool: the lattice method forks an enumeration subprocess
			rs = list(ex.map(one, [(f, mth, a.time_limit, a.seed_offset, a.ckpt_s1, a.ckpt_pv) for f in fs]))
		for r in rs: json.dump(r, open(f"{out}/{r['instance']}", 'w'))
		sol = [r for r in rs if r['solved']]
		print(f"  [E2E {mth}] solved {len(sol)}/{len(rs)}  median sec {np.median([r['sec'] for r in sol]) if sol else float('nan'):.2f}  PAR-2 {np.mean([r['sec'] if r['solved'] else 2 * a.time_limit for r in rs]):.1f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
	main()
