"""v26 E3: the learned branching rules inside SCIP's own branch-and-bound.

A PySCIPOpt Branchrule plugin re-derives the reduced instance at every node from the local bounds,
takes the node LP solution from SCIP, and branches with one of our rules (model = Stage 1,
pv = Stage 1+2, maxsd, lpfrac = most-integral LP value in Python, random); the preferred value's
child gets the higher node-selection priority so SCIP's DFS node selector descends into it first.
SCIP's own rules (relpscost = default, mostinf, inference, random) run under the same settings.
Settings for the rule comparison: DFS node selection, first solution stops the solve, presolving,
separation and primal heuristics off (so that the search is decided by branching), conflict
analysis on or off (--conflict). The network runs through the dense TorchScript path.
Usage: v26_scip_branch.py --data_dir ... --tag H --time_limit 600 --rules pv model maxsd scip_relpscost ...
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, glob, json, os, time, zlib
from multiprocessing import Pool
import numpy as np
import torch
from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING, quicksum

torch.set_num_threads(1)
NET = {}


class BLSBranch(Branchrule):
	"""Our branching rules as a SCIP plugin: reduce from local bounds, score, create two prioritised children."""

	def __init__(self, rule, A, b, rng):
		self.rule = rule; self.A = A; self.b = b; self.rng = rng; self.calls = 0; self.sec = 0.0; self.vars = None

	def branchexeclp(self, allowaddcons):
		t0 = time.perf_counter(); m = self.model
		if self.vars is None:
			self.vars = sorted(m.getVars(transformed=True), key=lambda v: int(v.name.split('_')[-1]))
		lb = np.fromiter((v.getLbLocal() for v in self.vars), float, len(self.vars))
		ub = np.fromiter((v.getUbLocal() for v in self.vars), float, len(self.vars))
		free = np.flatnonzero(ub - lb > 0.5)
		bp = self.b - self.A[:, lb > 0.5].sum(1)
		Ak = self.A[:, free]; rows = np.flatnonzero(Ak.sum(1) > 0)
		Ak = Ak[rows]; bk = bp[rows]
		x = np.fromiter((self.vars[j].getLPSol() for j in free), float, len(free))
		x = np.clip(x, 0.0, 1.0)
		if self.rule.endswith('_glp'):                    # diagnostic: the LP vertex our loop would see (Gurobi dual simplex on the reduced instance)
			import LPneuroBLS_v7 as lpm
			xg, ok = lpm.solve_lp_relaxation(Ak.astype(np.int32), bk.astype(np.int32))
			if ok: x = np.clip(xg, 0.0, 1.0)
		if self.rule == 'lpfrac':
			j = int(np.argmax(np.abs(x - 0.5))); v = int(x[j] >= 0.5)
		elif self.rule == 'random':
			j = int(self.rng.integers(len(free))); v = int(self.rng.integers(2))
		elif self.rule == 'maxsd':
			r = np.maximum(Ak.sum(1), 1); d1 = bk / r
			D1 = Ak * d1[:, None]; D0 = Ak * (1 - d1)[:, None]
			s1 = D1.max(0) + 1e-6 * D1.sum(0); s0 = D0.max(0) + 1e-6 * D0.sum(0)
			j, v = (int(np.argmax(s1)), 1) if s1.max() >= s0.max() else (int(np.argmax(s0)), 0)
		else:
			from v25_dense import dense_features
			with torch.no_grad():
				p, g = NET['net'](*dense_features(Ak, bk, x))
			p = p.numpy()
			j = int(np.argmax(g.numpy())) if self.rule.startswith('pv') else int(np.argmax(np.abs(p - 0.5)))
			v = int(p[j] >= 0.5)
		var = self.vars[free[j]]
		est = m.getLocalEstimate() if hasattr(m, 'getLocalEstimate') else 0.0
		pref = m.createChild(2.0, est); other = m.createChild(1.0, est)
		if v == 1:
			m.chgVarLbNode(pref, var, 1.0); m.chgVarUbNode(other, var, 0.0)
		else:
			m.chgVarUbNode(pref, var, 0.0); m.chgVarLbNode(other, var, 1.0)
		self.calls += 1; self.sec += time.perf_counter() - t0
		return {'result': SCIP_RESULT.BRANCHED}


def build(A, b, time_limit, conflict, presolve, heur, seed):
	mm, n = A.shape
	md = Model('bls'); md.hideOutput()
	x = [md.addVar(name=f'x_{j}', vtype='B') for j in range(n)]
	for i in range(mm):
		md.addCons(quicksum(int(A[i, j]) * x[j] for j in np.flatnonzero(A[i])) == int(b[i]))
	md.setObjective(0.0, 'minimize')
	md.setRealParam('limits/time', time_limit); md.setIntParam('limits/solutions', 1)
	md.setIntParam('nodeselection/dfs/stdpriority', 1000000); md.setIntParam('nodeselection/dfs/memsavepriority', 1000000)
	md.setBoolParam('conflict/enable', conflict)
	md.setIntParam('randomization/randomseedshift', seed)
	if not presolve: md.setPresolve(SCIP_PARAMSETTING.OFF)
	if not heur: md.setHeuristics(SCIP_PARAMSETTING.OFF)
	md.setSeparating(SCIP_PARAMSETTING.OFF)
	return md, x


def one(task):
	f, rule, tl, conflict, presolve, heur, seed_off, ckpt_s1, ckpt_pv = task
	d = json.load(open(f)); A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
	seed = (zlib.crc32(_Path(f).name.encode()) + seed_off * 1_000_003) % (2 ** 31)
	rng = np.random.default_rng(seed)
	if rule in ('model', 'pv', 'model_glp', 'pv_glp') and 'net' not in NET:
		from v25_dense import load_dense
		NET['net'] = load_dense(ckpt_s1, ckpt_pv if rule.startswith('pv') else None, script=True)
	md, x = build(A, b, tl, conflict, presolve, heur, seed % 100000)
	br = None
	if rule.startswith('scip_'):
		md.setIntParam(f'branching/{rule[5:]}/priority', 1000000)
	else:
		br = BLSBranch(rule, A, b, rng); md.includeBranchrule(br, 'bls', 'BLS learned/classical branching', priority=1000000, maxdepth=-1, maxbounddist=1.0)
	t0 = time.time(); md.optimize(); sec = time.time() - t0
	solved = md.getNSols() > 0
	if solved:
		sol = md.getBestSol(); xs = np.array([round(md.getSolVal(sol, v)) for v in x], dtype=np.int64); solved = bool(np.array_equal(A.dot(xs), b))
	return dict(instance=_Path(f).name, rule=rule, conflict=conflict, presolve=presolve, heur=heur, seed_offset=seed_off, solved=solved, nodes=int(md.getNNodes()), sec=round(sec, 3),
	            scip_sec=round(md.getSolvingTime(), 3), status=md.getStatus(), br_calls=(br.calls if br else None), br_sec=(round(br.sec, 3) if br else None))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--time_limit', type=float, required=True)
	ap.add_argument('--rules', nargs='+', default=['scip_relpscost', 'scip_mostinf', 'scip_inference', 'scip_random', 'lpfrac', 'random', 'maxsd', 'model', 'pv'])
	ap.add_argument('--conflict', type=int, default=0); ap.add_argument('--presolve', type=int, default=0); ap.add_argument('--heur', type=int, default=0)
	ap.add_argument('--seed_offset', type=int, default=0); ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--ckpt_s1', default='../../runs/v22/model_n25/conditional.pt'); ap.add_argument('--ckpt_pv', default='../../runs/v24/r2/pv_best.pt')
	ap.add_argument('--out_root', default='../../runs/v26')
	a = ap.parse_args()
	fs = sorted(glob.glob(a.data_dir + '/*.json'))
	cfg = f"c{a.conflict}p{a.presolve}h{a.heur}"
	for rule in a.rules:
		out = f'{a.out_root}/SCIP_{a.tag}_S{a.seed_offset}_{cfg}_{rule}'; os.makedirs(out, exist_ok=True)
		t0 = time.time()
		with Pool(a.jobs, maxtasksperchild=1) as pool:
			rs = pool.map(one, [(f, rule, a.time_limit, bool(a.conflict), bool(a.presolve), bool(a.heur), a.seed_offset, a.ckpt_s1, a.ckpt_pv) for f in fs], chunksize=1)
		for r in rs: json.dump(r, open(f"{out}/{r['instance']}", 'w'))
		sol = [r for r in rs if r['solved']]
		ov = f"  branch-callback share {np.median([r['br_sec'] / max(r['sec'], 1e-9) for r in sol]):.2f}" if sol and sol[0]['br_sec'] is not None else ''
		print(f"  [SCIP {cfg} {rule}] solved {len(sol)}/{len(rs)}  median nodes {np.median([r['nodes'] for r in sol]) if sol else float('nan'):,.0f}  median sec {np.median([r['sec'] for r in sol]) if sol else float('nan'):.2f}  PAR-2 {np.mean([r['sec'] if r['solved'] else 2 * a.time_limit for r in rs]):.1f}{ov}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
	main()
