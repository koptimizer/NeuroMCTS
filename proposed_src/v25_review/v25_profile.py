"""v25 A2: where the time of a node goes, measured over whole searches rather than at the root.
Runs the Stage-1 and Stage-1+2 arms (and the LP rule) on a few instances with per-step timers
(bound check + reduction, propagation, warm-started LP, network forward, bookkeeping) and reports
the share of each step and the mean forward time by number of free variables.
Output: runs/v25/profile_<set>.json. Run from proposed_src/v25_review.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import gurobipy as gp
import LPneuroBLS_v7 as m
from v23_warm_search import WarmLP
from v23_warmstart_cost import features
from v15_train_marginal import MarginalNet
from v24_rl_core import load_pv

torch.set_num_threads(1)


def profiled_search(A0, b0, arm, model, time_limit, env):
	T = dict(reduce=0.0, propagate=0.0, lp=0.0, forward=0.0, other=0.0); fw = []
	warm = WarmLP(A0, b0, env); t0 = time.time(); nodes = 0
	stack = [(A0.copy(), b0.copy(), [], np.arange(A0.shape[1]))]
	while stack:
		if time.time() - t0 > time_limit: break
		ta = time.perf_counter()
		A, b, fixed, alive = stack.pop(); nodes += 1
		if np.any(b < 0) or np.any(b > A.sum(axis=1)): T['reduce'] += time.perf_counter() - ta; continue
		live = A.sum(axis=1) > 0
		if not live.all():
			A, b = A[live], b[live]
			if A.shape[0] == 0: T['reduce'] += time.perf_counter() - ta; break
		tb = time.perf_counter(); T['reduce'] += tb - ta
		e = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32)); e.propagate_constraints()
		tc = time.perf_counter(); T['propagate'] += tc - tb
		if e.is_invalid: continue
		det = e.assignment
		if np.all(det != -1):
			if np.all(A.dot(det.astype(np.int64)) == b): break
			continue
		if np.any(det != -1):
			idx = np.where(det != -1)[0]; vals = det[idx].astype(np.int64); keep = np.setdiff1d(np.arange(A.shape[1]), idx)
			stack.append((A[:, keep], b - A[:, idx].dot(vals), fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep])); continue
		td = time.perf_counter(); warm.set_node(fixed); ok = warm.solve(); te = time.perf_counter(); T['lp'] += te - td
		if not ok: continue
		x = warm.values(alive)
		if arm == 'lp':
			j = int(np.argmax(np.abs(x - 0.5))); v = int(x[j] >= 0.5)
		else:
			tf = time.perf_counter()
			with torch.no_grad():
				out = model(*features(A, b, x, torch.device('cpu')))
			tg = time.perf_counter(); T['forward'] += tg - tf; fw.append((int(A.shape[1]), tg - tf))
			if arm == 'model':
				p = torch.sigmoid(out).numpy(); j = int(np.argmax(np.abs(p - 0.5))); v = int(p[j] >= 0.5)
			else:
				p = torch.sigmoid(out[0]).numpy(); j = int(torch.argmax(out[1])); v = int(p[j] >= 0.5)
		th = time.perf_counter()
		keep = np.setdiff1d(np.arange(A.shape[1]), [j])
		for vv in (1 - v, v):
			stack.append((A[:, keep], b - A[:, j] * vv, fixed + [(int(alive[j]), int(vv))], alive[keep]))
		T['other'] += time.perf_counter() - th
	warm.dispose()
	tot = sum(T.values()); wall = time.time() - t0
	return dict(nodes=nodes, wall=round(wall, 2), accounted=round(tot, 2), shares={k: round(v / max(tot, 1e-9), 3) for k, v in T.items()},
	            ms_per_node=round(1000 * wall / max(nodes, 1), 3), forward_by_free=fw)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--set', default='v24test_21x60_100'); ap.add_argument('--k', type=int, default=6); ap.add_argument('--time_limit', type=float, default=600)
	ap.add_argument('--ckpt_s1', default='../../runs/v22/model_n25/conditional.pt'); ap.add_argument('--ckpt_pv', default='../../runs/v24/r2/pv_best.pt')
	a = ap.parse_args()
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
	s1 = MarginalNet(); s1.load_state_dict(torch.load(a.ckpt_s1, map_location='cpu', weights_only=False)['model_state_dict']); s1.eval()
	pv = load_pv(a.ckpt_s1, a.ckpt_pv)
	out = {}
	for f in sorted(Path(f'../../instances/{a.set}').glob('*.json'))[:a.k]:
		d = json.load(open(f)); A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
		for arm, mdl in [('lp', None), ('model', s1), ('pv', pv)]:
			r = profiled_search(A, b, arm, mdl, a.time_limit, env)
			fw = r.pop('forward_by_free')
			byk = {}
			for k, t in fw: byk.setdefault(k // 10 * 10, []).append(t)
			r['forward_ms_by_free'] = {str(k): round(1000 * float(np.mean(v)), 3) for k, v in sorted(byk.items())}
			out[f'{f.stem}|{arm}'] = r
			print(f"{f.stem} {arm:5s} nodes {r['nodes']:7d}  {r['ms_per_node']:.3f} ms/node  shares {r['shares']}  fwd ms by free {r['forward_ms_by_free']}", flush=True)
	json.dump(out, open(f'../../runs/v25/profile_{a.set}.json', 'w'), indent=1)


if __name__ == '__main__':
	main()
