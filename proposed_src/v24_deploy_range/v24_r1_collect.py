"""v24 R1 data: exact per-decision subtree costs under the supervised policy (M0).

Runs the recording DFS from range states (n_free 45-55) and keeps a random subset of the
recorded decisions per state, so that no single deep subtree dominates the critic's data.
Also self-checks that the recursive DFS reproduces the iterative loop's node counts.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import time
from multiprocessing import Pool
import numpy as np
import torch
import gurobipy as gp

from v24_rl_core import load_pv, dfs_collect
from v23_warm_search import GUIDES, search as search_iter

torch.set_num_threads(1)
NET = None


def init(ckpt):
	global NET
	NET = load_pv(ckpt)


def one(task):
	idx, rec, budget, per_state = task
	A = np.array(rec['A'], np.int64); b = np.array(rec['b'], np.int64)
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
	t0 = time.time()
	found, nodes, recs = dfs_collect(A, b, NET, torch.device('cpu'), env, budget)
	env.dispose()
	rng = np.random.default_rng(idx)
	if len(recs) > per_state:
		recs = [recs[i] for i in sorted(rng.choice(len(recs), per_state, replace=False))]
	for r in recs:
		r['src'] = idx; r['src_n_free'] = rec['n_free']
	return idx, found, nodes, len(recs), time.time() - t0, recs


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--states', required=True)
	ap.add_argument('--ckpt', default='../../runs/v22/model_n25/conditional.pt')
	ap.add_argument('--n_free', type=int, nargs='+', default=[45, 50, 55])
	ap.add_argument('--limit', type=int, default=100000)
	ap.add_argument('--budget', type=float, default=60.0)
	ap.add_argument('--per_state', type=int, default=25)
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--selfcheck', type=int, default=3)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	states = [r for r in json.load(open(a.states)) if r['n_free'] in a.n_free][:a.limit]
	# self-check: recursive DFS == iterative loop on the first few states
	init(a.ckpt)
	from v15_train_marginal import MarginalNet
	base = MarginalNet(); base.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=False)['model_state_dict']); base.eval()
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
	for r in states[:a.selfcheck]:
		A = np.array(r['A'], np.int64); b = np.array(r['b'], np.int64)
		f1, n1, _ = dfs_collect(A, b, NET, torch.device('cpu'), env, a.budget, record=False)
		f2, n2, _ = search_iter(A, b, GUIDES['model'], base, torch.device('cpu'), a.budget, 0, env)
		print(f"  selfcheck n_free={r['n_free']}: recursive found={f1} nodes={n1} | iterative found={f2} nodes={n2} | {'OK' if (f1, n1) == (f2, n2) else 'MISMATCH'}", flush=True)
	env.dispose()
	print(f"collecting from {len(states)} states (n_free {a.n_free}), budget {a.budget}s, {a.per_state}/state", flush=True)
	out, t0 = [], time.time()
	solved, tot_nodes = 0, []
	with Pool(a.jobs, initializer=init, initargs=(a.ckpt,)) as pool:
		for k, (idx, found, nodes, kept, sec, recs) in enumerate(pool.imap_unordered(one, [(i, r, a.budget, a.per_state) for i, r in enumerate(states)]), 1):
			out.extend(recs); solved += found; tot_nodes.append(nodes)
			if k % 100 == 0:
				print(f"  {k}/{len(states)} states, {len(out)} decisions, solved {solved}/{k}, median nodes {np.median(tot_nodes):.0f} ({time.time()-t0:.0f}s)", flush=True)
	costs = np.array([r['cost'] for r in out])
	print(f"\n{len(out)} decisions from {len(states)} states; solved {solved}/{len(states)}; "
	      f"cost median {np.median(costs):.0f}, p10 {np.percentile(costs,10):.0f}, p90 {np.percentile(costs,90):.0f}; "
	      f"found-subtree share {np.mean([r['found'] for r in out]):.2f}")
	json.dump(out, open(a.out, 'w'))
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
