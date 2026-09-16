"""v24 R0: is there anything left for a cost-aware policy to learn?

The supervised guidance branches on argmax |p_hat - 1/2|. That ranks variables by how sure the
network is of their value and ignores what fixing them does to the rest of the instance (how
much propagation cascades, how the relaxation tightens). R0 measures the gap directly: on
held-out states, the root decision is forced to each of the top-k candidates by confidence and
the search below it is otherwise identical, so the subtree size of each candidate is an exact
cost. If the confidence-ranked choice is already the cheapest almost always, RL fine-tuning
(Track 3) has nothing to gain and is not started.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import time
import numpy as np
import torch
import gurobipy as gp

from v15_train_marginal import MarginalNet
from v23_warm_search import GUIDES, WarmLP
from v23_warmstart_cost import features
from v24_search_util import search

torch.set_num_threads(1)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--states', required=True)
	ap.add_argument('--ckpt', required=True)
	ap.add_argument('--n_free', type=int, nargs='+', default=[45, 50])
	ap.add_argument('--n_states', type=int, default=200)
	ap.add_argument('--topk', type=int, default=5)
	ap.add_argument('--budget', type=float, default=120.0)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	dev = torch.device('cpu')
	net = MarginalNet()
	net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)['model_state_dict'])
	net.eval()
	recs = [r for r in json.load(open(a.states)) if r['n_free'] in a.n_free][:a.n_states]
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
	rows = []
	t0 = time.time()
	for k, r in enumerate(recs):
		A = np.array(r['A'], np.int64); b = np.array(r['b'], np.int64)
		warm = WarmLP(A, b, env); warm.set_node([])
		if not warm.solve():
			warm.dispose(); continue
		x = warm.values(np.arange(A.shape[1])); warm.dispose()
		with torch.no_grad():
			p = torch.sigmoid(net(*features(A, b, x, dev))).numpy()
		conf = np.abs(p - 0.5)
		cands = np.argsort(-conf)[:a.topk]
		j_lp = int(np.argmax(np.abs(x - 0.5)))
		costs = {}
		for j in list(cands) + [j_lp]:
			j = int(j)
			if j in costs:
				continue
			v = int(p[j] >= 0.5) if j != j_lp or j in cands else int(x[j] >= 0.5)
			ok, nodes, sec = search(A, b, GUIDES['model'], net, dev, a.budget, 0, env, root_action=(j, v))
			costs[j] = nodes if ok else None
		chosen = int(cands[0])
		valid = {j: c for j, c in costs.items() if c is not None and j in set(int(c_) for c_ in cands)}
		if costs[chosen] is None or not valid:
			continue
		best = min(valid, key=valid.get)
		rows.append(dict(n_free=r['n_free'], chosen=chosen, cost_chosen=costs[chosen],
		                 best=int(best), cost_best=valid[best], cost_lp=costs[j_lp],
		                 costs={int(j): c for j, c in costs.items()}))
		if (k + 1) % 20 == 0:
			print(f"  {k+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
	env.dispose()
	json.dump(rows, open(a.out, 'w'), indent=1)
	cc = np.array([r['cost_chosen'] for r in rows]); cb = np.array([r['cost_best'] for r in rows])
	cl = np.array([r['cost_lp'] if r['cost_lp'] is not None else np.nan for r in rows])
	print(f"\nR0 on {len(rows)} states (n_free {a.n_free}, top-{a.topk} candidates, budget {a.budget}s)")
	print(f"  confidence choice is the cheapest candidate : {100*np.mean(cc == cb):.1f}%")
	print(f"  median cost(chosen)/cost(best)              : {np.median(cc / np.maximum(cb, 1)):.2f}")
	print(f"  mean   cost(chosen)/cost(best)              : {np.mean(cc / np.maximum(cb, 1)):.2f}")
	print(f"  best candidate at least 2x cheaper          : {100*np.mean(cc >= 2 * cb):.1f}% of states")
	print(f"  median cost(LP choice)/cost(chosen)         : {np.nanmedian(cl / np.maximum(cc, 1)):.2f}")
	print(f"  attainable node reduction if root always best: {100*(1 - cb.sum()/cc.sum()):.1f}% (sum over states)")
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
