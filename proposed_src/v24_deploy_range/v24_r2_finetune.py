"""v24 R2: actor-critic fine-tuning of the branching policy on exact subtree costs.

Each epoch: roll out the current policy with sampling (softmax of the policy logit at
temperature tau) from training states, record every branching decision with its exact subtree
cost c, and update
    critic:  MSE(V(s), log c)
    actor :  -(adv) * log pi(a|s) - lambda * H(pi),   adv = V(s) - log c   (normalised per epoch)
The policy logit starts as log|p_hat - 1/2| (the supervised rule) plus a zero-initialised
correction, so epoch 0 is exactly the deployed heuristic. The base network (trunk and marginal
head) stays frozen unless --unfreeze, so the marginal used for the first value cannot drift.
Held-out monitor: deterministic (argmax) root subtree cost on test states against the frozen
initial policy; every epoch's weights are saved so selection is explicit.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import math
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn.functional as F
import gurobipy as gp

from v24_rl_core import load_pv, dfs_collect
from v23_warmstart_cost import features

torch.set_num_threads(1)
NET = None


def init(base, state):
	global NET
	NET = load_pv(base)
	if state is not None:
		NET.load_state_dict(state)
	NET.eval()


def rollout(task):
	idx, rec, budget, tau, cap, seed = task
	A = np.array(rec['A'], np.int64); b = np.array(rec['b'], np.int64)
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
	rng = np.random.default_rng(seed)
	found, nodes, recs = dfs_collect(A, b, NET, torch.device('cpu'), env, budget, tau=tau, rng=rng)
	env.dispose()
	if cap and len(recs) > cap:
		recs = [recs[i] for i in sorted(rng.choice(len(recs), cap, replace=False))]
	return idx, found, nodes, recs


def monitor(task):
	idx, rec, budget = task
	A = np.array(rec['A'], np.int64); b = np.array(rec['b'], np.int64)
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
	found, nodes, _ = dfs_collect(A, b, NET, torch.device('cpu'), env, budget, record=False)
	env.dispose()
	return idx, found, nodes


def run_pool(fn, tasks, jobs, base, state):
	with Pool(jobs, initializer=init, initargs=(base, state)) as pool:
		return list(pool.imap_unordered(fn, tasks))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--train', default='../../runs/v24/cond_21x60_range_train.json')
	ap.add_argument('--test', default='../../runs/v24/cond_21x60_range_test.json')
	ap.add_argument('--base', default='../../runs/v22/model_n25/conditional.pt')
	ap.add_argument('--init_pv', default=None, help='critic-pretrained pv.pt from R1')
	ap.add_argument('--n_free', type=int, nargs='+', default=[45, 50, 55, 60])
	ap.add_argument('--states_per_epoch', type=int, default=400)
	ap.add_argument('--n_monitor', type=int, default=150)
	ap.add_argument('--epochs', type=int, default=6)
	ap.add_argument('--tau', type=float, default=1.0)
	ap.add_argument('--budget', type=float, default=90.0)
	ap.add_argument('--cap', type=int, default=60)
	ap.add_argument('--lr', type=float, default=3e-4)
	ap.add_argument('--entropy', type=float, default=0.01)
	ap.add_argument('--unfreeze', action='store_true')
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
	tr = [r for r in json.load(open(a.train)) if r['n_free'] in a.n_free]
	te = [r for r in json.load(open(a.test)) if r['n_free'] in a.n_free][:a.n_monitor]
	net = load_pv(a.base, a.init_pv)
	params = list(net.pol.parameters()) + list(net.val.parameters()) + (list(net.base.parameters()) if a.unfreeze else [])
	opt = torch.optim.Adam(params, lr=a.lr)
	rng = np.random.default_rng(0)
	dev = torch.device('cpu')
	log = []

	def state_cpu(): return {k: v.detach().clone() for k, v in net.state_dict().items()}

	print(f"train states={len(tr)}, monitor states={len(te)}, tau={a.tau}, unfreeze={a.unfreeze}", flush=True)
	t0 = time.time()
	base_costs = {}
	res = run_pool(monitor, [(i, r, a.budget) for i, r in enumerate(te)], a.jobs, a.base, state_cpu())
	for i, f, n in res:
		base_costs[i] = n if f else None
	ok = [i for i in base_costs if base_costs[i] is not None]
	print(f"epoch 0 (initial policy): monitor solved {len(ok)}/{len(te)}, median root cost {np.median([base_costs[i] for i in ok]):.0f} ({time.time()-t0:.0f}s)", flush=True)
	torch.save({'model_state_dict': net.state_dict()}, out / 'pv_ep0.pt')

	for ep in range(1, a.epochs + 1):
		te0 = time.time()
		pick = rng.choice(len(tr), min(a.states_per_epoch, len(tr)), replace=False)
		tasks = [(int(i), tr[i], a.budget, a.tau, a.cap, 1000 * ep + int(i)) for i in pick]
		res = run_pool(rollout, tasks, a.jobs, a.base, state_cpu())
		recs = [r for _, _, _, rs in res for r in rs]
		solved = sum(f for _, f, _, _ in res)
		# gradient pass over recorded decisions
		net.train()
		data = []
		for r in recs:
			A = np.array(r['A'], np.int64); b = np.array(r['b'], np.int64); x = np.array(r['x_lp'], np.float64)
			data.append((features(A, b, x, dev), r['j'], math.log(r['cost'])))
		with torch.no_grad():
			V = np.array([float(net(*f)[2]) for f, _, _ in data]); Y = np.array([y for _, _, y in data])
		adv = V - Y
		adv = (adv - adv.mean()) / (adv.std() + 1e-6)
		order = rng.permutation(len(data))
		la = lc = lh = 0.0
		for k in order:
			f, j, y = data[k]
			_, pol, v = net(*f)
			logp = F.log_softmax(pol / a.tau, 0)
			ent = -(logp.exp() * logp).sum()
			loss_a = -float(adv[k]) * logp[j] - a.entropy * ent
			loss_c = F.mse_loss(v, torch.tensor(y))
			opt.zero_grad(); (loss_a + loss_c).backward()
			torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
			la += loss_a.item(); lc += loss_c.item(); lh += ent.item()
		net.eval()
		torch.save({'model_state_dict': net.state_dict()}, out / f'pv_ep{ep}.pt')
		# monitor: deterministic root cost on held-out states vs the initial policy
		res = run_pool(monitor, [(i, r, a.budget) for i, r in enumerate(te)], a.jobs, a.base, state_cpu())
		cur = {i: (n if f else None) for i, f, n in res}
		both = [i for i in ok if cur[i] is not None]
		ratio = np.array([cur[i] / base_costs[i] for i in both])
		wins = int((ratio < 1).sum()); ties = int((ratio == 1).sum())
		rec_ = dict(epoch=ep, decisions=len(data), rollouts_solved=f"{solved}/{len(tasks)}",
		            actor=la / len(data), critic=lc / len(data), entropy=lh / len(data),
		            monitor_solved=f"{len(both)}/{len(te)}", median_ratio=float(np.median(ratio)),
		            mean_log_ratio=float(np.mean(np.log(ratio))), wins=wins, ties=ties, losses=len(both) - wins - ties,
		            sum_ratio=float(sum(cur[i] for i in both) / sum(base_costs[i] for i in both)), sec=time.time() - te0)
		log.append(rec_)
		print(f"epoch {ep}: decisions={len(data)} rollouts solved {rec_['rollouts_solved']} | actor {rec_['actor']:+.3f} critic {rec_['critic']:.3f} H {rec_['entropy']:.2f} | "
		      f"monitor: median cost ratio {rec_['median_ratio']:.3f} (mean log {rec_['mean_log_ratio']:+.3f}), wins/ties/losses {wins}/{ties}/{rec_['losses']}, "
		      f"total-node ratio {rec_['sum_ratio']:.3f} ({rec_['sec']:.0f}s)", flush=True)
		json.dump(log, open(out / 'log.json', 'w'), indent=1)
	print(f"saved: {out}/pv_ep*.pt")


if __name__ == '__main__':
	main()
