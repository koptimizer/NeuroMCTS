"""v25 B2: learned baselines for Stage 2, same protocol as v24_r2_finetune.py, differing in the
policy's starting point and in what is trained:
  --mode posterior   g_j = log|p_hat_j - 1/2| + h_phi(h_j), h_phi zero-initialised   (the paper)
  --mode lp          g_j = log|x_lp_j  - 1/2| + h_phi(h_j), h_phi zero-initialised   (starts at the LP rule)
  --mode random      g_j = h_phi(h_j) with a random last layer                        (from scratch, FMSTS-style)
  --mode mostfrac    g_j = -log|x_lp_j - 1/2| + h_phi(h_j), h_phi zero-initialised  (P7: starts at most-fractional)
  --unfreeze         also train the trunk and marginal head (the value rule may then drift)
The first value is 1[p_hat >= 1/2] in every mode. The critic is initialised from the R1 value head
(critic_frozen/pv.pt) in every mode, so only the actor's starting point differs.
Output: <out>/pv_ep*.pt, <out>/log.json. Run from proposed_src/v25_review.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, json, math, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gp
from v15_train_marginal import MarginalNet
from v24_rl_core import PolicyValueNet, dfs_collect
from v23_warmstart_cost import features

torch.set_num_threads(1)
NET = None


class VariantNet(PolicyValueNet):
	"""PolicyValueNet with a switchable starting point for the policy logit."""

	def __init__(self, base, mode='posterior', hidden=64):
		super().__init__(base, hidden)
		self.mode = mode
		if mode == 'random':
			nn.init.normal_(self.pol[2].weight, std=0.1); nn.init.zeros_(self.pol[2].bias)

	def forward(self, xv, xc, ev2c, ec2v):
		hv, hc = self.trunk(xv, xc, ev2c, ec2v)
		m_logit = self.base.head(hv).squeeze(-1)
		p = torch.sigmoid(m_logit)
		corr = self.pol(hv).squeeze(-1)
		if self.mode == 'posterior':
			pol_logit = torch.log(torch.abs(p - 0.5) + 1e-6) + corr
		elif self.mode == 'lp':
			pol_logit = torch.log(torch.abs(xv[:, 0] - 0.5) + 1e-6) + corr
		elif self.mode == 'mostfrac':
			pol_logit = -torch.log(torch.abs(xv[:, 0] - 0.5) + 1e-6) + corr
		else:
			pol_logit = corr
		g = torch.cat([hv.mean(0), hv.max(0).values, hc.mean(0), hc.max(0).values, torch.tensor([math.log(hv.shape[0]), math.log(hc.shape[0])])])
		return m_logit, pol_logit, self.val(g).squeeze(-1)


def build(base_ckpt, mode, init_pv=None):
	base = MarginalNet(); base.load_state_dict(torch.load(base_ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
	net = VariantNet(base, mode)
	if init_pv:
		sd = torch.load(init_pv, map_location='cpu', weights_only=False)['model_state_dict']
		net.load_state_dict({k: v for k, v in sd.items() if k.startswith('val.') or k.startswith('base.')}, strict=False)   # critic (and base) from R1; policy head keeps its mode-specific init
	net.eval(); return net


def init(base, mode, state):
	global NET
	NET = build(base, mode)
	if state is not None: NET.load_state_dict(state)
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


def run_pool(fn, tasks, jobs, base, mode, state):
	with Pool(jobs, initializer=init, initargs=(base, mode, state)) as pool:
		return list(pool.imap_unordered(fn, tasks))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--train', default='../../runs/v24/cond_21x60_range_train.json')
	ap.add_argument('--test', default='../../runs/v24/cond_21x60_range_test.json')
	ap.add_argument('--base', default='../../runs/v22/model_n25/conditional.pt')
	ap.add_argument('--init_pv', default='../../runs/v24/critic_frozen/pv.pt')
	ap.add_argument('--mode', choices=['posterior', 'lp', 'random', 'mostfrac'], default='posterior')
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
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
	tr = [r for r in json.load(open(a.train)) if r['n_free'] in a.n_free]
	te = [r for r in json.load(open(a.test)) if r['n_free'] in a.n_free][:a.n_monitor]
	torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
	net = build(a.base, a.mode, a.init_pv)
	params = list(net.pol.parameters()) + list(net.val.parameters()) + (list(net.base.parameters()) if a.unfreeze else [])
	opt = torch.optim.Adam(params, lr=a.lr)
	dev = torch.device('cpu'); log = []

	def state_cpu(): return {k: v.detach().clone() for k, v in net.state_dict().items()}

	print(f"mode={a.mode} unfreeze={a.unfreeze} train states={len(tr)}, monitor states={len(te)}, tau={a.tau}", flush=True)
	t0 = time.time()
	res = run_pool(monitor, [(i, r, a.budget) for i, r in enumerate(te)], a.jobs, a.base, a.mode, state_cpu())
	base_costs = {i: (n if f else None) for i, f, n in res}
	ok = [i for i in base_costs if base_costs[i] is not None]
	print(f"epoch 0 (initial policy): monitor solved {len(ok)}/{len(te)}, median root cost {np.median([base_costs[i] for i in ok]) if ok else float('nan'):.0f} ({time.time()-t0:.0f}s)", flush=True)
	torch.save({'model_state_dict': net.state_dict(), 'mode': a.mode}, out / 'pv_ep0.pt')
	for ep in range(1, a.epochs + 1):
		te0 = time.time()
		pick = rng.choice(len(tr), min(a.states_per_epoch, len(tr)), replace=False)
		tasks = [(int(i), tr[i], a.budget, a.tau, a.cap, 1000 * ep + int(i) + 100000 * a.seed) for i in pick]
		res = run_pool(rollout, tasks, a.jobs, a.base, a.mode, state_cpu())
		recs = [r for _, _, _, rs in res for r in rs]; solved = sum(f for _, f, _, _ in res)
		net.train(); data = []
		for r in recs:
			A = np.array(r['A'], np.int64); b = np.array(r['b'], np.int64); x = np.array(r['x_lp'], np.float64)
			data.append((features(A, b, x, dev), r['j'], math.log(r['cost'])))
		with torch.no_grad():
			V = np.array([float(net(*f)[2]) for f, _, _ in data]); Y = np.array([y for _, _, y in data])
		adv = V - Y; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
		la = lc = lh = 0.0
		for k in rng.permutation(len(data)):
			f, j, y = data[k]
			_, pol, v = net(*f)
			logp = F.log_softmax(pol / a.tau, 0); ent = -(logp.exp() * logp).sum()
			loss_a = -float(adv[k]) * logp[j] - a.entropy * ent; loss_c = F.mse_loss(v, torch.tensor(y))
			opt.zero_grad(); (loss_a + loss_c).backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
			la += loss_a.item(); lc += loss_c.item(); lh += ent.item()
		net.eval()
		torch.save({'model_state_dict': net.state_dict(), 'mode': a.mode}, out / f'pv_ep{ep}.pt')
		res = run_pool(monitor, [(i, r, a.budget) for i, r in enumerate(te)], a.jobs, a.base, a.mode, state_cpu())
		cur = {i: (n if f else None) for i, f, n in res}
		both = [i for i in ok if cur[i] is not None]
		ratio = np.array([cur[i] / base_costs[i] for i in both]) if both else np.array([1.0])
		rec_ = dict(epoch=ep, decisions=len(data), rollouts_solved=f"{solved}/{len(tasks)}", actor=la / max(1, len(data)), critic=lc / max(1, len(data)), entropy=lh / max(1, len(data)),
		            monitor_solved=f"{sum(v is not None for v in cur.values())}/{len(te)}", median_ratio=float(np.median(ratio)), mean_log_ratio=float(np.mean(np.log(ratio))),
		            wins=int((ratio < 1).sum()), ties=int((ratio == 1).sum()), losses=int((ratio > 1).sum()),
		            sum_ratio=float(sum(cur[i] for i in both) / max(1, sum(base_costs[i] for i in both))) if both else None,
		            monitor_total_nodes=int(sum(v for v in cur.values() if v is not None)), sec=time.time() - te0)
		log.append(rec_)
		print(f"epoch {ep}: decisions={len(data)} rollouts solved {rec_['rollouts_solved']} | actor {rec_['actor']:+.3f} critic {rec_['critic']:.3f} H {rec_['entropy']:.2f} | monitor solved {rec_['monitor_solved']} total-node ratio vs epoch 0 {rec_['sum_ratio']} ({rec_['sec']:.0f}s)", flush=True)
		json.dump(log, open(out / 'log.json', 'w'), indent=1)
	print(f"saved: {out}/pv_ep*.pt")


if __name__ == '__main__':
	main()
