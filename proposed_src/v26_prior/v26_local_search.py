"""v26 E2: NSNet's way of using marginals — estimate once at the root, round to an assignment,
then repair it by local search (WalkSAT-style on the residual |Ax - b|_1) — on our instances.
Initialisations: s1 (Stage-1 marginals), bp (sum-product marginals), lp (rounded LP relaxation),
uniform (random 0/1). Reports solve rate within a wall-clock budget, flips and time.
Usage: v26_local_search.py --data_dir ... --tag B --time_limit 300 --inits s1 bp lp uniform
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, glob, json, os, time, zlib
from multiprocessing import Pool
import numpy as np
import torch

torch.set_num_threads(1)


def walksat(A, b, x0, time_limit, rng, p_noise=0.3, restart_every=20000):
	# flip one variable per step: pick a violated row, flip the variable of that row that most reduces
	# |r|_1 (with probability p_noise a random one); periodic restarts from a perturbed x0
	mm, n = A.shape; At = A.T
	x = x0.copy(); r = A.dot(x) - b; best = np.abs(r).sum(); flips = 0; t0 = time.time()
	while time.time() - t0 < time_limit:
		if best == 0: return True, flips, time.time() - t0
		if flips and flips % restart_every == 0:
			x = x0.copy(); k = rng.integers(1, max(2, n // 10)); x[rng.choice(n, k, replace=False)] ^= 1; r = A.dot(x) - b
		viol = np.flatnonzero(r)
		if len(viol) == 0: return True, flips, time.time() - t0
		i = viol[rng.integers(len(viol))]; J = np.flatnonzero(A[i])
		if rng.random() < p_noise:
			j = J[rng.integers(len(J))]
		else:
			s = 1 - 2 * x[J]                              # +1 if flipping 0->1, -1 if 1->0
			d = np.abs(r[None, :] + s[:, None] * At[J]).sum(1) - np.abs(r).sum()
			j = J[int(np.argmin(d))]
		x[j] ^= 1; r += (1 if x[j] else -1) * At[j]; flips += 1
		best = min(best, np.abs(r).sum())
	return False, flips, time.time() - t0


def init_assignment(kind, A, b, ckpt, rng):
	import LPneuroBLS_v7 as m
	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if kind == 'uniform': return (rng.random(A.shape[1]) < 0.5).astype(np.int64)
	if kind == 'lp': return (np.clip(x_lp, 0, 1) >= 0.5).astype(np.int64)
	if kind == 'bp':
		from v25_bp import bp_marginals
		p, _ = bp_marginals(A, b); return (p >= 0.5).astype(np.int64)
	from v15_train_marginal import MarginalNet
	from v23_warmstart_cost import features
	net = MarginalNet(); net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']); net.eval()
	with torch.no_grad(): p = torch.sigmoid(net(*features(A, b, x_lp, torch.device('cpu')))).numpy()
	return (p >= 0.5).astype(np.int64)


def one(task):
	f, kind, ckpt, tl, seed_off = task
	d = json.load(open(f)); A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
	rng = np.random.default_rng((zlib.crc32(_Path(f).name.encode()) + seed_off * 1_000_003) % (2 ** 31))
	t0 = time.time(); x0 = init_assignment(kind, A, b, ckpt, rng); t_init = time.time() - t0
	ok, flips, sec = walksat(A, b, x0, tl - t_init, rng)
	return dict(instance=_Path(f).name, init=kind, solved=bool(ok), flips=int(flips), sec=round(t_init + sec, 3), init_sec=round(t_init, 3), init_residual=int(np.abs(A.dot(x0) - b).sum()))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True); ap.add_argument('--tag', required=True)
	ap.add_argument('--time_limit', type=float, required=True); ap.add_argument('--inits', nargs='+', default=['s1', 'bp', 'lp', 'uniform'])
	ap.add_argument('--ckpt_s1', default='../../runs/v22/model_n25/conditional.pt'); ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--seed_offset', type=int, default=0); ap.add_argument('--out_root', default='../../runs/v26')
	a = ap.parse_args()
	fs = sorted(glob.glob(a.data_dir + '/*.json'))
	for kind in a.inits:
		out = f'{a.out_root}/LS_{a.tag}_S{a.seed_offset}_{kind}'; os.makedirs(out, exist_ok=True)
		t0 = time.time()
		with Pool(a.jobs, maxtasksperchild=4) as pool:
			rs = pool.map(one, [(f, kind, a.ckpt_s1, a.time_limit, a.seed_offset) for f in fs])
		for r in rs: json.dump(r, open(f"{out}/{r['instance']}", 'w'))
		sol = [r for r in rs if r['solved']]
		print(f"  [LS {kind}] solved {len(sol)}/{len(rs)}  median sec {np.median([r['sec'] for r in sol]) if sol else float('nan'):.2f}  median flips {np.median([r['flips'] for r in sol]) if sol else float('nan'):,.0f}  median init residual {np.median([r['init_residual'] for r in rs]):.0f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
	main()
