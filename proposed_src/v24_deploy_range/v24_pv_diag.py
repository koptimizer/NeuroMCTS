"""What did the cost-aware fine-tuning change? On held-out range states: how often the fine-tuned
policy branches on a different variable than the supervised rule, where its pick sits in the
confidence ranking, and whether its pick triggers more propagation (a cheap proxy for the
child-shrinking effect the marginal cannot see)."""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, json, numpy as np, torch, gurobipy as gp
import LPneuroBLS_v7 as m
from v24_rl_core import load_pv
from v23_warm_search import WarmLP
from v23_warmstart_cost import features
torch.set_num_threads(1)

def cascade(A, b, j, v):
	# variables propagation fixes after branching x_j = v (child instance), or -1 if the child is dead
	keep = np.setdiff1d(np.arange(A.shape[1]), [j]); Ar = A[:, keep]; br = b - A[:, j] * v
	if np.any(br < 0) or np.any(br > Ar.sum(1)): return -1
	e = m.FastBinaryEnv(Ar.astype(np.int32), br.astype(np.int32)); e.propagate_constraints()
	return -1 if e.is_invalid else int((e.assignment != -1).sum())

ap = argparse.ArgumentParser()
ap.add_argument('--states', default='../../runs/v24/cond_21x60_range_test.json')
ap.add_argument('--pv', default='../../runs/v24/r2/pv_best.pt')
ap.add_argument('--n_free', type=int, nargs='+', default=[45, 50, 55, 60])
ap.add_argument('--n', type=int, default=300)
a = ap.parse_args()
net = load_pv('../../runs/v22/model_n25/conditional.pt', a.pv)
recs = [r for r in json.load(open(a.states)) if r['n_free'] in a.n_free][:a.n]
env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
rows = []
for r in recs:
	A = np.array(r['A'], np.int64); b = np.array(r['b'], np.int64); pt = np.array(r['marginals'])
	w = WarmLP(A, b, env); w.set_node([])
	if not w.solve(): w.dispose(); continue
	x = w.values(np.arange(A.shape[1])); w.dispose()
	with torch.no_grad(): ml, pl, V = net(*features(A, b, x, torch.device('cpu')))
	p = torch.sigmoid(ml).numpy(); conf = np.abs(p - 0.5)
	j0 = int(np.argmax(conf)); j1 = int(torch.argmax(pl)); v0 = int(p[j0] >= 0.5); v1 = int(p[j1] >= 0.5)
	rank1 = int((conf > conf[j1]).sum()) + 1
	det = (pt < 1e-9) | (pt > 1 - 1e-9)
	rows.append(dict(same=j0 == j1, rank_pv=rank1, conf0=conf[j0], conf1=conf[j1],
	                 det0=bool(det[j0]), det1=bool(det[j1]), ok0=bool(det[j0] and (v0 == (pt[j0] > 0.5))), ok1=bool(det[j1] and (v1 == (pt[j1] > 0.5))),
	                 casc0=cascade(A, b, j0, v0), casc1=cascade(A, b, j1, v1), coldeg0=int(A[:, j0].sum()), coldeg1=int(A[:, j1].sum())))
env.dispose()
R = rows; same = np.mean([r['same'] for r in R]); D = [r for r in R if not r['same']]
print(f"states {len(R)}: PV picks the same root variable as the supervised rule in {100*same:.1f}%")
if D:
	print(f"when different ({len(D)}): PV pick's confidence rank median {np.median([r['rank_pv'] for r in D]):.0f} (p90 {np.percentile([r['rank_pv'] for r in D],90):.0f}); "
	      f"|p-.5| chosen {np.mean([r['conf1'] for r in D]):.3f} vs rule {np.mean([r['conf0'] for r in D]):.3f}")
	print(f"  DET & correct first value: rule {100*np.mean([r['ok0'] for r in D]):.1f}%  PV {100*np.mean([r['ok1'] for r in D]):.1f}%   (DET share: rule {100*np.mean([r['det0'] for r in D]):.0f}%, PV {100*np.mean([r['det1'] for r in D]):.0f}%)")
	print(f"  propagation cascade after the first child: rule median {np.median([r['casc0'] for r in D]):.0f}, PV median {np.median([r['casc1'] for r in D]):.0f}; mean {np.mean([r['casc0'] for r in D]):.1f} vs {np.mean([r['casc1'] for r in D]):.1f}")
	print(f"  column degree of the picked variable: rule {np.mean([r['coldeg0'] for r in D]):.1f}, PV {np.mean([r['coldeg1'] for r in D]):.1f}")
