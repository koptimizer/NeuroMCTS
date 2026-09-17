"""Isolate the root decision: on held-out disagreement states, force the root to the supervised
rule's pick and to the fine-tuned policy's pick with the SAME supervised policy (M0) below, and
compare total cost, first-child success and the refutation cost of a dead first child.
Saved verbatim from the inline run of 2026-09-17 14:15 (output in runs/v24/pv_root_diag.log)."""
import sys, json, numpy as np, torch, gurobipy as gp
from pathlib import Path
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].iterdir()) if p.is_dir() and not p.name.startswith('.')]
from v24_rl_core import load_pv
from v15_train_marginal import MarginalNet
from v23_warm_search import GUIDES, WarmLP
from v23_warmstart_cost import features
from v24_search_util import search
torch.set_num_threads(1)
net = load_pv('../../runs/v22/model_n25/conditional.pt', '../../runs/v24/r2/pv_best.pt')
base = MarginalNet(); base.load_state_dict(torch.load('../../runs/v22/model_n25/conditional.pt', map_location='cpu', weights_only=False)['model_state_dict']); base.eval()
recs = [r for r in json.load(open('../../runs/v24/cond_21x60_range_test.json')) if r['n_free'] in (50, 55)][:120]
env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start(); dev = torch.device('cpu')
rows = []
for r in recs:
	A = np.array(r['A'], np.int64); b = np.array(r['b'], np.int64); pt = np.array(r['marginals'])
	w = WarmLP(A, b, env); w.set_node([])
	if not w.solve(): w.dispose(); continue
	x = w.values(np.arange(A.shape[1])); w.dispose()
	with torch.no_grad(): ml, pl, _ = net(*features(A, b, x, dev))
	p = torch.sigmoid(ml).numpy(); j0 = int(np.argmax(np.abs(p - 0.5))); j1 = int(torch.argmax(pl))
	if j0 == j1: continue
	out = {}
	for tag, j in [('rule', j0), ('pv', j1)]:
		v = int(p[j] >= 0.5)
		ok, n, _ = search(A, b, GUIDES['model'], base, dev, 90, 0, env, root_action=(j, v))
		keep = np.setdiff1d(np.arange(A.shape[1]), [j]); ok1, n1, _ = search(A[:, keep], b - A[:, j] * v, GUIDES['model'], base, dev, 90, 0, env)
		out[tag] = dict(total=n if ok else None, first_ok=ok1, first_cost=n1, first_right=bool((pt[j] > 0.5) == v) if (pt[j] < 1e-9 or pt[j] > 1 - 1e-9) else None)
	rows.append(out)
env.dispose()
R = [r for r in rows if r['rule']['total'] and r['pv']['total']]
ratio = np.array([r['rule']['total'] / r['pv']['total'] for r in R])
print(f"disagreement states {len(R)} (n_free 50/55), M0 policy below the root in both arms")
print(f"  total cost rule/pv: median {np.median(ratio):.2f}, pv cheaper on {(ratio > 1).sum()}/{len(R)}, mean log ratio {np.mean(np.log(ratio)):+.2f}")
for tag in ['rule', 'pv']:
	fo = np.mean([r[tag]['first_ok'] for r in R]); fc = np.median([r[tag]['first_cost'] for r in R])
	fr = [r[tag]['first_right'] for r in R if r[tag]['first_right'] is not None]
	fail_cost = [r[tag]['first_cost'] for r in R if not r[tag]['first_ok']]
	print(f"  {tag}: first child solvable {100*fo:.0f}% | first-child subtree median {fc:.0f} | dead first child refutation cost median {np.median(fail_cost) if fail_cost else float('nan'):.0f} (n={len(fail_cost)}) | DET-correct first value {100*np.mean(fr):.0f}% of DET picks (n={len(fr)})")
