"""FORCED-only accuracy of five predictors on the 10x25 in-tree test states (260930).

Same states and ground truth as tab:pred (reference solution x, all free variables) plus the
FORCED-only subset (p_true in {0,1}); reported for all depths and depth 8. Predictors:
Bayes ceiling, GAT in-tree (runs/v22/model_n25), GAT root control (runs/revision/ctrl_root),
LP rounding, and an MLP with constraint aggregates trained in-tree on the full train file.
Run from proposed_src/v23_ablation. Output: runs/revision/forced_compare_10x25.log.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path[:0] = [str(p) for p in sorted(Path('..').resolve().iterdir()) if p.is_dir() and not p.name.startswith('.')]
import LPneuroBLS_v7 as m
from v15_train_marginal import MarginalNet, featurize
from v23_mlp_baseline import tabular_features, MLP
torch.set_num_threads(4)
tr = json.load(open('../../runs/v22/cond_10x25_train.json'))
te = json.load(open('../../runs/v22/cond_10x25_test.json'))
dev = torch.device('cpu')


def load(p):
	g = MarginalNet()
	g.load_state_dict(torch.load(p, map_location=dev, weights_only=False)['model_state_dict'])
	g.eval()
	return g


gat_in = load('../../runs/v22/model_n25/conditional.pt')
gat_root = load('../../runs/revision/ctrl_root/conditional.pt')
torch.manual_seed(0); np.random.seed(0)
trX = [(torch.tensor(X), torch.tensor(y)) for X, y in (tabular_features(r, True) for r in tr) if X is not None]
mlp = MLP(9); opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
for ep in range(20):
	for i in np.random.permutation(len(trX)):
		X, y = trX[i]
		loss = F.binary_cross_entropy_with_logits(mlp(X).squeeze(-1), y)
		opt.zero_grad(); loss.backward(); opt.step()
mlp.eval()
rows = []
with torch.no_grad():
	for r in te:
		A = np.array(r['A'], dtype=np.int64); b = np.array(r['b'], dtype=np.int64)
		x = np.array(r['x'], dtype=np.int64); pt = np.array(r['marginals'])
		f = featurize(r, dev)
		if f is None:
			continue
		Xt, _ = tabular_features(r, True)
		x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
		preds = dict(
			gat_in=(torch.sigmoid(gat_in(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy() >= 0.5).astype(int),
			gat_root=(torch.sigmoid(gat_root(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy() >= 0.5).astype(int),
			lp=(np.clip(x_lp, 0, 1) >= 0.5).astype(int),
			mlp_in=(torch.sigmoid(mlp(torch.tensor(Xt)).squeeze(-1)).numpy() >= 0.5).astype(int))
		forced = (pt < 1e-9) | (pt > 1 - 1e-9)
		rec = dict(depth=r['depth'], n_forced=int(forced.sum()), n=len(pt), ceil_all=np.maximum(pt, 1 - pt).mean())
		for k, p in preds.items():
			rec[k + '_all'] = (p == x).mean()
			rec[k + '_forced'] = (p[forced] == x[forced]).mean() if forced.any() else np.nan
		rows.append(rec)


def agg(sel, key):
	v = [r[key] for r in rows if sel(r) and not np.isnan(r[key])]
	return 100 * np.mean(v)


for name, sel in [('all depths', lambda r: True), ('depth 8', lambda r: r['depth'] == 8)]:
	print(f"\n== {name}: states={sum(sel(r) for r in rows)}, FORCED share={100*sum(r['n_forced'] for r in rows if sel(r))/sum(r['n'] for r in rows if sel(r)):.1f}%")
	print(f"{'method':28s}{'all free vars':>16s}{'FORCED only':>14s}")
	print(f"{'Bayes ceiling':28s}{agg(sel,'ceil_all'):15.1f}%{100.0:13.1f}%")
	for k, lab in [('gat_in', 'GAT + in-tree'), ('gat_root', 'GAT + root (control)'), ('lp', 'LP rounding'), ('mlp_in', 'MLP+agg + in-tree')]:
		print(f"{lab:28s}{agg(sel,k+'_all'):15.1f}%{agg(sel,k+'_forced'):13.1f}%")
