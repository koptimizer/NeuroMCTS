"""Per-depth prediction curves on the 10x25 in-tree test states (260930 paper, Fig. prediction).

For each depth 0..12: Bayes ceiling, accuracy of GAT in-tree (runs/v22/model_n25), GAT root control
(runs/revision/ctrl_root), LP rounding and an MLP with constraint aggregates trained in-tree on the
full train file, on all free variables and on FORCED variables only (ground truth = reference solution x),
plus the FORCED share. Run from proposed_src/v23_ablation. Output: runs/revision/depth_curves.json
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
		rec = dict(depth=int(r['depth']), n_forced=int(forced.sum()), n=int(len(pt)), n_sol=int(r.get('n_surviving', -1)),
		           ceil_all=float(np.maximum(pt, 1 - pt).mean()))
		for k, p in preds.items():
			rec[k + '_all'] = float((p == x).mean())
			rec[k + '_forced'] = float((p[forced] == x[forced]).mean()) if forced.any() else None
		rows.append(rec)
out = {}
for d in sorted({r['depth'] for r in rows}):
	sel = [r for r in rows if r['depth'] == d]
	o = dict(states=len(sel), forced_share=float(sum(r['n_forced'] for r in sel) / sum(r['n'] for r in sel)),
	         n_sol_median=float(np.median([r['n_sol'] for r in sel])), ceil_all=float(np.mean([r['ceil_all'] for r in sel])))
	for k in ['gat_in', 'gat_root', 'lp', 'mlp_in']:
		o[k + '_all'] = float(np.mean([r[k + '_all'] for r in sel]))
		v = [r[k + '_forced'] for r in sel if r[k + '_forced'] is not None]
		o[k + '_forced'] = float(np.mean(v)) if v else None
	out[str(d)] = o
	print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
json.dump(out, open('../../runs/revision/depth_curves.json', 'w'), indent=1)
print('saved runs/revision/depth_curves.json')
