"""Aggregate runs/v23 (warm-started loop, 3 seeds) into the paper's search, transfer and cascade tables.
Per condition and AHL setting: solve rate mean±sd over seeds, per-instance median time, median lp/model
time ratio, sign-test wins, median nodes. Output: runs/v23/aggregate.json and a text table on stdout."""
import json, glob, os, sys
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../runs/v23')
CONDS = ['A_n25to25', 'B_n25to50', 'R_n50to50', 'C_n25to60', 'D_n50to60', 'Crand']
SEEDS = [0, 1, 2]

def load(d):
	return {os.path.basename(f): json.load(open(f)) for f in glob.glob(d + '/inst-*_r.json')}

def arm_stats(R):
	ks = sorted(R)
	t = np.array([R[k]['total_sec'] for k in ks]); nd = np.array([R[k]['nodes'] for k in ks])
	solved = np.array([R[k]['solved'] for k in ks]); by_ahl = sum(R[k]['solved_by'] == 'ahl' for k in ks)
	return dict(n=len(ks), solved=int(solved.sum()), by_ahl=int(by_ahl), med_t=float(np.median(t)),
	            mean_t=float(t.mean()), med_nodes=float(np.median(nd)), keys=ks, t=t, nodes=nd, solved_v=solved)

def paired(lp, mo):
	ks = sorted(set(lp) & set(mo))
	tl = np.array([lp[k]['total_sec'] for k in ks]); tm = np.array([mo[k]['total_sec'] for k in ks])
	nl = np.array([lp[k]['nodes'] for k in ks]); nm = np.array([mo[k]['nodes'] for k in ks])
	w = int((tm < tl).sum())
	return dict(ratio_med=float(np.median(tl / tm)), wins=w, n=len(ks), p_sign=float(binomtest(w, len(ks), 0.5).pvalue),
	            node_ratio=float(np.median(nl) / max(np.median(nm), 1)))

out = {}
for cond in CONDS:
	for ahl in ['off', 'on']:
		arms = {}
		for arm in ['lp', 'model', 'random']:
			per = []
			for s in SEEDS:
				d = f'{ROOT}/{cond}_S{s}_ahl{ahl}_{arm}'
				if os.path.isdir(d) and glob.glob(d + '/inst-*_r.json'): per.append((s, load(d)))
			if not per: continue
			st = [arm_stats(R) for _, R in per]
			arms[arm] = dict(seeds=[s for s, _ in per],
			                 rate=[x['solved'] / x['n'] for x in st], by_ahl=[x['by_ahl'] for x in st],
			                 med_t=[x['med_t'] for x in st], mean_t=[x['mean_t'] for x in st],
			                 med_nodes=[x['med_nodes'] for x in st], raw={s: R for s, R in per})
		if not arms: continue
		row = {a: {k: v for k, v in arms[a].items() if k != 'raw'} for a in arms}
		if 'lp' in arms and 'model' in arms:
			row['paired'] = [paired(arms['lp']['raw'][s], arms['model']['raw'][s]) for s in arms['lp']['seeds'] if s in arms['model']['raw']]
		out[f'{cond}_ahl{ahl}'] = row

def ms(v): return f'{100 * np.mean(v):.1f}±{100 * np.std(v):.1f}'
print(f"{'condition':22s} {'arm':7s} {'solve%':12s} {'med t (s)':>22s} {'med nodes':>22s}  paired: ratio  wins  p  node-ratio")
for key, row in out.items():
	for a in ['random', 'lp', 'model']:
		if a not in row: continue
		r = row[a]
		print(f"{key:22s} {a:7s} {ms(r['rate']):12s} {' / '.join(f'{x:.2f}' for x in r['med_t']):>22s} {' / '.join(f'{x:.0f}' for x in r['med_nodes']):>22s}", end='')
		if a == 'model' and 'paired' in row:
			print('  ' + ' | '.join(f"{p['ratio_med']:.2f}x {p['wins']}/{p['n']} p={p['p_sign']:.3f} nodes {p['node_ratio']:.1f}x" for p in row['paired']), end='')
		print()
json.dump(out, open(f'{ROOT}/aggregate.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('saved', f'{ROOT}/aggregate.json')
