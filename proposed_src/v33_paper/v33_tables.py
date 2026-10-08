"""Paper tables at the single paper density p=0.3, organised by research question. Every directory
read here holds p=0.3 results only (v28 lo sets, v29 dense re-measurement, v32 100-instance sets,
v33 completion runs). Reads what exists, so it can be run while v33 is still going.

Per (set, arm): solved, median nodes, median sec, PAR-2, and the median per-instance ratio of
nodes and sec against the reference arm with a bootstrap 95% CI and a two-sided sign test.
Usage: v33_tables.py [--out ../../runs/v33/paper_tables.md]
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import binomtest

R = '../../runs'
SETS = {  # tag -> (label, budget)
	'A_S0': ('10x25 (30)', 60), 'B_S0': ('18x50 (30)', 300), 'H_S0': ('21x60 (100)', 600),
	'G70_S0': ('24x70 (100)', 1200), 'G80_S0': ('28x80 (100)', 1200)}
# where the p=0.3 runs of each arm live, per set (first existing directory wins)
LOC = {
	'A_S0': [f'{R}/v33/main'],
	'B_S0': [f'{R}/v33/main', f'{R}/v28/p03', f'{R}/v28/p05'],
	'H_S0': [f'{R}/v33/main', f'{R}/v29', f'{R}/v28/p05'],
	'G70_S0': [f'{R}/v33/main', f'{R}/v32/d03_m03', f'{R}/v32/d03'],
	'G80_S0': [f'{R}/v33/main', f'{R}/v32/d03_m03', f'{R}/v32/d03'],
}
CLASSICAL = [('LP rule (most-integral)', 'lp'), ('most-fractional', 'lpfrac'), ('maxSD', 'maxsd'), ('dom/wdeg', 'domwdeg'),
             ('impact', 'impact'), ('LP-probing', 'lp_probe'), ('uniform', 'random')]
LEARNED = [('Stage 1', 'model'), ('Stage 1+2', 'pv')]
REFSIG = [('oracle (exact marginals)', 'oracle'), ('BP marginals (NSNet-based)', 'bp')]
CROSS = [('LP var / Stage-1 val', 'lpvar_s1val'), ('Stage-1 var / LP val', 's1var_lpval'), ('Stage-2 var / LP val', 'pvvar_lpval')]
VARIANTS = [('Stage 1, hard labels', f'{R}/v33/s1_hard', 'model_fast'), ('Stage 1, seed 1', f'{R}/v33/s1_seed1', 'model_fast'), ('Stage 1, seed 2', f'{R}/v33/s1_seed2', 'model_fast'),
            ('Stage 1, 10x25 size-matched (2,771 states)', f'{R}/v33/s1_ctrl', 'model_fast'), ('Stage 1, 21x60-trained (M1)', f'{R}/v33/M1', 'model_fast'),
            ('Stage 1, NSNet-style network', f'{R}/v33/nsnet', 'model'),
            ('Stage 2 from random init', f'{R}/v33/rl_random', 'pv'), ('Stage 2 from LP init', f'{R}/v33/rl_lp', 'pv'), ('Stage 2, trunk unfrozen', f'{R}/v33/rl_unfrozen', 'pv')]


def load(d):
	out = {}
	for f in glob.glob(d + '/inst-*.json'):
		r = json.load(open(f)); out[r['instance']] = dict(solved=bool(r['solved']), nodes=r.get('nodes'), sec=r.get('sec'))
	return out


def find(tag, arm):
	for base in LOC[tag]:
		for suffix in ('_fast', ''):
			d = f'{base}/{tag}_{arm}{suffix}'
			if os.path.isdir(d) and glob.glob(d + '/inst-*.json'): return d
	return None


def pair(A, B, key, rng):
	ks = [k for k in sorted(set(A) & set(B)) if A[k]['solved'] and B[k]['solved'] and A[k][key] is not None]
	if not ks: return '—'
	r = np.array([A[k][key] / max(B[k][key], 1e-9) for k in ks]); w = int((r > 1).sum()); nt = int((r != 1).sum())
	bs = [np.median(rng.choice(r, len(r))) for _ in range(1000)]
	p = binomtest(w, nt, 0.5).pvalue if nt else 1.0
	return f"{np.median(r):.2f}x [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}] p={p:.2g}"


def row(name, X, budget, ref, rng):
	s = [r for r in X.values() if r['solved']]
	nd = f"{np.median([r['nodes'] for r in s]):,.0f}" if s else '—'
	sec = f"{np.median([r['sec'] for r in s]):.2f}" if s else '—'
	par = np.mean([r['sec'] if r['solved'] else 2 * budget for r in X.values()])
	rn = pair(X, ref, 'nodes', rng) if ref is not None and set(X) != set(ref) or (ref is not None and any(X[k] != ref[k] for k in X)) else '—'
	rs = pair(X, ref, 'sec', rng) if ref is not None and set(X) != set(ref) or (ref is not None and any(X[k] != ref[k] for k in X)) else '—'
	return f"| {name} | {len(s)}/{len(X)} | {nd} | {sec} | {par:.1f} | {rn} | {rs} |"


HEAD = "| rule | solved | median nodes | median sec | PAR-2 | nodes, rule / Stage 1+2 | sec, rule / Stage 1+2 |\n|---|---|---|---|---|---|---|"


def main():
	ap = argparse.ArgumentParser(); ap.add_argument('--out', default=f'{R}/v33/paper_tables.md'); a = ap.parse_args()
	rng = np.random.default_rng(0); L = ['# Paper tables, density 0.3 only', '', 'Ratios are median per-instance ratios on commonly solved instances, bootstrap 95% CI, two-sided sign test (ties excluded).']
	for tag, (label, budget) in SETS.items():
		pvd = find(tag, 'pv'); ref = load(pvd) if pvd else None
		L += ['', f'## RQ1/RQ2 — {label}, budget {budget}s', '', HEAD]
		for name, arm in CLASSICAL + LEARNED + REFSIG:
			d = find(tag, arm)
			if d: L.append(row(name, load(d), budget, ref, rng))
		if tag in ('H_S0', 'G70_S0', 'G80_S0'):
			L += ['', f'### RQ3 — value x variable cross ablation, {label}', '', HEAD]
			for name, arm in CROSS:
				d = find(tag, arm)
				if d: L.append(row(name, load(d), budget, ref, rng))
			L += ['', f'### RQ3 — training variants, {label}', '', HEAD]
			for name, base, arm in VARIANTS:
				d = f'{base}/{tag}_{arm}'
				if os.path.isdir(d) and glob.glob(d + '/inst-*.json'): L.append(row(name, load(d), budget, ref, rng))
	# RQ4: SCIP branching + end-to-end
	for tag, label, budget in [('H', '21x60 (100)', 600), ('G70', '24x70 (100)', 1200), ('G80', '28x80 (100)', 1200)]:
		for cfg, cl in [('c0p0h0', 'conflict analysis off'), ('c1p0h0', 'conflict analysis on')]:
			rows = []
			for name, rule in [('RPB (SCIP default)', 'scip_relpscost'), ('most-fractional (SCIP)', 'scip_mostinf'), ('inference (SCIP)', 'scip_inference'), ('Stage 1+2 plugin', 'pv')]:
				for base in (f'{R}/v33/scip', f'{R}/v28/scip'):
					d = f'{base}/SCIP_{tag}_S0_{cfg}_{rule}'
					if os.path.isdir(d) and glob.glob(d + '/inst-*.json'): rows.append((name, load(d))); break
			if rows:
				ref = dict(rows).get('Stage 1+2 plugin')
				L += ['', f'## RQ4 — inside SCIP, {label}, {cl}', '', HEAD.replace('Stage 1+2', 'plugin')]
				for name, X in rows: L.append(row(name, X, budget, ref, rng))
		rows = []
		for name, m in [('SCIP (all defaults)', 'scip_default'), ('CP-SAT (1 worker)', 'cpsat'), ('BKZ-40 x30 + enumeration', 'lattice')]:
			d = f'{R}/v33/e2e/E2E_{tag}_S0_{m}'
			if os.path.isdir(d) and glob.glob(d + '/inst-*.json'): rows.append((name, load(d)))
		if rows:
			ours = find(tag + '_S0', 'pv'); ref = load(ours) if ours else None
			L += ['', f'## RQ4 — end-to-end solvers, {label} (ours = Stage 1+2 in our loop, dense path)', '', HEAD.replace('rule', 'solver')]
			if ref: L.append(row('Stage 1+2 (ours)', ref, budget, None, rng))
			for name, X in rows: L.append(row(name, X, budget, ref, rng))
	open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
	main()
