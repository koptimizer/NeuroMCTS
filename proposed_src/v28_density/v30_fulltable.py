"""One comparison table per evaluation set: classical branching rules, learned branching (ours),
learning-free reference signals, heuristics (lattice, local search) and complete solvers.

Node counts are comparable only within one engine, so the engine is a column and the table is
grouped by it. Times are wall-clock on the same machine; PAR-2 counts an unsolved instance at twice
the budget. Reads whatever directories exist, so it can be run while later cycles are still going.
Usage: v30_fulltable.py [--out ../../runs/full_table.md]
"""
import argparse, glob, json, os
import numpy as np

B = '../../runs'


def load(d):
	fs = glob.glob(d + '/inst-*.json') or glob.glob(d + '/inst-*_r.json')
	out = {}
	for f in fs:
		r = json.load(open(f))
		k = r['instance'].replace('_r.json', '').replace('.json', '')
		out[k] = dict(solved=bool(r['solved']), nodes=r.get('nodes'), sec=r.get('sec', r.get('total_sec')))
	return out


def row(name, engine, d, budget, ref=None):
	X = load(d)
	if not X: return None
	s = [r for r in X.values() if r['solved']]
	nd = f"{np.median([r['nodes'] for r in s]):,.0f}" if s and s[0]['nodes'] is not None else '—'
	sec = f"{np.median([r['sec'] for r in s]):.2f}" if s else '—'
	par = np.mean([r['sec'] if r['solved'] else 2 * budget for r in X.values()])
	rr = '—'
	if ref:
		ks = [k for k in set(X) & set(ref) if X[k]['solved'] and ref[k]['solved'] and X[k]['nodes'] and ref[k]['nodes']]
		if ks: rr = f"{np.median([X[k]['nodes'] / ref[k]['nodes'] for k in ks]):.2f}x"
	return f"| {name} | {engine} | {len(s)}/{len(X)} | {nd} | {sec} | {par:.1f} | {rr} |"


SETS = {
	'21x60 density 0.5 (100)': (600, 'H_S0', 'G70' if 0 else '', [
		('classical', [('LP rule (most-integral)', 'ours', f'{B}/v25/H_S0_lp'), ('most-fractional', 'ours', f'{B}/v25/H_S0_lpfrac'),
		               ('maxSD', 'ours', f'{B}/v25/H_S0_maxsd'), ('dom/wdeg', 'ours', f'{B}/v25/H_S0_domwdeg'),
		               ('impact', 'ours', f'{B}/v25/H_S0_impact'), ('LP-probing', 'ours', f'{B}/v25/H_S0_lp_probe'),
		               ('uniform', 'ours', f'{B}/v25/H_S0_random')]),
		('learned (ours)', [('Stage 1', 'ours', f'{B}/v25/H_S0_model'), ('Stage 1+2', 'ours', f'{B}/v25/H_S0_pv_fast')]),
		('reference signal', [('BP marginals (NSNet-based)', 'ours', f'{B}/v25/H_S0_bp')]),
		('SCIP branching', [('RPB (SCIP default)', 'SCIP', f'{B}/v26/SCIP_H_S0_c1p0h0_scip_relpscost'),
		                    ('most-fractional', 'SCIP', f'{B}/v26/SCIP_H_S0_c1p0h0_scip_mostinf'),
		                    ('inference', 'SCIP', f'{B}/v26/SCIP_H_S0_c1p0h0_scip_inference'),
		                    ('Stage 1+2 plugin', 'SCIP', f'{B}/v26/SCIP_H_S0_c1p0h0_pv')]),
		('complete solver', [('SCIP (all defaults)', 'SCIP', f'{B}/v26/E2E_H_S0_scip_default'), ('CP-SAT (1 worker)', 'CP-SAT', f'{B}/v26/E2E_H_S0_cpsat')]),
		('heuristic', [('BKZ-40 x30 + enumeration', 'fpylll', f'{B}/v26/E2E_H_S0_lattice')]),
	]),
	'24x70 density 0.5': (1200, 'G70_S0', '', [
		('classical', [('LP rule (most-integral)', 'ours', f'{B}/v24/G70_M0_S0_ahloff_lp'), ('most-fractional', 'ours', f'{B}/v25/G70_S0_lpfrac'),
		               ('maxSD', 'ours', f'{B}/v25/G70_S0_maxsd'), ('dom/wdeg', 'ours', f'{B}/v25/G70_S0_domwdeg'),
		               ('LP-probing', 'ours', f'{B}/v25/G70_S0_lp_probe'), ('uniform', 'ours', f'{B}/v25/G70_S0_random')]),
		('learned (ours)', [('Stage 1', 'ours', f'{B}/v25/G70_S0_model_fast'), ('Stage 1+2', 'ours', f'{B}/v25/G70_S0_pv_fast')]),
		('SCIP branching', [('RPB (SCIP default)', 'SCIP', f'{B}/v26/SCIP_G70_S0_c1p0h0_scip_relpscost'),
		                    ('most-fractional', 'SCIP', f'{B}/v26/SCIP_G70_S0_c1p0h0_scip_mostinf'),
		                    ('Stage 1+2 plugin', 'SCIP', f'{B}/v26/SCIP_G70_S0_c1p0h0_pv')]),
		('complete solver', [('SCIP (all defaults)', 'SCIP', f'{B}/v26/E2E_G70_S0_scip_default'), ('CP-SAT (1 worker)', 'CP-SAT', f'{B}/v26/E2E_G70_S0_cpsat')]),
		('heuristic', [('BKZ-40 x30 + enumeration', 'fpylll', f'{B}/v26/E2E_G70_S0_lattice')]),
	]),
	'21x60 density 0.3 (100)': (600, 'H_S0', '', [
		('classical', [('LP rule (most-integral)', 'ours', f'{B}/v28/p05/H_S0_lp'), ('most-fractional', 'ours', f'{B}/v28/p05/H_S0_lpfrac'),
		               ('maxSD', 'ours', f'{B}/v28/p05/H_S0_maxsd'), ('dom/wdeg', 'ours', f'{B}/v28/p05/H_S0_domwdeg'),
		               ('LP-probing', 'ours', f'{B}/v28/p05/H_S0_lp_probe'), ('uniform', 'ours', f'{B}/v28/p05/H_S0_random')]),
		('learned (ours)', [('Stage 1 (p=0.3)', 'ours', f'{B}/v29/H_S0_model_fast'), ('Stage 1+2 (p=0.3)', 'ours', f'{B}/v29/H_S0_pv_fast'),
		                    ('Stage 1+2 amortised k=2', 'ours', f'{B}/v29/H_S0_pv_k2_fast'), ('Stage 1+2 amortised k=4', 'ours', f'{B}/v29/H_S0_pv_k4_fast'),
		                    ('Stage 1+2 gated 50%', 'ours', f'{B}/v29/H_S0_pv_gate50_fast'), ('Stage 1+2 (p=0.5 model)', 'ours', f'{B}/v28/p05/H_S0_pv')]),
		('SCIP branching', [('RPB (SCIP default)', 'SCIP', f'{B}/v28/scip/SCIP_H_S0_c1p0h0_scip_relpscost'),
		                    ('most-fractional', 'SCIP', f'{B}/v28/scip/SCIP_H_S0_c1p0h0_scip_mostinf'),
		                    ('Stage 1+2 plugin', 'SCIP', f'{B}/v28/scip/SCIP_H_S0_c1p0h0_pv')]),
	]),
	'24x70 density 0.3': (1200, 'G70_S0', '', [
		('classical', [('LP rule (most-integral)', 'ours', f'{B}/v28/p05/G70_S0_lp'), ('most-fractional', 'ours', f'{B}/v28/p05/G70_S0_lpfrac'),
		               ('maxSD', 'ours', f'{B}/v28/p05/G70_S0_maxsd'), ('dom/wdeg', 'ours', f'{B}/v28/p05/G70_S0_domwdeg'),
		               ('LP-probing', 'ours', f'{B}/v28/p05/G70_S0_lp_probe'), ('uniform', 'ours', f'{B}/v28/p05/G70_S0_random')]),
		('learned (ours)', [('Stage 1 (p=0.3)', 'ours', f'{B}/v29/G70_S0_model_fast'), ('Stage 1+2 (p=0.3)', 'ours', f'{B}/v29/G70_S0_pv_fast'),
		                    ('Stage 1+2 amortised k=2', 'ours', f'{B}/v29/G70_S0_pv_k2_fast'), ('Stage 1+2 amortised k=4', 'ours', f'{B}/v29/G70_S0_pv_k4_fast'),
		                    ('Stage 1+2 gated 50%', 'ours', f'{B}/v29/G70_S0_pv_gate50_fast')]),
		('SCIP branching', [('RPB (SCIP default)', 'SCIP', f'{B}/v28/scip/SCIP_G70_S0_c1p0h0_scip_relpscost'),
		                    ('Stage 1+2 plugin', 'SCIP', f'{B}/v28/scip/SCIP_G70_S0_c1p0h0_pv')]),
	]),
	'28x80 density 0.3': (1200, 'G80_S0', '', [
		('classical', [('most-fractional', 'ours', f'{B}/v28/p05/G80_S0_lpfrac'), ('maxSD', 'ours', f'{B}/v28/p05/G80_S0_maxsd'),
		               ('dom/wdeg', 'ours', f'{B}/v28/p05/G80_S0_domwdeg'), ('LP-probing', 'ours', f'{B}/v28/p05/G80_S0_lp_probe'),
		               ('uniform', 'ours', f'{B}/v28/p05/G80_S0_random')]),
		('learned (ours)', [('Stage 1 (p=0.3)', 'ours', f'{B}/v29/G80_S0_model_fast'), ('Stage 1+2 (p=0.3)', 'ours', f'{B}/v29/G80_S0_pv_fast'),
		                    ('Stage 1+2 amortised k=2', 'ours', f'{B}/v29/G80_S0_pv_k2_fast'), ('Stage 1+2 amortised k=4', 'ours', f'{B}/v29/G80_S0_pv_k4_fast'),
		                    ('Stage 1+2 gated 50%', 'ours', f'{B}/v29/G80_S0_pv_gate50_fast'), ('Stage 1+2 (p=0.5 model)', 'ours', f'{B}/v28/p05/G80_S0_pv')]),
		('SCIP branching', [('RPB (SCIP default)', 'SCIP', f'{B}/v28/scip/SCIP_G80_S0_c1p0h0_scip_relpscost'),
		                    ('Stage 1+2 plugin', 'SCIP', f'{B}/v28/scip/SCIP_G80_S0_c1p0h0_pv')]),
	]),
}


def main():
	ap = argparse.ArgumentParser(); ap.add_argument('--out', default=f'{B}/full_table.md'); a = ap.parse_args()
	L = ['# Full comparison: classical branching, heuristics, complete solvers and the learned rules',
	     '',
	     'Node counts are comparable only inside one engine (the engine column says which).',
	     'PAR-2 counts an instance unsolved within the budget as twice the budget.',
	     'The last column is the median per-instance node ratio against our best learned arm in the same engine.']
	for label, (budget, _tag, _x, groups) in SETS.items():
		ours = None
		for g, items in groups:
			if g.startswith('learned'):
				cand = [load(d) for _, _, d in items]
				cand = [c for c in cand if c]
				if cand:
					ours = min(cand, key=lambda X: np.median([r['nodes'] for r in X.values() if r['solved'] and r['nodes']] or [np.inf]))
		L += ['', f'## {label}  (budget {budget}s)', '', '| rule / method | engine | solved | median nodes | median sec | PAR-2 s | nodes vs our best |', '|---|---|---|---|---|---|---|']
		for g, items in groups:
			L.append(f'| **{g}** | | | | | | |')
			for name, eng, d in items:
				r = row(name, eng, d, budget, ours)
				if r: L.append(r)
	open(a.out, 'w').write('\n'.join(L) + '\n')
	print('\n'.join(L))


if __name__ == '__main__':
	main()
