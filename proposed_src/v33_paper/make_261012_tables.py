"""LaTeX tables for docs/tex/261012_paper from the p=0.3 runs (same sources and statistics as
v33_tables.py): rq1_main (three 100-instance sizes, all rules), small (10x25, 18x50), density,
rq3_cross, rq3_variants, rq4_scip, rq4_e2e. Writes docs/tex/tab/261008/<name>.tex; a table whose
inputs are incomplete is still written with the rows that exist. Run from this directory.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, '.')
import v33_tables as T

OUT = '../../docs/tex/tab/261012'; os.makedirs(OUT, exist_ok=True); R = T.R
X_ = '$\\times$'
HL = lambda x: f'\\textbf{{\\underline{{{x}}}}}'   # highlighted cell: bold + underline


def pr(A, B, key):
	ks = [k for k in sorted(set(A) & set(B)) if A[k]['solved'] and B[k]['solved']]
	if not ks: return None
	r = np.array([A[k][key] / max(B[k][key], 1e-9) for k in ks]); w = int((r > 1).sum()); nt = int((r != 1).sum())
	rng = np.random.default_rng(0); bs = [np.median(rng.choice(r, len(r))) for _ in range(1000)]
	return dict(m=float(np.median(r)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), w=w, nt=nt, p=float(binomtest(w, nt, 0.5).pvalue) if nt else 1.0)


def fp(p):
	if p >= 0.01: return f'{p:.2f}'
	e = int(np.floor(np.log10(p))); return f'10^{{{e}}}' if p / 10 ** e < 3 else f'{p / 10 ** e:.0f}\\cdot10^{{{e}}}'


def cell(c, bold_if_less=False):
	if c is None: return '---'
	s = f"{c['m']:.2f}{X_} [{c['lo']:.2f}, {c['hi']:.2f}] ${fp(c['p'])}$"
	return s


def arm(tag, a):
	d = T.find(tag, a); return T.load(d) if d else None


def stats(X, budget):
	s = [r for r in X.values() if r['solved']]
	nd = [r['nodes'] for r in s if r.get('nodes') is not None]
	return len(s), len(X), np.median(nd) if nd else np.nan, np.median([r['sec'] for r in s]) if s else np.nan, float(np.mean([r['sec'] if r['solved'] else 2 * budget for r in X.values()]))


def write(name, body): open(f'{OUT}/{name}.tex', 'w').write(body); print('wrote', name)


RULES = [('LP rule (most-integral)', 'lp'), ('most-fractional', 'lpfrac'), ('maxSD', 'maxsd'), ('dom/wdeg', 'domwdeg'), ('impact', 'impact'), ('LP probing', 'lp_probe'), ('uniform', 'random'),
         ('oracle marginals', 'oracle'), ('BP marginals (NSNet-based)', 'bp'), ('Stage 1', 'model'), ('Stage 1+2', 'pv')]


def block(tag, rules=RULES, ref='pv'):
	label, budget = T.SETS[tag]; P = arm(tag, ref); L = []
	for name, a in rules:
		X = arm(tag, a)
		if not X: continue
		sol, n, mn, ms, par = stats(X, budget)
		if a == ref: L.append(f"{HL(name)} & {sol}/{n} & {HL(f'{mn:,.0f}')} & {HL(f'{ms:.2f}')} & {par:.1f} & --- & --- \\\\"); continue
		L.append(f"{name} & {sol}/{n} & {mn:,.0f} & {ms:.2f} & {par:.1f} & {cell(pr(X, P, 'nodes'))} & {cell(pr(X, P, 'sec'))} \\\\")
	return label, budget, L


def main():
	# RQ1 main table: three 100-instance sizes
	L = ['\\begin{table*}[t]', '\\centering',
	     '\\caption{Branching rules in the same search loop, $100$ instances per size. Medians are over solved instances; the last two columns are the median per-instance ratio of the rule to Stage~1+2 on commonly solved instances, its $95\\%$ bootstrap interval, and the two-sided sign test $p$-value (ties excluded). A ratio above $1$ means the rule expands more nodes or takes longer than the learned rule. PAR-2 counts an unsolved instance at twice the budget.}',
	     '\\label{tab:rq1}', '\\footnotesize', '\\begin{tabular}{lrrrrll}', '\\toprule', 'rule & solved & med.\\ nodes & med.\\ s & PAR-2 s & nodes, rule / Stage 1+2 & time, rule / Stage 1+2 \\\\']
	for tag in ['H_S0', 'G70_S0', 'G80_S0']:
		label, budget, rows = block(tag); lab = label.replace('x', X_)
		L += ['\\midrule', f"\\multicolumn{{7}}{{l}}{{\\emph{{{lab}, budget {budget}\\,s}}}} \\\\"] + rows
	L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
	write('rq1_main', '\n'.join(L))
	# small sizes (appendix)
	L = ['\\begin{table}[t]', '\\centering', '\\caption{The two small sizes ($30$ instances each): same columns as Table~\\ref{tab:rq1}. Trees are tens of nodes and no rule separates from the others in time except through the cost of its score.}', '\\label{tab:small}', '\\footnotesize',
	     '\\fitc{\\begin{tabular}{lrrrrll}', '\\toprule', 'rule & solved & med.\\ nodes & med.\\ s & PAR-2 s & nodes, rule / Stage 1+2 & time, rule / Stage 1+2 \\\\']
	for tag in ['A_S0', 'B_S0']:
		label, budget, rows = block(tag); lab = label.replace('x', X_)
		L += ['\\midrule', f"\\multicolumn{{7}}{{l}}{{\\emph{{{lab}, budget {budget}\\,s}}}} \\\\"] + rows
	L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
	write('small', '\n'.join(L))
	# RQ3 cross ablation
	cross = [('LP var., LP value (LP rule)', 'lp'), ('LP var., S1 value', 'lpvar_s1val'), ('S1 var., S1 value (Stage 1)', 'model'), ('S1 var., LP value', 's1var_lpval'), ('S2 var., S1 value (Stage 1+2)', 'pv'), ('S2 var., LP value', 'pvvar_lpval')]
	L = ['\\begin{table*}[t]', '\\centering', '\\caption{Value $\\times$ variable cross ablation: each arm combines one variable-selection score with one first-value rule; everything else is identical. Columns per size: median nodes and the median per-instance node ratio to Stage~1+2 with bootstrap interval and sign-test $p$-value.}', '\\label{tab:cross}', '\\scriptsize',
	     '\\begin{tabular}{lrlrlrl}', '\\toprule', ' & \\multicolumn{2}{c}{$21\\times60$ (100)} & \\multicolumn{2}{c}{$24\\times70$ (100)} & \\multicolumn{2}{c}{$28\\times80$ (100)} \\\\', '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}', 'variable, value & nodes & ratio & nodes & ratio & nodes & ratio \\\\', '\\midrule']
	for name, a in cross:
		cells = []
		for tag in ['H_S0', 'G70_S0', 'G80_S0']:
			X = arm(tag, a); P = arm(tag, 'pv')
			if not X or not P: cells += ['---', '---']; continue
			mn = np.median([r['nodes'] for r in X.values() if r['solved']]); c = pr(X, P, 'nodes')
			cells += [f"{mn:,.0f}", '---' if a == 'pv' else cell(c)]
		L.append(f"{name} & " + ' & '.join(cells) + ' \\\\')
	L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
	write('rq3_cross', '\n'.join(L))
	# RQ3 training variants (dirs under runs/v33/<variant>/<tag>_<arm>[_fast])
	var = [('Stage 1 (reference)', 'main', 'model'), ('Stage 1, hard $0/1$ labels', 's1_hard', 'model'), ('Stage 1, training seed 1', 's1_seed1', 'model'), ('Stage 1, training seed 2', 's1_seed2', 'model'),
	       ('Stage 1, $2{,}771$ states (size ctrl.)', 's1_ctrl', 'model'), ('Stage 1 on $21\\times60$ states (M1)', 'M1', 'model'), ('Stage 1, NSNet architecture', 'nsnet', 'model'),
	       ('Stage 1+2 (reference)', 'main', 'pv'), ('Stage 2 from a random head', 'rl_random', 'pv'), ('Stage 2 from the LP rule', 'rl_lp', 'pv'), ('Stage 2, trunk unfrozen', 'rl_unfrozen', 'pv')]
	L = ['\\begin{table*}[t]', '\\centering', '\\caption{Training variants: median nodes and median per-instance node ratio to the reference Stage~1+2 rule, per size. Stage-1 variants are run with the confidence rule (no Stage 2); Stage-2 variants share the frozen reference trunk unless stated.}', '\\label{tab:variants}', '\\scriptsize',
	     '\\begin{tabular}{lrlrlrl}', '\\toprule', ' & \\multicolumn{2}{c}{$21\\times60$ (100)} & \\multicolumn{2}{c}{$24\\times70$ (100)} & \\multicolumn{2}{c}{$28\\times80$ (100)} \\\\', '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}', 'variant & nodes & ratio & nodes & ratio & nodes & ratio \\\\', '\\midrule']
	for name, sub, a in var:
		cells = []
		for tag in ['H_S0', 'G70_S0', 'G80_S0']:
			P = arm(tag, 'pv')
			if sub == 'main': X = arm(tag, a)
			elif sub == 'p05':
				d = {'H_S0': f'{R}/v28/p05/H_S0_pv', 'G70_S0': f'{R}/v32/d03_m05/G70_S0_pv_fast', 'G80_S0': f'{R}/v32/d03_m05/G80_S0_pv_fast'}[tag]; X = T.load(d) if os.path.isdir(d) else None
			else:
				X = None
				for suf in ('_fast', ''):
					d = f'{R}/v33/{sub}/{tag}_{a}{suf}'
					if os.path.isdir(d) and glob.glob(d + '/inst-*.json'): X = T.load(d); break
			if not X or not P: cells += ['---', '---']; continue
			mn = np.median([r['nodes'] for r in X.values() if r['solved']]); sol = sum(r['solved'] for r in X.values())
			cells += [f"{mn:,.0f}" + (f" ({sol}/{len(X)})" if sol < len(X) else ''), '---' if (sub == 'main' and a == 'pv') else cell(pr(X, P, 'nodes'))]
		L.append(f"{name} & " + ' & '.join(cells) + ' \\\\')
		if a == 'model' and name.startswith('Stage 1, NSNet'): L.append('\\midrule')
	L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
	write('rq3_variants', '\n'.join(L))
	# RQ4 SCIP plug-in
	L = ['\\begin{table*}[t]', '\\centering', '\\caption{Inside SCIP (presolve, cuts and primal heuristics off, depth-first node selection, first solution stops): SCIP\'s own rules against the learned rule as a branching plug-in, with conflict analysis off and on. Ratios are rule / plug-in as in Table~\\ref{tab:rq1}.}', '\\label{tab:scip}', '\\footnotesize',
	     '\\begin{tabular}{llrrrll}', '\\toprule', 'conflict & rule & solved & med.\\ nodes & med.\\ s & nodes, rule / plug-in & time, rule / plug-in \\\\']
	for tag, label in [('H', '$21\\times60$'), ('G70', '$24\\times70$'), ('G80', '$28\\times80$')]:
		for cf, cl in [(0, 'off'), (1, 'on')]:
			base = next((b for b in (f'{R}/v33/scip/SCIP_{tag}_S0_c{cf}p0h0', f'{R}/v28/scip/SCIP_{tag}_S0_c{cf}p0h0') if os.path.isdir(b + '_pv') and glob.glob(b + '_pv/inst-*.json')), None)
			if base is None: continue
			P = T.load(base + '_pv')
			L.append('\\midrule'); L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{label} ({len(P)})}}}} \\\\")
			for name, m in [('RPB (SCIP default)', 'scip_relpscost'), ('most-fractional (SCIP)', 'scip_mostinf'), ('inference (SCIP)', 'scip_inference'), ('Stage 1+2 plug-in', 'pv')]:
				d = f'{base}_{m}'
				if not os.path.isdir(d): continue
				X = T.load(d); sol, n, mn, ms, par = stats(X, 1200)
				if m == 'pv': L.append(f"{cl} & {HL(name)} & {sol}/{n} & {HL(f'{mn:,.0f}')} & {HL(f'{ms:.2f}')} & --- & --- \\\\")
				else: L.append(f"{cl} & {name} & {sol}/{n} & {mn:,.0f} & {ms:.2f} & {cell(pr(X, P, 'nodes'))} & {cell(pr(X, P, 'sec'))} \\\\")
	L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
	write('rq4_scip', '\n'.join(L))
	# RQ4 end to end
	L = ['\\begin{table}[t]', '\\centering', '\\caption{End to end, same instances and budgets, single-threaded: complete solvers with all defaults, lattice reduction with bounded enumeration, and the learned search. Median time over solved instances and PAR-2; node counts are not comparable across engines and are omitted.}', '\\label{tab:e2e}', '\\footnotesize',
	     '\\fitc{\\begin{tabular}{lrrr}', '\\toprule', 'method & solved & med.\\ s & PAR-2 s \\\\']
	for tag, label, budget, otag in [('H', '$21\\times60$ (100)', 600, 'H_S0'), ('G70', '$24\\times70$ (100)', 1200, 'G70_S0'), ('G80', '$28\\times80$ (100)', 1200, 'G80_S0')]:
		rows = []
		for name, m in [('SCIP (all defaults)', 'scip_default'), ('CP-SAT (1 worker)', 'cpsat'), ('Gurobi (1 thread)', 'gurobi'), ('BKZ-40 $\\times$ 30 + enumeration', 'lattice')]:
			if m == 'gurobi':
				f = {'H': f'{R}/v25/gurobi_v28lo_21x60_100.json', 'G70': f'{R}/v25/gurobi_v32d03_24x70_100.json', 'G80': f'{R}/v25/gurobi_v32d03_28x80_100.json'}[tag]
				if not os.path.exists(f): continue
				G = json.load(open(f)); X = {r.get('instance', r.get('inst')): dict(solved=bool(r['solved']), nodes=None, sec=r.get('time_sec', r.get('sec'))) for r in (G if isinstance(G, list) else G.get('results', []))}
			else:
				d = f'{R}/v33/e2e/E2E_{tag}_S0_{m}'
				if not os.path.isdir(d): continue
				X = T.load(d)
			if not X: continue
			sol, n, mn, ms, par = stats(X, budget); rows.append(f"{name} & {sol}/{n} & {ms:.2f} & {par:.1f} \\\\")
		P = arm(otag, 'pv')
		if P:
			sol, n, mn, ms, par = stats(P, budget); rows.append(f"{HL('Stage 1+2 (our loop)')} & {sol}/{n} & {HL(f'{ms:.2f}')} & {par:.1f} \\\\")
		if rows: L += ['\\midrule', f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\"] + rows
	L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
	write('rq4_e2e', '\n'.join(L))


if __name__ == '__main__':
	main()


def seeds_table():
	# timing replication: learned arms over three evaluation seeds (node counts are seed-invariant)
	L = ['\\begin{table}[t]', '\\centering', '\\caption{Timing replication of the learned arms over three evaluation seeds (seed $0$ is the one reported in the main tables): instances solved and median wall-clock. Node counts are identical across seeds because the search is deterministic given the relaxation vertex.}', '\\label{tab:seeds}', '\\footnotesize',
	     '\\fitc{\\begin{tabular}{llrrr}', '\\toprule', 'size & arm & seed 0 & seed 1 & seed 2 \\\\', '\\midrule']
	for tag, base in [('H_S0', 'H'), ('G70_S0', 'G70'), ('G80_S0', 'G80')]:
		for name, a in [('Stage 1', 'model'), ('Stage 1+2', 'pv')]:
			cells = []
			for k in range(3):
				d = T.find(tag, a) if k == 0 else next((p for p in (f'{R}/v33/main/{base}_S{k}_{a}_fast', f'{R}/v33/main/{base}_S{k}_{a}') if os.path.isdir(p) and glob.glob(p + '/inst-*.json')), None)
				if not d: cells.append('---'); continue
				X = T.load(d); s = [r for r in X.values() if r['solved']]
				cells.append(f"{len(s)}/{len(X)}, {np.median([r['sec'] for r in s]):.2f}\\,s")
			L.append(f"${T.SETS[tag][0].split(' ')[0].replace('x', chr(92) + 'times')}$ & {name} & " + ' & '.join(cells) + ' \\\\')
	L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
	write('seeds', '\n'.join(L))


if __name__ == '__main__':
	seeds_table()
