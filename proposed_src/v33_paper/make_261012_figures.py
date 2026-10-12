"""Figures for docs/tex/261012_paper (single density p=0.3). Everything is matplotlib so the same PDF
(LaTeX) and PNG (Korean edition) come from one source. Run from this directory with the wsl_neuroMCTS
interpreter. Inputs: the p=0.3 result directories resolved by v33_tables.find, runs/v28 (Stage-1
by-depth accuracy, Stage-2 fine-tuning log), runs/v33/density_table.md. Panel titles sit below the panels.
The reduction, network and training diagrams are unchanged from fig/261001 and are copied."""
import glob, json, os, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, '.')
import v33_tables as T

OUT = '../../docs/tex/fig/261012'; PREV = '../../docs/tex/legacy/261001/fig'; R = '../../runs'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6})
C = dict(lp='#7f7f7f', lpfrac='#bcbd22', maxsd='#17becf', domwdeg='#8c564b', impact='#e377c2', lp_probe='#2ca02c', random='#c7c7c7',
         model='#1f77b4', pv='#d62728', oracle='#000000', bp='#9467bd', box='#f2f2f2', sound='#e8f3e4', learn='#e3ecfa', rl='#fbe6e4', edge='#333333')
NAME = dict(lp='LP rule', lpfrac='most-fractional', maxsd='maxSD', domwdeg='dom/wdeg', impact='impact', lp_probe='LP probing', random='uniform',
            model='Stage 1', pv='Stage 1+2', oracle='oracle marginals', bp='BP marginals')
MK = dict(lp='s', lpfrac='P', maxsd='X', domwdeg='h', impact='p', lp_probe='^', random='.', model='o', pv='D', oracle='*', bp='v')
SIZES = [('A_S0', 25), ('B_S0', 50), ('H_S0', 60), ('G70_S0', 70), ('G80_S0', 80)]


def save(fig, name):
	fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png', dpi=220, bbox_inches='tight'); plt.close(fig); print('saved', name)


def sub(ax, text, dy=-0.30):
	ax.text(0.5, dy, text, transform=ax.transAxes, ha='center', va='top', fontsize=7.6)


def box(ax, x, y, w, h, text, fc=C['box'], fs=6.8, bold=False, ec=C['edge'], lw=0.7):
	ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.03', fc=fc, ec=ec, lw=lw))
	t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.25)
	fig = ax.figure; fig.canvas.draw()
	while fs > 4.5:
		bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
		if bb.width <= 0.92 * w and bb.height <= 0.9 * h: break
		fs -= 0.25; t.set_fontsize(fs)
	return t


def arrow(ax, x0, y0, x1, y1, color=C['edge'], lw=0.8, ls='-'):
	ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def arm(tag, a):
	d = T.find(tag, a); return T.load(d) if d else None


def pairn(A, B, key='nodes'):
	# numeric paired comparison (median per-instance ratio and bootstrap CI), unlike T.pair which formats a string
	ks = [k for k in sorted(set(A) & set(B)) if A[k]['solved'] and B[k]['solved']]
	r = np.array([A[k][key] / max(B[k][key], 1e-9) for k in ks]); rng = np.random.default_rng(0)
	bs = [np.median(rng.choice(r, len(r))) for _ in range(1000)]
	return dict(median_ratio=float(np.median(r)), ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))))


def med(X, key='nodes'):
	s = [r[key] for r in X.values() if r['solved']]; return float(np.median(s)) if s else np.nan


# copies of the unchanged diagrams
for f in ['reduction', 'network', 'training']:
	for ext in ['pdf', 'png']: shutil.copy(f'{PREV}/{f}.{ext}', f'{OUT}/{f}.{ext}')

# ---------------- framework: one node of the guided DFS, no lattice stage ----------------
fig, ax = plt.subplots(figsize=(7.0, 2.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.43); ax.axis('off')
ax.text(0.30, 3.22, 'One node of the guided depth-first search', fontsize=7.4, va='center', fontweight='bold')
bw, gap, x0 = 1.45, 0.15, 0.28; xs = [x0 + k * (bw + gap) for k in range(6)]
labels = [('1  Reduce\n$A_K\\mathbf{y}=\\mathbf{b}-A_F\\mathbf{v}$', C['sound']), ('2  Bound check\nprune if $b^\\prime_i<0$\nor $b^\\prime_i>r_i$', C['sound']),
          ('3  Propagate\nfix a row when\n$b^\\prime_i\\in\\{0,\\,r_i\\}$', C['sound']), ('4  Warm-started LP\nprune if infeasible;\nread $\\mathbf{x}^{LP}$', C['sound']),
          ('5  Network\n$\\hat p_j$,  $g_j$', C['learn']), ('6  Branch\n$x_{j^*}=v$ first,\n$1-v$ on backtrack', '#ffffff')]
for k, (t, fc) in enumerate(labels):
	box(ax, xs[k], 1.70, bw, 0.95, t, fc=fc, fs=6.3)
	if k < 5: arrow(ax, xs[k] + bw, 2.175, xs[k + 1], 2.175)
ax.plot([xs[0], xs[3] + bw], [2.80, 2.80], color='#4a7a4a', lw=0.7); ax.text((xs[0] + xs[3] + bw) / 2, 2.86, 'sound deduction: the only steps that fix variables or prune', fontsize=6.0, color='#4a7a4a', ha='center', va='bottom')
ax.plot([xs[4], xs[5] + bw], [2.80, 2.80], color='#3a5a9a', lw=0.7); ax.text((xs[4] + xs[5] + bw) / 2, 2.86, 'learned: chooses, never fixes', fontsize=6.0, color='#3a5a9a', ha='center', va='bottom')
box(ax, 0.28, 0.45, 4.45, 0.80, 'Stage 1 (supervised) supplies the first value\n$v=\\mathbb{1}[\\hat p_{j^*}\\geq 1/2]$  from the marginal $\\hat p_j$', fc=C['learn'], fs=6.5)
box(ax, 5.05, 0.45, 4.65, 0.80, 'Stage 2 (cost-tuned) supplies the branching variable\n$j^*=\\arg\\max_j g_j$,   $g_j=\\log|\\hat p_j-1/2|+h_\\phi(\\mathbf{h}_j)$', fc=C['rl'], fs=6.5)
arrow(ax, xs[4] + 0.45, 1.70, 2.50, 1.25); arrow(ax, xs[4] + 1.00, 1.70, 7.35, 1.25)
ax.text(0.30, 0.15, 'A classical rule replaces step 5 by its own score; every rule in the paper shares steps 1--4, 6 and the budget.', fontsize=6.2, va='center', color='#444')
save(fig, 'framework')

# ---------------- RQ1: per-instance node curves and nodes vs time ----------------
rules = ['lp', 'lpfrac', 'maxsd', 'domwdeg', 'impact', 'lp_probe', 'random', 'model', 'pv']
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, (tag, n) in zip(axs, SIZES[2:]):
	for a in rules:
		X = arm(tag, a)
		if not X: continue
		v = sorted(r['nodes'] for r in X.values() if r['solved']); k = len(X)
		ax.plot(np.arange(len(v)) / k, v, color=C[a], lw=1.6 if a == 'pv' else 1.0, marker=MK[a] if a in ('model', 'pv') else None, ms=2, markevery=10, label=NAME[a])
	ax.set_yscale('log'); ax.set_xlabel('fraction of instances (sorted per rule)'); ax.tick_params(labelsize=6.5)
	sub(ax, f'{T.SETS[tag][0]}', dy=-0.26)
axs[0].set_ylabel('nodes to a solution'); axs[0].legend(frameon=False, fontsize=5.4, loc='upper left', ncol=2, handlelength=1.2, columnspacing=0.8)
fig.tight_layout(w_pad=1.2); save(fig, 'rq1_nodes')


# ---------------- RQ2: scaling with size ----------------
fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.4))
ns = [n for _, n in SIZES]
for a in ['lp', 'lpfrac', 'maxsd', 'domwdeg', 'lp_probe', 'random', 'model', 'pv']:
	nodes, secs = [], []
	for tag, n in SIZES:
		X = arm(tag, a); nodes.append(med(X) if X else np.nan); secs.append(med(X, 'sec') if X else np.nan)
	axs[0].plot(ns, nodes, color=C[a], marker=MK[a], ms=3, lw=1.6 if a == 'pv' else 0.9, label=NAME[a])
	axs[1].plot(ns, secs, color=C[a], marker=MK[a], ms=3, lw=1.6 if a == 'pv' else 0.9, label=NAME[a])
axs[0].set_yscale('log'); axs[0].set_ylabel('median nodes to a solution'); axs[1].set_yscale('log'); axs[1].set_ylabel('median time (s)')
axs[0].legend(frameon=False, fontsize=5.4, ncol=2, loc='upper left', handlelength=1.2, columnspacing=0.8)
for a_ in axs: a_.set_xticks(ns); a_.set_xlabel('variables $n$ (rows: 10, 18, 21, 24, 28)'); a_.tick_params(labelsize=6.5)
sub(axs[0], '(a) tree size', dy=-0.26); sub(axs[1], '(b) wall-clock', dy=-0.26)
fig.tight_layout(w_pad=1.2); save(fig, 'rq2_scaling')

# ---------------- RQ3: value x variable cross ablation ----------------
fig, ax = plt.subplots(figsize=(3.6, 2.5))
cross = [('lp', 'LP var\nLP val', C['lp']), ('lpvar_s1val', 'LP var\nS1 val', '#b0b0b0'), ('model', 'S1 var\nS1 val', C['model']), ('s1var_lpval', 'S1 var\nLP val', '#6baed6'), ('pv', 'S2 var\nS1 val', C['pv']), ('pvvar_lpval', 'S2 var\nLP val', '#f4a3a3')]
w = 0.26
for i, (tag, lab) in enumerate([('H_S0', '$21\\times60$'), ('G70_S0', '$24\\times70$'), ('G80_S0', '$28\\times80$')]):
	P = arm(tag, 'pv')
	for j, (a, _, col) in enumerate(cross):
		X = arm(tag, a)
		if not X or not P: continue
		c = pairn(X, P); x = j + (i - 1) * w
		ax.bar(x, c['median_ratio'], width=w, color=col, alpha=[0.55, 0.8, 1.0][i], yerr=[[c['median_ratio'] - c['ci'][0]], [c['ci'][1] - c['median_ratio']]], capsize=1.2, error_kw=dict(lw=0.5))
for i, lab in enumerate(['$21\\times60$', '$24\\times70$', '$28\\times80$']): ax.bar([np.nan], [np.nan], color='#777', alpha=[0.55, 0.8, 1.0][i], label=lab)
ax.axhline(1, color='k', lw=0.6, ls=':'); ax.set_yscale('log'); ax.set_xticks(range(len(cross))); ax.set_xticklabels([c[1] for c in cross], fontsize=5.8); ax.set_ylabel('nodes, arm / Stage 1+2'); ax.tick_params(axis='y', labelsize=6.5)
ax.legend(frameon=False, fontsize=5.8, loc='upper right', ncol=1, handlelength=1.0, labelspacing=0.2); ax.set_ylim(0.4, 400)
sub(ax, 'median per-instance node ratio to Stage 1+2 with 95% bootstrap CI', dy=-0.26)
save(fig, 'rq3')

# ---------------- Stage-1 accuracy by depth at p=0.3 ----------------
B = json.load(open(f'{R}/v28/s1_p03/v17_by_depth.json')); ds = sorted(int(k) for k in B)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.plot(ds, [100 * B[str(d)]['ceil'] for d in ds], color='k', ls='--', lw=1.0, label='Bayes ceiling')
ax.plot(ds, [100 * B[str(d)]['model'] for d in ds], color=C['model'], marker='o', ms=2.5, label='Stage 1 network')
ax.plot(ds, [100 * B[str(d)]['lp'] for d in ds], color=C['lp'], marker='s', ms=2.5, label='LP rounding')
ax.set_xlabel('depth of the in-tree state'); ax.set_ylabel('accuracy on free variables (%)'); ax.legend(frameon=False, fontsize=6); ax.tick_params(labelsize=6.5)
sub(ax, 'held-out $10\\times25$ in-tree states, $p=0.3$', dy=-0.22)
save(fig, 'prediction')

# ---------------- reduction identity + FORCED/FREE partition (p=0.3 numbers) ----------------
fig, ax = plt.subplots(figsize=(7.0, 2.7)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.86); ax.axis('off')
ax.text(0.15, 3.65, 'Conditioning on a partial assignment is the same as reducing the instance (Proposition 1)', fontsize=7.2, va='center', fontweight='bold')
box(ax, 0.15, 1.35, 2.5, 1.9, 'root instance\n$A x = b$\n$n$ variables\nsolution set $\\mathcal{S}$', fc='#ffffff')
arrow(ax, 2.65, 2.3, 3.35, 2.3); ax.text(3.0, 2.62, 'fix\n$x_F{=}v$', ha='center', va='bottom', fontsize=6, linespacing=1.1)
box(ax, 3.35, 1.35, 3.4, 1.9, 'node = reduced instance\n$A_K\\, y = b - A_F v$,  $|K|$ free variables\n\nsurviving solutions $\\mathcal{S}(F,v)$\n$p_j$ = fraction of $\\mathcal{S}(F,v)$ with $x_j{=}1$', fc='#ffffff')
arrow(ax, 6.75, 2.3, 7.35, 2.3)
box(ax, 7.35, 2.35, 2.5, 0.9, 'FORCED: $p_j \\in \\{0, 1\\}$\nwrong branch empties\nthe subtree', fc='#fde2e2')
box(ax, 7.35, 1.35, 2.5, 0.9, 'FREE: $0 < p_j < 1$\neither branch still\nreaches a solution', fc='#e3f2e1')
ax.text(0.15, 0.6, 'Only errors on FORCED variables cost a backtrack. Their share is 3% at the root, 41% at depth 5 and 73% at depth 8 ($10\\times25$ in-tree states, $p=0.3$),\n'
                   'so the network is trained on reduced instances (in-tree states); labels are exact because $\\mathcal{S}$ is enumerated.', fontsize=6.4, va='center', color='#333')
save(fig, 'reduction')
