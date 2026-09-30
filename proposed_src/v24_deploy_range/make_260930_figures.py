"""Figures for docs/tex/260930_paper (EN) and its Korean edition. Everything is matplotlib so the
same PDF (LaTeX) and PNG (reportlab) come from one source. Run from this directory with the
wsl_neuroMCTS interpreter. Inputs: runs/v24 (search results, fine-tuning logs, where_M012, r0),
runs/revision (depth_curves, lattice_mech, lattice_only); the summary tables are transcribed from
docs/version.md where a figure only restates a table."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = '../../docs/tex/fig/260930'; R = '../../runs/v24'; RV = '../../runs/revision'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6})
C = dict(lp='#7f7f7f', m0='#1f77b4', ft='#d62728', ft2='#ff9896', mlp='#2ca02c', root='#9467bd', cpsat='#000000', scip='#8c564b',
         box='#f2f2f2', learn='#e8f0fb', rl='#fbe9e7', edge='#333333')


def save(fig, name):
	fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png', dpi=220, bbox_inches='tight'); plt.close(fig)
	print('saved', name)


def box(ax, x, y, w, h, text, fc=C['box'], fs=6.8, bold=False, ec=C['edge'], lw=0.7):
	"""Rounded box with text that is shrunk until it fits inside the box (never overflows)."""
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


def canvas(w, h, xmax=10, ymax=None):
	fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xmax); ax.set_ylim(0, ymax or xmax * h / w); ax.axis('off'); return fig, ax


# ---------------- Fig: framework (the system as evaluated: lattice stage + guided DFS) ----------------
fig, ax = canvas(7.0, 3.6)          # y range 0..5.14
ax.text(0.15, 5.02, 'Deployed system: a lattice stage, then a guided depth-first search', fontsize=7.4, va='center', fontweight='bold')
box(ax, 0.15, 3.85, 1.35, 0.75, 'instance\n$(A, b)$', fc='#ffffff', fs=7)
arrow(ax, 1.50, 4.22, 1.85, 4.22)
box(ax, 1.85, 3.85, 2.75, 0.75, 'AHL / BKZ lattice reduction\nup to 10 random column permutations\nsound: solution verified before return', fc=C['box'])
arrow(ax, 4.60, 4.35, 5.55, 4.60); ax.text(5.05, 4.60, 'solved', fontsize=6.2, color='#2a7', ha='center', va='bottom')
box(ax, 5.55, 4.32, 1.6, 0.55, 'return solution', fc='#eaf6ea', fs=6.8)
arrow(ax, 4.60, 4.05, 5.55, 3.90); ax.text(5.05, 3.86, 'residual', fontsize=6.2, color='#a55', ha='center', va='top')
box(ax, 5.55, 3.42, 1.6, 0.55, 'guided DFS', fc=C['box'], bold=True, fs=7.4)
ax.text(7.30, 4.10, r'lattice closes 100% at $n=25$,' '\n' r'80-93% at $n=50$, 33-53% at $n=60$,' '\n' r'7-13% at $n=70$, 0-3% at $n=80$', fontsize=6.2, va='center', color='#444')
ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 3.02, boxstyle='round,pad=0.02,rounding_size=0.04', fc='#fbfbfb', ec='#9a9a9a', lw=0.6, ls='--'))
arrow(ax, 6.10, 3.42, 5.60, 3.22, ls='--', color='#9a9a9a')
ax.text(0.30, 2.95, 'One node of the guided DFS (the only place learning acts)', fontsize=7.2, va='center', fontweight='bold')
seq = [(r'reduced instance' '\n' r'$A_K y = b - A_F v$' '\n' r'(Proposition 1)', '#ffffff'),
       (r'bound check' '\n' r'$b^\prime_i<0$ or $b^\prime_i>r_i$' '\n' r'$\Rightarrow$ prune', '#ffffff'),
       (r'propagation' '\n' r'row $i$: $b^\prime_i$ of $r_i$ must be 1' '\n' r'$b^\prime_i=0$ or $b^\prime_i=r_i \Rightarrow$ fix', '#eef4ea'),
       (r'warm-started LP' '\n' r'0.08 ms' '\n' r'infeasible $\Rightarrow$ prune', '#eef4ea'),
       (r'MarginalNet (frozen)' '\n' r'1.88 ms, $\hat p_j$', C['learn'])]
bw, gap, x = 1.78, 0.18, 0.30
for k, (t, fc) in enumerate(seq):
	box(ax, x, 1.50, bw, 1.18, t, fc=fc, fs=6.4)
	if k < len(seq) - 1: arrow(ax, x + bw, 2.09, x + bw + gap, 2.09)
	x += bw + gap
ax.plot([4.06, 7.68], [1.40, 1.40], color='#4a7a4a', lw=0.5, ls=':')
ax.text(5.87, 1.27, 'sound deduction (only these fix variables)', fontsize=6.0, color='#4a7a4a', ha='center')
arrow(ax, 8.30, 1.50, 8.30, 1.12, color=C['edge'])
box(ax, 4.55, 0.35, 2.05, 0.75, r'Stage 1: first value' '\n' r'$v=\mathbb{1}[\hat p_{j^*}\geq 1/2]$', fc=C['learn'], fs=6.6)
box(ax, 6.80, 0.35, 2.90, 0.75, r'Stage 2: branching variable' '\n' r'$j^*=\arg\max_j g_j$,   $g_j=\log|\hat p_j-1/2|+h_\phi(\mathbf{h}_j)$', fc=C['rl'], fs=6.4)
box(ax, 0.30, 0.35, 4.00, 0.75, r'branch $x_{j^*}=v$ first, $1-v$ on backtrack' '\n' r'the network never fixes a variable:' '\n' r'a wrong prediction costs backtracking only', fc='#ffffff', fs=6.4)
arrow(ax, 4.55, 0.72, 4.35, 0.72); arrow(ax, 6.80, 0.60, 6.62, 0.60)
save(fig, 'framework')

# ---------------- Fig: reduction identity + FORCED/OPEN partition ----------------
fig, ax = canvas(7.0, 2.7)          # y range 0..3.86
ax.text(0.15, 3.65, 'Conditioning on a partial assignment is the same as reducing the instance (Proposition 1)', fontsize=7.2, va='center', fontweight='bold')
box(ax, 0.15, 1.35, 2.5, 1.9, 'root instance\n$A x = b$\n$n$ variables\nsolution set $\\mathcal{S}$', fc='#ffffff')
arrow(ax, 2.65, 2.3, 3.35, 2.3); ax.text(3.0, 2.62, 'fix\n$x_F{=}v$', ha='center', va='bottom', fontsize=6, linespacing=1.1)
box(ax, 3.35, 1.35, 3.4, 1.9, 'node = reduced instance\n$A_K\\, y = b - A_F v$,  $|K|$ free variables\n\nsurviving solutions $\\mathcal{S}(F,v)$\n$p_j$ = fraction of $\\mathcal{S}(F,v)$ with $x_j{=}1$', fc='#ffffff')
arrow(ax, 6.75, 2.3, 7.35, 2.3)
box(ax, 7.35, 2.35, 2.5, 0.9, 'FORCED: $p_j \\in \\{0, 1\\}$\nwrong branch empties\nthe subtree', fc='#fde2e2')
box(ax, 7.35, 1.35, 2.5, 0.9, 'OPEN: $0 < p_j < 1$\neither branch still\nreaches a solution', fc='#e3f2e1')
ax.text(0.15, 0.6, 'Only errors on FORCED variables cost a backtrack. Their share is 7.5% at the root and 88% at depth 8 ($10\\times25$ in-tree states),\n'
                   'so the network is trained on reduced instances (in-tree states); labels are exact because $\\mathcal{S}$ is enumerated.', fontsize=6.4, va='center', color='#333')
save(fig, 'reduction')

# ---------------- Fig: two-stage training ----------------
fig, ax = canvas(7.0, 3.1)          # y range 0..4.43
def row(y0, title, fc, ec, items, last_bold=True):
	ax.add_patch(FancyBboxPatch((0.15, y0), 9.7, 1.95, boxstyle='round,pad=0.02,rounding_size=0.04', fc=fc, ec=ec, lw=0.6))
	ax.text(0.3, y0 + 1.75, title, fontsize=7.4, fontweight='bold', va='center')
	bw, gap, x = 2.1, 0.35, 0.35
	for k, t in enumerate(items):
		box(ax, x, y0 + 0.2, bw, 1.3, t, fc='#ffffff', bold=(last_bold and k == len(items) - 1), fs=6.5)
		if k < len(items) - 1: arrow(ax, x + bw, y0 + 0.85, x + bw + gap, y0 + 0.85)
		x += bw + gap
row(2.4, 'Stage 1 (supervised): which value should a variable take?', C['learn'], '#8899bb',
    ['small instances ($10\\times25$)\nenumerate $\\mathcal{S}$ exactly', 'in-tree states\nprefix of a real solution\nfixed, depth 0--12', 'exact conditional\nmarginals $p_j$\n(soft labels)', 'train MarginalNet $f_\\theta$\ncross-entropy to $p_j$'])
row(0.15, 'Stage 2 (cost-aware fine-tuning): which variable is cheapest to branch on?', C['rl'], '#cc8888',
    ['deployment-size states\n($21\\times60$, 45--60 free)\npolicy starts at the\nStage-1 rule', 'roll out the sampled\npolicy $\\pi \\propto e^{g_j/\\tau}$\nto a solution', 'exact cost per decision\n$c(u)$ = nodes in its subtree\n(from the search itself)', 'actor-critic update\n$\\alpha = V_\\psi - \\log c$\ntrains $h_\\phi, V_\\psi$ only'])
arrow(ax, 8.75, 2.6, 8.75, 1.65, ls='--', color='#c55'); ax.text(8.85, 2.12, '$f_\\theta$ frozen', fontsize=6.3, color='#c55', va='center')
save(fig, 'training')

# ---------------- Fig: prediction against the exact posterior, by depth ----------------
if os.path.exists(f'{RV}/depth_curves.json'):
	D = json.load(open(f'{RV}/depth_curves.json')); ds = sorted(int(k) for k in D)
	g = lambda key: [100 * D[str(d)][key] if D[str(d)][key] is not None else np.nan for d in ds]
	fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
	ax = axs[0]
	ax.bar(ds, g('forced_share'), color='#fde2e2', edgecolor='#c55', lw=0.6, label='FORCED share')
	ax.set_xlabel('depth'); ax.set_ylabel('FORCED share of free variables (%)'); ax.set_ylim(0, 100)
	ax2 = ax.twinx(); ax2.plot(ds, [D[str(d)]['n_sol_median'] for d in ds], color='k', marker='o', ms=2.5, lw=0.9); ax2.set_yscale('log'); ax2.set_ylabel('surviving $|\\mathcal{S}(F,v)|$ (median)')
	ax.set_title('(a) what a node looks like', fontsize=8)
	ax = axs[1]
	ax.plot(ds, g('ceil_all'), color='k', ls='--', lw=1.0, label='Bayes ceiling')
	ax.plot(ds, g('gat_in_all'), color=C['m0'], marker='o', ms=2.5, label='GAT, in-tree (Stage 1)')
	ax.plot(ds, g('mlp_in_all'), color=C['mlp'], marker='^', ms=2.5, label='MLP, in-tree')
	ax.plot(ds, g('gat_root_all'), color=C['root'], marker='v', ms=2.5, label='GAT, root-only')
	ax.plot(ds, g('lp_all'), color=C['lp'], marker='s', ms=2.5, label='LP rounding')
	ax.set_xlabel('depth'); ax.set_ylabel('accuracy, all free variables (%)'); ax.set_ylim(55, 101); ax.legend(frameon=False, fontsize=5.6, loc='upper left'); ax.set_title('(b) accuracy vs. the ceiling', fontsize=8)
	ax = axs[2]
	ax.plot(ds, g('gat_in_forced'), color=C['m0'], marker='o', ms=2.5, label='GAT, in-tree (Stage 1)')
	ax.plot(ds, g('mlp_in_forced'), color=C['mlp'], marker='^', ms=2.5, label='MLP, in-tree')
	ax.plot(ds, g('gat_root_forced'), color=C['root'], marker='v', ms=2.5, label='GAT, root-only')
	ax.plot(ds, g('lp_forced'), color=C['lp'], marker='s', ms=2.5, label='LP rounding')
	ax.set_xlabel('depth'); ax.set_ylabel('accuracy on FORCED variables (%)'); ax.set_ylim(70, 101); ax.legend(frameon=False, fontsize=5.6, loc='lower right'); ax.set_title('(c) where errors cost a backtrack', fontsize=8)
	for a in axs: a.tick_params(labelsize=6.5)
	ax2.tick_params(labelsize=6.5)
	fig.tight_layout(w_pad=1.2)
	save(fig, 'prediction')
else:
	print('depth_curves.json missing: prediction figure skipped')

# ---------------- Fig: the lattice stage, coverage and mechanism ----------------
M = json.load(open(f'{RV}/lattice_mech.json'))
sizes = [('v22test_10x25', '$10{\\times}25$', [30, 30, 30], 30, 0.001), ('v22test_18x50', '$18{\\times}50$', [27, 28, 24], 30, 0.03),
         ('v22test_21x60', '$21{\\times}60$', [16, 14, 10], 30, 0.16), ('v24test_24x70', '$24{\\times}70$', [3, 4, 2], 30, 0.30), ('v24test_28x80', '$28{\\times}80$', [0, 1, 0], 30, 0.54)]
fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.9))
ax = axs[0]
cov = [100 * np.mean(s[2]) / s[3] for s in sizes]; lo = [100 * (np.mean(s[2]) - min(s[2])) / s[3] for s in sizes]; hi = [100 * (max(s[2]) - np.mean(s[2])) / s[3] for s in sizes]
ax.bar(range(5), cov, yerr=[lo, hi], color='#dde6f3', edgecolor=C['m0'], lw=0.7, capsize=2, error_kw=dict(lw=0.7))
for i, s in enumerate(sizes): ax.text(i, cov[i] + max(hi[i], 2) + 3, f'{s[4]:g} s', ha='center', fontsize=6, color='#444')
ax.set_xticks(range(5)); ax.set_xticklabels([s[1] for s in sizes], fontsize=6.5); ax.set_ylabel('instances solved by\nlattice stage (%)'); ax.set_ylim(0, 118)
ax.set_title('(a) coverage, 10 permutations, 3 seeds (range); time per instance above', fontsize=7)
ax = axs[1]
data = [[r['kernel_min'] for r in M[s[0]] if r['kernel_min'] is not None] for s in sizes]
bp = ax.boxplot(data, positions=range(5), widths=0.5, showfliers=False, patch_artist=True, medianprops=dict(color='k', lw=0.8))
for b in bp['boxes']: b.set(facecolor='#f0f0f0', edgecolor='#555', lw=0.7)
ax.plot(range(5), [M[s[0]][0]['sol_len'] for s in sizes], color=C['ft'], marker='D', ms=3.5, lw=1.0, label='solution vector, $\\|\\cdot\\|^2 = n+1$')
ax.plot([], [], color='#555', lw=0.7, label='shortest kernel vector in the reduced basis')
for i, s in enumerate(sizes):
	k = sum(r['sol_rows'] > 0 for r in M[s[0]]); ax.text(i, 92, f'{k}/30', ha='center', fontsize=6, color='#444')
ax.text(-0.45, 99, 'bases containing a solution row (one permutation):', fontsize=5.6, color='#444', va='bottom')
ax.set_xticks(range(5)); ax.set_xticklabels([s[1] for s in sizes], fontsize=6.5); ax.set_ylabel('squared length'); ax.set_ylim(0, 106); ax.legend(frameon=False, fontsize=5.8, loc='lower right')
ax.set_title('(b) why it fails: kernel vectors get shorter than the solution', fontsize=7)
for a in axs: a.tick_params(labelsize=6.5)
fig.tight_layout(h_pad=1.0)
save(fig, 'lattice')

# ---------------- Fig: scaling with size (transcribed from the size-transfer table) ----------------
n = [25, 50, 60, 70, 80]
T = {'CP-SAT': ([0.002, 0.046, 0.79, 16.1, 270], None, None), 'SCIP': ([0.009, 0.385, 2.59, 11.7, 112], None, None),
     'LP rule': ([0.03, 2.43, 41.2, 590, 1200], [0, 0, 0, 53, 0], ['', '', '', '', '4/30']),
     'Stage 1': ([0.07, 1.82, 16.5, 407, 1200], [0, 0, 0, 50, 0], ['', '', '', '', '8/30']),
     'Stage 1+2 (60)': ([np.nan, 0.52, 7.6, 155, 720], [0, 0, 0, 30, 0], ['', '', '', '', '18/30']),
     'Stage 1+2 (50)': ([np.nan, 1.5, 10.7, 150, 852], [0, 0, 0, 15, 0], ['', '', '', '', '17/30'])}
OFF = {'LP rule': (-16, 5), 'Stage 1': (4, 4), 'Stage 1+2 (60)': (4, -9), 'Stage 1+2 (50)': (4, 2)}
ANNOTATE_POINTS = False
N = {'LP rule': [46, 4224, 79609, 980743, 1929041], 'Stage 1': [24, 1085, 13386, 260114, 803535],
     'Stage 1+2 (60)': [np.nan, 424, 5495, 90054, 470606], 'Stage 1+2 (50)': [np.nan, 1147, 7796, 95818, 560077]}
style = {'CP-SAT': (C['cpsat'], 'x', ':'), 'SCIP': (C['scip'], '+', ':'), 'LP rule': (C['lp'], 's', '-'), 'Stage 1': (C['m0'], 'o', '-'), 'Stage 1+2 (60)': (C['ft'], 'D', '-'), 'Stage 1+2 (50)': (C['ft2'], 'v', '--')}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
ax = axs[0]
for k, (t, err, lab) in T.items():
	col, mk, ls = style[k]; t = np.array(t, dtype=float)
	ax.plot(n, t, color=col, marker=mk, ms=3.5, lw=1.0, ls=ls, label=k)
	if err is not None:
		ax.errorbar(n, t, yerr=err, color=col, lw=0.6, capsize=1.5, ls='none')
		if ANNOTATE_POINTS:
			for xi, ti, li in zip(n, t, lab):
				if li: ax.annotate(li, (xi, ti), textcoords='offset points', xytext=OFF[k], fontsize=5.6, color=col)
ax.axhline(1200, color='#bbb', lw=0.5, ls='--'); ax.text(66, 1550, 'budget at $n\\geq70$', fontsize=5.4, color='#888')
ax.text(0.985, 0.03, 'solved at $n=80$ (of 30):\nCP-SAT 23, SCIP 30, LP rule 4,\nStage 1 8, Stage 1+2 (60) 18, (50) 17\nunsolved arms drawn at the budget', transform=ax.transAxes, ha='right', va='bottom', fontsize=5.4, color='#444')
ax.set_ylim(1e-3, 4e3)
ax.set_yscale('log'); ax.set_xlabel('variables $n$ (rows $m$: 10, 18, 21, 24, 28)'); ax.set_ylabel('median solve time (s)'); ax.legend(frameon=False, fontsize=5.8, ncol=2, loc='upper left'); ax.set_title('(a) wall-clock time', fontsize=8)
ax = axs[1]
for k, v in N.items():
	col, mk, ls = style[k]; ax.plot(n, v, color=col, marker=mk, ms=3.5, lw=1.0, ls=ls, label=k)
ax.set_yscale('log'); ax.set_xlabel('variables $n$'); ax.set_ylabel('median nodes to a solution'); ax.legend(frameon=False, fontsize=5.8, loc='lower right'); ax.set_title('(b) search tree size (lattice off)', fontsize=8)
for a in axs: a.tick_params(labelsize=6.5); a.set_xticks(n)
fig.tight_layout(w_pad=1.5)
save(fig, 'scaling')

# ---------------- Fig: results (fine-tuning, per-instance nodes, where prediction helps, root residual) ----------------
def load(d): return {json.load(open(f))['instance']: json.load(open(f)) for f in glob.glob(d + '/inst-*_r.json')}
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
ax = axs[0]
for tag, lab, col in [('r2', 'seed 0', C['ft']), ('r2_seed1', 'seed 1', C['ft2'])]:
	L = json.load(open(f'{R}/{tag}/log.json'))
	ax.plot([0] + [r['epoch'] for r in L], [1.0] + [r['sum_ratio'] for r in L], marker='o', ms=3, color=col, label=lab)
ax.axhline(1.0, color='k', lw=0.5, ls=':'); ax.set_xlabel('fine-tuning epoch'); ax.set_ylabel('held-out subtree cost\n(relative to Stage-1 rule)'); ax.set_ylim(0.3, 1.1); ax.legend(frameon=False, fontsize=6.5); ax.set_title('(a) Stage-2 fine-tuning', fontsize=8)
ax = axs[1]
lp = load(f'{R}/H_M0_S0_ahloff_lp'); m0 = load(f'{R}/H_M0_S0_ahloff_model'); pv = load(f'{R}/H_PV_S0_ahloff_pv'); ks = sorted(lp, key=lambda k: m0[k]['nodes'])
for lab, Dd, col in [('LP rule', lp, C['lp']), ('Stage 1', m0, C['m0']), ('Stage 1+2', pv, C['ft'])]:
	ax.plot(range(len(ks)), sorted([Dd[k]['nodes'] for k in ks]), color=col, lw=1.2, label=lab)
ax.set_yscale('log'); ax.set_xlabel('instance (sorted per arm)'); ax.set_ylabel('nodes to a solution'); ax.legend(frameon=False, fontsize=6, loc='lower right', handlelength=1.2); ax.set_title('(b) $21\\times60$, 100 instances', fontsize=8)
ax = axs[2]
W = json.load(open(f'{R}/where_M012.json')); nfs = sorted({r['n_free'] for r in W})
lpv = [100 * np.mean([r['lp_forced'] for r in W if r['n_free'] == k]) for k in nfs]; m0v = [100 * np.mean([r['M0_forced'] for r in W if r['n_free'] == k]) for k in nfs]
ax.plot(nfs, lpv, marker='s', ms=3, color=C['lp'], label='LP rounding'); ax.plot(nfs, m0v, marker='o', ms=3, color=C['m0'], label='Stage 1 network')
ax.axvspan(13, 25, color='#ddd', alpha=0.6, lw=0); ax.text(19, 96, 'train\nrange', ha='center', fontsize=5.8, color='#555')
ax.set_xlabel('free variables ($21\\times60$ states)'); ax.set_ylabel('accuracy on FORCED (%)'); ax.legend(frameon=False, fontsize=6, loc='lower center'); ax.set_ylim(60, 103); ax.set_title('(c) where prediction helps', fontsize=8)
ax = axs[3]
r0 = json.load(open(f'{R}/r0_M0.json')); ratio = np.array([r['cost_chosen'] / max(r['cost_best'], 1) for r in r0])
ax.hist(np.log2(ratio), bins=np.arange(0, 6.5, 0.5), color=C['m0'], alpha=0.85)
ax.set_xlabel('$\\log_2$(pick cost / best of 5)'); ax.set_ylabel('states'); ax.set_title('(d) residual at the root', fontsize=8)
ax.text(0.97, 0.9, f'cheapest: {100*np.mean(ratio<=1):.0f}%\nmedian {np.median(ratio):.2f}x', transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
for a in axs: a.tick_params(labelsize=6.5)
fig.tight_layout(w_pad=1.6)
save(fig, 'results')

# ---------------- Fig: the two signals are not interchangeable ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0]))
ax = axs[0]
keys = [('lp_forced', 'LP rounding', C['lp']), ('M0_forced', 'Stage 1 ($10{\\times}25$ states)', C['m0']), ('M2_forced', 'M2 (half/half)', '#6baed6'), ('M1_forced', 'M1 ($21{\\times}60$ states)', '#08306b')]
xs = [50, 60]; w = 0.2
for i, (k, lab, col) in enumerate(keys):
	vals = [100 * np.mean([r[k] for r in W if r['n_free'] == x]) for x in xs]
	ax.bar([p + (i - 1.5) * w for p in range(2)], vals, width=w, color=col, label=lab)
	for p, v in enumerate(vals): ax.text(p + (i - 1.5) * w, v + 0.6, f'{v:.1f}', ha='center', fontsize=5.0)
ax.set_xticks(range(2)); ax.set_xticklabels(['50 free\nvariables', '60 free\nvariables'], fontsize=6.2); ax.set_ylabel('accuracy on FORCED (%)'); ax.set_ylim(55, 122)
ax.legend(frameon=False, fontsize=5.2, loc='upper left', ncol=2, columnspacing=0.8, handlelength=1.0); ax.set_yticks([60, 70, 80, 90, 100])
ax.set_title('(a) value accuracy where the margin lives', fontsize=7.6)
ax = axs[1]
names = ['LP\nrule', 'Stage 1', 'M2', 'M1', 'switch', 'S1+2\nseed 0', 'S1+2\nseed 1']; nodes = [79609, 13386, 18558, 21337, 17252, 5495, 4940]
cols = [C['lp'], C['m0'], '#6baed6', '#08306b', '#9ecae1', C['ft'], C['ft2']]
ax.bar(range(len(names)), nodes, color=cols); ax.set_yscale('log'); ax.set_ylim(2000, 200000)
for i, v in enumerate(nodes): ax.text(i, v * 1.12, f'{v:,}', ha='center', fontsize=5.0)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=5.4, rotation=35, ha='right', rotation_mode='anchor'); ax.set_ylabel('median nodes, $21\\times60$ (100 inst.)'); ax.set_title('(b) tree size', fontsize=7.6)
ax = axs[2]
labels = ['$|\\hat p-1/2|$\nof the pick', 'pick FORCED\n& correct (%)', 'subtree cost\n(relative)']
s1 = [0.43, 82.5, 1.00]; s2 = [0.26, 57.5, 0.48]
for i in range(3):
	a = ax.inset_axes([0.02 + i * 0.36, 0.14, 0.24, 0.60])
	a.bar([0, 1], [s1[i], s2[i]], color=[C['m0'], C['ft']], width=0.7)
	a.set_xticks([0, 1]); a.set_xticklabels(['S1', 'S1+2'], fontsize=5.2); a.tick_params(labelsize=5.0, length=2)
	a.set_title(labels[i], fontsize=5.0, pad=2)
	for j, v in enumerate([s1[i], s2[i]]): a.text(j, v, f'{v:g}', ha='center', va='bottom', fontsize=5.0)
	a.set_ylim(0, max(s1[i], s2[i]) * 1.28)
ax.axis('off'); ax.set_title('(c) what fine-tuning changed', fontsize=7.6)
for a in axs[:2]: a.tick_params(labelsize=6.5)
fig.tight_layout(w_pad=1.2)
save(fig, 'signals')
