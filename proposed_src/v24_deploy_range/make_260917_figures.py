"""Figures for docs/tex/260917_paper (EN) and its Korean edition. Everything is matplotlib so the
same PDF (LaTeX) and PNG (reportlab) come from one source. Run from this directory."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = '../../docs/tex/legacy/260917/fig'; R = '../../runs/v24'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6})
C = dict(lp='#7f7f7f', m0='#1f77b4', ft='#d62728', box='#f2f2f2', learn='#e8f0fb', rl='#fbe9e7', edge='#333333')


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


# ---------------- Fig 1: framework ----------------
fig, ax = canvas(7.0, 3.4)          # y range 0..4.86
ax.text(0.15, 4.65, 'Deployed cascade: sound stages first, learned components only in the last stage', fontsize=7.2, va='center')
stages = ['propagation', 'LP infeasibility\nrule', 'kernel pump', 'AHL / BKZ\nlattice reduction', 'guided DFS\n(this paper)']
bw, gap = 1.72, 0.25
for i, t in enumerate(stages):
	x = 0.15 + i * (bw + gap)
	box(ax, x, 3.65, bw, 0.8, t, fc='#ffffff' if i < 4 else C['box'], bold=(i == 4), fs=7)
	if i < 4: arrow(ax, x + bw, 4.05, x + bw + gap, 4.05)
# dashed panel: one node
ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 3.15, boxstyle='round,pad=0.02,rounding_size=0.04', fc='#fbfbfb', ec='#9a9a9a', lw=0.6, ls='--'))
arrow(ax, 8.9, 3.65, 8.4, 3.32, ls='--', color='#9a9a9a')
ax.text(0.3, 3.1, 'Inside one search node', fontsize=7.2, va='center', fontweight='bold')
box(ax, 0.3, 1.45, 1.75, 1.35, 'reduced instance\n$(A^{\prime}, b^{\prime})$\nafter propagation\n+ LP vertex', fc='#ffffff')
arrow(ax, 2.05, 2.12, 2.4, 2.12)
box(ax, 2.4, 1.45, 2.45, 1.35, 'MarginalNet $f_\\theta$ (frozen)\nbipartite graph attention\n21.8K parameters\n$\\hat p_j \\approx \\Pr[x_j{=}1 \\mid A^{\prime}, b^{\prime}]$', fc=C['learn'])
arrow(ax, 4.85, 2.4, 5.35, 2.62); arrow(ax, 4.85, 1.85, 5.35, 1.6)
box(ax, 5.35, 2.25, 2.55, 0.85, 'Stage 1: first value\n$v = \\mathbb{1}[\\hat p_{j^*} \\geq 1/2]$', fc=C['learn'])
box(ax, 5.35, 1.1, 2.55, 1.0, 'Stage 2: variable (cost-tuned)\n$j^* = \\arg\\max_j g_j$\n$g_j = \\log|\\hat p_j - 1/2| + h_\\phi(\\mathbf{h}_j)$', fc=C['rl'])
arrow(ax, 7.9, 2.67, 8.35, 2.3); arrow(ax, 7.9, 1.6, 8.35, 1.95)
box(ax, 8.35, 1.7, 1.35, 0.85, 'branch\n$x_{j^*} = v$ first,\n$1{-}v$ on backtrack', fc='#ffffff')
ax.text(0.3, 0.55, 'Neither head fixes a variable: fixing is done only by propagation and LP-probing (sound),\n'
                   'so a wrong prediction costs backtracking and nothing else.', fontsize=6.4, va='center', color='#333')
save(fig, 'framework')

# ---------------- Fig 2: reduction identity + FORCED/OPEN partition ----------------
fig, ax = canvas(7.0, 2.7)          # y range 0..3.86
ax.text(0.15, 3.65, 'Conditioning on a partial assignment is the same as reducing the instance (Proposition 1)', fontsize=7.2, va='center', fontweight='bold')
box(ax, 0.15, 1.35, 2.5, 1.9, 'root instance\n$A x = b$\n$n$ variables\nsolution set $\\mathcal{S}$', fc='#ffffff')
arrow(ax, 2.65, 2.3, 3.35, 2.3); ax.text(3.0, 2.62, 'fix\n$x_F{=}v$', ha='center', va='bottom', fontsize=6, linespacing=1.1)
box(ax, 3.35, 1.35, 3.4, 1.9, 'node = reduced instance\n$A_K\\, y = b - A_F v$,  $|K|$ free variables\n\nsurviving solutions $\\mathcal{S}(F,v)$\n$p_j$ = fraction of $\\mathcal{S}(F,v)$ with $x_j{=}1$', fc='#ffffff')
arrow(ax, 6.75, 2.3, 7.35, 2.3)
box(ax, 7.35, 2.35, 2.5, 0.9, 'FORCED: $p_j \\in \\{0, 1\\}$\nwrong branch empties\nthe subtree', fc='#fde2e2')
box(ax, 7.35, 1.35, 2.5, 0.9, 'OPEN: $0 < p_j < 1$\neither branch still\nreaches a solution', fc='#e3f2e1')
ax.text(0.15, 0.6, 'Only errors on FORCED variables cost a backtrack. Their share is 6.3% at the root and 87.9% at depth 8 ($10\\times25$),\n'
                   'so the network is trained on reduced instances (in-tree states); labels are exact because $\\mathcal{S}$ is enumerated.', fontsize=6.4, va='center', color='#333')
save(fig, 'reduction')

# ---------------- Fig 3: two-stage training ----------------
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

# ---------------- Fig 4: results ----------------
def load(d): return {json.load(open(f))['instance']: json.load(open(f)) for f in glob.glob(d + '/inst-*_r.json')}
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
# (a) fine-tuning monitor curves
ax = axs[0]
for tag, lab, col in [('r2', 'seed 0', C['ft']), ('r2_seed1', 'seed 1', '#ff9896')]:
	L = json.load(open(f'{R}/{tag}/log.json'))
	ax.plot([0] + [r['epoch'] for r in L], [1.0] + [r['sum_ratio'] for r in L], marker='o', ms=3, color=col, label=lab)
ax.axhline(1.0, color='k', lw=0.5, ls=':'); ax.set_xlabel('fine-tuning epoch'); ax.set_ylabel('held-out subtree cost\n(relative to supervised rule)'); ax.set_ylim(0.3, 1.1); ax.legend(frameon=False, fontsize=6.5); ax.set_title('(a) fine-tuning', fontsize=8)
# (b) per-instance nodes on the 100-instance set
ax = axs[1]
lp = load(f'{R}/H_M0_S0_ahloff_lp'); m0 = load(f'{R}/H_M0_S0_ahloff_model'); pv = load(f'{R}/H_PV_S0_ahloff_pv'); ks = sorted(lp, key=lambda k: m0[k]['nodes'])
for lab, D, col in [('LP rule', lp, C['lp']), ('supervised', m0, C['m0']), ('+ cost fine-tuning', pv, C['ft'])]:
	ax.plot(range(len(ks)), sorted([D[k]['nodes'] for k in ks]), color=col, lw=1.2, label=lab)
ax.set_yscale('log'); ax.set_xlabel('instance (sorted per arm)'); ax.set_ylabel('nodes to a solution'); ax.legend(frameon=False, fontsize=5.8, loc='lower right', handlelength=1.2); ax.set_title('(b) $21\\times60$, 100 instances', fontsize=8)
# (c) where the supervised network beats LP rounding
ax = axs[2]
W = json.load(open(f'{R}/where_M012.json')); nfs = sorted({r['n_free'] for r in W})
lpv = [100 * np.mean([r['lp_forced'] for r in W if r['n_free'] == n]) for n in nfs]; m0v = [100 * np.mean([r['M0_forced'] for r in W if r['n_free'] == n]) for n in nfs]
ax.plot(nfs, lpv, marker='s', ms=3, color=C['lp'], label='LP rounding'); ax.plot(nfs, m0v, marker='o', ms=3, color=C['m0'], label='MarginalNet')
ax.axvspan(13, 25, color='#ddd', alpha=0.6, lw=0); ax.text(19, 96, 'train\nrange', ha='center', fontsize=5.8, color='#555')
ax.set_xlabel('free variables at the node'); ax.set_ylabel('accuracy on FORCED (%)'); ax.legend(frameon=False, fontsize=6, loc='lower center'); ax.set_ylim(60, 103); ax.set_title('(c) where prediction helps', fontsize=8)
# (d) R0: cost of the confident choice vs the cheapest candidate
ax = axs[3]
r0 = json.load(open(f'{R}/r0_M0.json')); ratio = np.array([r['cost_chosen'] / max(r['cost_best'], 1) for r in r0])
ax.hist(np.log2(ratio), bins=np.arange(0, 6.5, 0.5), color=C['m0'], alpha=0.85)
ax.set_xlabel('$\\log_2$ cost ratio, pick / best of 5'); ax.set_ylabel('states'); ax.set_title('(d) residual at the root', fontsize=8)
ax.text(0.97, 0.9, f'cheapest: {100*np.mean(ratio<=1):.0f}%\nmedian {np.median(ratio):.2f}x', transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
for a in axs: a.tick_params(labelsize=6.5)
fig.tight_layout(w_pad=1.0)
save(fig, 'results')
