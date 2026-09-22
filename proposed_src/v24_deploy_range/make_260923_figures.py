"""Figures for docs/tex/260923_paper (EN) and its Korean edition. Everything is matplotlib so the
same PDF (LaTeX) and PNG (reportlab) come from one source. Run from this directory."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = '../../docs/tex/fig/260923'; R = '../../runs/v24'
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


# ---------------- Fig 1: framework (the system as evaluated: lattice stage + guided DFS) ----------------
fig, ax = canvas(7.0, 3.6)          # y range 0..5.14
ax.text(0.15, 5.02, 'Deployed system: a lattice stage, then a guided depth-first search', fontsize=7.4, va='center', fontweight='bold')
box(ax, 0.15, 3.85, 1.35, 0.75, 'instance\n$(A, b)$', fc='#ffffff', fs=7)
arrow(ax, 1.50, 4.22, 1.85, 4.22)
box(ax, 1.85, 3.85, 2.75, 0.75, 'AHL / BKZ lattice reduction\nup to 10 random column permutations\nsound: solution verified before return', fc=C['box'])
arrow(ax, 4.60, 4.35, 5.55, 4.60); ax.text(5.05, 4.60, 'solved', fontsize=6.2, color='#2a7', ha='center', va='bottom')
box(ax, 5.55, 4.32, 1.6, 0.55, 'return solution', fc='#eaf6ea', fs=6.8)
arrow(ax, 4.60, 4.05, 5.55, 3.90); ax.text(5.05, 3.86, 'residual', fontsize=6.2, color='#a55', ha='center', va='top')
box(ax, 5.55, 3.42, 1.6, 0.55, 'guided DFS', fc=C['box'], bold=True, fs=7.4)
ax.text(7.30, 4.10, r'lattice closes 30/30 at $10\times25$,' '\n' r'24-28/30 at $18\times50$,' '\n' r'10-16/30 at $21\times60$', fontsize=6.2, va='center', color='#444')
# node panel
ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 3.02, boxstyle='round,pad=0.02,rounding_size=0.04', fc='#fbfbfb', ec='#9a9a9a', lw=0.6, ls='--'))
arrow(ax, 6.10, 3.42, 5.60, 3.22, ls='--', color='#9a9a9a')
ax.text(0.30, 2.95, 'One node of the guided DFS (the only place learning acts)', fontsize=7.2, va='center', fontweight='bold')
seq = [(r'reduced instance' '\n' r'$A_K y = b - A_F v$' '\n' r'(Proposition 1)', '#ffffff'),
       (r'bound check' '\n' r'$b^\prime<0$ or $b^\prime>A^\prime \mathbf{1}$' '\n' r'$\Rightarrow$ prune', '#ffffff'),
       (r'propagation' '\n' r'row $i$: need $r_i$ of $u_i$' '\n' r'$r_i=0$ or $r_i=u_i \Rightarrow$ fix', '#eef4ea'),
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
