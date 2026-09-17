"""Figures for docs/tex/260917_paper (EN) and its Korean edition. Everything is matplotlib so the
same PDF (LaTeX) and PNG (reportlab) come from one source. Run from this directory."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = '../../docs/tex/fig'; R = '../../runs/v24'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6})
C = dict(lp='#7f7f7f', m0='#1f77b4', ft='#d62728', box='#f2f2f2', learn='#e8f0fb', rl='#fbe9e7', edge='#333333')


def save(fig, name):
	fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png', dpi=220, bbox_inches='tight'); plt.close(fig)
	print('saved', name)


def box(ax, x, y, w, h, text, fc=C['box'], fs=7.5, bold=False, ec=C['edge'], lw=0.8, style='round,pad=0.02,rounding_size=0.02'):
	ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
	ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, color=C['edge'], lw=0.9, style='-|>', ls='-'):
	ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color, lw=lw, linestyle=ls))


# ---------------- Fig 1: framework ----------------
fig, ax = plt.subplots(figsize=(7.0, 3.1)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis('off')
stages = ['propagation', 'LP infeasibility\nrule', 'kernel pump', 'AHL / BKZ\n(lattice)', 'guided DFS']
for i, t in enumerate(stages):
	box(ax, 0.2 + i * 1.98, 3.3, 1.7, 0.8, t, fc='#fff' if i < 4 else C['box'], bold=(i == 4))
	if i < 4: arrow(ax, 0.2 + i * 1.98 + 1.7, 3.7, 0.2 + (i + 1) * 1.98, 3.7)
ax.text(0.2, 4.25, 'Deployed cascade (sound stages first; the learned components act only in the last stage)', fontsize=7.5, va='bottom')
# node decision zoom
ax.add_patch(FancyBboxPatch((0.2, 0.15), 9.6, 2.75, boxstyle='round,pad=0.02,rounding_size=0.03', fc='#fafafa', ec='#999', lw=0.6, ls='--'))
arrow(ax, 9.0, 3.3, 8.2, 2.9, ls='--', color='#999')
ax.text(0.35, 2.7, 'One search node: reduced instance $(A^{\\prime}, b^{\\prime})$ after propagation', fontsize=7.5, va='center')
box(ax, 0.4, 0.9, 1.8, 1.4, 'reduced\ninstance\n$(A^{\\prime}, b^{\\prime})$\n+ LP vertex $x_{LP}$', fc='#fff')
arrow(ax, 2.2, 1.6, 2.9, 1.6)
box(ax, 2.9, 0.9, 2.4, 1.4, 'MarginalNet (frozen)\nbipartite GAT, 21.8K params\n$\\hat p_j \\approx \\Pr[x_j{=}1 \\mid A^{\\prime}, b^{\\prime}]$', fc=C['learn'], fs=6.8)
arrow(ax, 5.3, 1.9, 6.1, 2.15); arrow(ax, 5.3, 1.3, 6.1, 1.05)
box(ax, 6.1, 1.85, 2.2, 0.75, 'value head: first value\n$v = \\mathbb{1}[\\hat p_{j^*} \\geq 1/2]$', fc=C['learn'])
box(ax, 6.1, 0.65, 2.2, 0.85, 'policy head (cost-tuned)\n$j^* = \\arg\\max_j\\, g_j$,\n$g_j = \\log|\\hat p_j - 1/2| + h_\\phi(\\mathbf{h}_j)$', fc=C['rl'], fs=7)
arrow(ax, 8.3, 2.2, 9.0, 1.7); arrow(ax, 8.3, 1.05, 9.0, 1.45)
box(ax, 9.0, 1.2, 0.75, 0.75, 'branch\n$x_{j^*}{=}v$\nfirst', fc='#fff', fs=6.5)
ax.text(0.4, 0.4, 'Stage 1 (supervised, exact conditional marginals) trains the network and the value rule; '
                  'Stage 2 (actor-critic on exact subtree costs) trains only $h_\\phi$.\nNeither fixes a variable: soundness rests on propagation and LP-probing; a wrong prediction costs backtracking only.', fontsize=6.6, va='center')
save(fig, 'framework')

# ---------------- Fig 2: reduction identity + FORCED/OPEN partition ----------------
fig, ax = plt.subplots(figsize=(7.0, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis('off')
ax.text(0.2, 3.2, 'Conditioning on a partial assignment is the same as reducing the instance (Proposition 1)', fontsize=7.5, va='center')
box(ax, 0.2, 1.2, 2.6, 1.7, 'root: $Ax = b$\n$n$ variables, solution set $\\mathcal{S}$\n\n$\\mathcal{S}$ = {solutions}', fc='#fff')
arrow(ax, 2.8, 2.05, 3.5, 2.05); ax.text(3.15, 2.95, 'fix $x_F{=}v$', ha='center', fontsize=6.5)
box(ax, 3.5, 1.2, 3.4, 1.7, 'node: $A_K\\, y = b - A_F v$ ($|K|$ free)\n$\\mathcal{S}(F,v)$: solutions agreeing with $v$\n$p_j = $ share of $\\mathcal{S}(F,v)$ with $x_j{=}1$', fc='#fff', fs=7)
arrow(ax, 6.9, 2.05, 7.3, 2.05)
box(ax, 7.3, 1.9, 2.5, 1.0, 'FORCED: $p_j \\in \\{0, 1\\}$\nbranching against it\nempties the subtree', fc='#fde2e2', fs=7)
box(ax, 7.3, 0.75, 2.5, 1.0, 'OPEN: $0 < p_j < 1$\neither branch still\nreaches a solution', fc='#e3f2e1', fs=7)
ax.text(0.2, 0.55, 'Only errors on FORCED variables cost a backtrack. Their share rises from 6.3% at the root to 87.9% at depth 8 of a $10\\times25$ instance,\n'
                   'so the same network is trained on in-tree states (reduced instances) rather than roots; the labels are exact because $\\mathcal{S}$ is enumerated.', fontsize=6.6, va='center')
save(fig, 'reduction')

# ---------------- Fig 3: two-stage training ----------------
fig, ax = plt.subplots(figsize=(7.0, 3.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.15, 2.2), 9.7, 1.9, boxstyle='round,pad=0.02,rounding_size=0.03', fc=C['learn'], ec='#88a', lw=0.6))
ax.text(0.3, 3.95, 'Stage 1 -- supervised: what value is a variable likely to take?', fontsize=8, fontweight='bold', va='center')
box(ax, 0.3, 2.4, 2.0, 1.2, 'small instances\n($10\\times25$)\nenumerate $\\mathcal{S}$ exactly', fc='#fff')
arrow(ax, 2.3, 3.0, 2.8, 3.0)
box(ax, 2.8, 2.4, 2.3, 1.2, 'in-tree states\nprefix of a real solution,\ndepth $0$--$12$', fc='#fff')
arrow(ax, 5.1, 3.0, 5.6, 3.0)
box(ax, 5.6, 2.4, 2.0, 1.2, 'exact conditional\nmarginals $p_j$\n(soft labels)', fc='#fff')
arrow(ax, 7.6, 3.0, 8.1, 3.0)
box(ax, 8.1, 2.4, 1.6, 1.2, 'MarginalNet\nBCE to $p_j$', fc='#fff', bold=True)
ax.add_patch(FancyBboxPatch((0.15, 0.1), 9.7, 1.9, boxstyle='round,pad=0.02,rounding_size=0.03', fc=C['rl'], ec='#c88', lw=0.6))
ax.text(0.3, 1.85, 'Stage 2 -- cost-aware fine-tuning: which variable is cheapest to branch on?', fontsize=8, fontweight='bold', va='center')
box(ax, 0.3, 0.3, 2.0, 1.2, 'deployment-size states\n($21\\times60$, 45--60 free)\npolicy init = Stage-1 rule', fc='#fff')
arrow(ax, 2.3, 0.9, 2.8, 0.9)
box(ax, 2.8, 0.3, 2.3, 1.2, 'roll out sampled policy\n$\\pi \\propto \\exp(g_j/\\tau)$\nto a solution', fc='#fff')
arrow(ax, 5.1, 0.9, 5.6, 0.9)
box(ax, 5.6, 0.3, 2.0, 1.2, 'exact cost per decision\n$c(u)$ = subtree nodes\n(from the search itself)', fc='#fff')
arrow(ax, 7.6, 0.9, 8.1, 0.9)
box(ax, 8.1, 0.3, 1.6, 1.2, 'actor-critic\n$\\alpha = V_\\psi - \\log c$\ntrains $h_\\phi, V_\\psi$', fc='#fff', bold=True)
arrow(ax, 8.9, 2.4, 8.9, 1.5, ls='--', color='#c55'); ax.text(9.0, 1.95, 'frozen', fontsize=6.5, color='#c55', va='center')
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
