#!/usr/bin/env python3
"""Generates the two v11 report figures (negative-result scoreboard, baseline
reality-check time/accuracy comparison) from this cycle's recorded numbers."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

# ---------------------------------------------------------------------------
# Figure 1: ten independent negative results, all pinned near chance (AUC 0.5),
# vs the one positive result found this cycle (not AUC-comparable, annotated).
fig, ax = plt.subplots(figsize=(8, 6))
labels = [
	'Single-shot scalar (v5-v7)', 'Multi-probe aggregate (v10)',
	'Frozen GNN embedding (v10)', 'Paired data, 7-feat (this cycle)',
	'Dual-node GNN, 16L (this cycle)', 'Dual-node GNN, 16L+LP (this cycle)',
	'Dual-node GNN, 4L (this cycle)', 'Bipartite GNN, rand-feat=0 (this cycle)',
	'Bipartite GNN, rand-feat=8 (this cycle)', 'Bipartite GNN, rand-feat=32 (this cycle)',
	'Bipartite GNN, max-pool (this cycle)', 'Bipartite GNN, mean+max-pool (this cycle)',
	'GS-profile sequence model (this cycle)',
]
aucs = [0.54, 0.588, 0.497, 0.599, 0.5001, 0.5003, 0.5002, 0.5001, 0.5001, 0.5001, 0.5001, 0.5002, 0.5000]
colors = ['#888888'] * 3 + ['#4c72b0'] * 10
y = np.arange(len(labels))
ax.barh(y, aucs, color=colors, height=0.6)
ax.axvline(0.5, color='black', linewidth=1, linestyle='--', label='chance (AUC 0.5)')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0.45, 0.65)
ax.set_xlabel('Held-out AUC (feasibility classification)')
ax.set_title('Ten independent feasibility-classification attempts:\nall pinned near chance except the weak aggregate-scalar signal', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('../figures/v11_negative_scoreboard.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: baseline reality check -- solve accuracy and time by size/method
sizes = ['10x25', '20x50', '40x100', '60x150']
methods = {
	'Gurobi':  ([100.0, 100.0, 99.4, None], [0.0023, 0.0276, 1.875, None]),
	'SCIP':    ([100.0, 100.0, 93.2, None], [0.0053, 0.141, 9.24, None]),
	'CP-SAT':  ([100.0, 100.0, 90.9, 16.67], [0.0011, 0.0307, 8.03, 17.72]),
	'Ours (proposed pipeline)': ([100.0, 78.25, 66.67, 6.67], [0.23, 3.78, 22.0, 107.95]),
}
colors2 = {'Gurobi': '#55a868', 'SCIP': '#c44e52', 'CP-SAT': '#8172b2', 'Ours (proposed pipeline)': '#dd8452'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(sizes))
width = 0.2
for i, (m, (acc, t)) in enumerate(methods.items()):
	vals = [a if a is not None else 0 for a in acc]
	mask = [a is not None for a in acc]
	xs = x + (i - 1.5) * width
	bars = axes[0].bar(xs, vals, width, label=m, color=colors2[m])
	for xb, v, present in zip(xs, vals, mask):
		if not present:
			axes[0].text(xb, 2, 'n/a', ha='center', fontsize=8, rotation=90, color='gray')
axes[0].set_xticks(x); axes[0].set_xticklabels(sizes)
axes[0].set_ylabel('Solution accuracy (%)')
axes[0].set_title('Solve rate by size')
axes[0].legend(fontsize=8, loc='lower left')
axes[0].set_ylim(0, 108)

for i, (m, (acc, t)) in enumerate(methods.items()):
	xs_present = [j for j, v in enumerate(t) if v is not None]
	ys_present = [t[j] for j in xs_present]
	axes[1].plot([sizes[j] for j in xs_present], ys_present, marker='o', label=m, color=colors2[m])
axes[1].set_yscale('log')
axes[1].set_ylabel('Avg time / instance (s, log scale)')
axes[1].set_title('Wall-clock time by size\n(CP-SAT/ours at 60x150: 30s cap / full pipeline)')
axes[1].legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig('../figures/v11_baseline_reality_check.png', dpi=150)
plt.close()

print("wrote v11_negative_scoreboard.png and v11_baseline_reality_check.png")
