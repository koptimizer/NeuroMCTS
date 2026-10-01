"""v25 result tables from runs/v25: for every tag (instance set x seed) and arm, solved count, median
nodes, median time, PAR-2, median nodes in refuted subtrees, and the per-instance ratio against a
reference arm with a sign test (ties excluded) and a bootstrap CI. Also the oracle/variant/classical
blocks and the per-arm comparison across seeds. Writes runs/v25/tables.md.
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import binomtest

R = '../../runs/v25'
BUD = dict(A=60, B=300, C=600, H=600, G70=1200, G80=1200, V=600, INF=600)


def load(d):
	out = {}
	for f in glob.glob(d + '/inst-*.json'):
		r = json.load(open(f)); out[r['instance']] = r
	return out


def ratio(A, B, key, budget):
	ks = sorted(set(A) & set(B)); both = [k for k in ks if A[k]['solved'] and B[k]['solved']]
	if not both: return None
	r = np.array([A[k][key] / max(B[k][key], 1e-9) for k in both]); w = int((r > 1).sum()); t = int((r == 1).sum()); nt = len(r) - t
	rng = np.random.default_rng(0); bs = [np.median(rng.choice(r, len(r))) for _ in range(1000)]
	return dict(common=len(both), median_ratio=float(np.median(r)), ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), wins=w, nt=nt, p=float(binomtest(w, nt, 0.5).pvalue) if nt else 1.0)


def par2(X, key, budget):
	return float(np.mean([X[k][key] if X[k]['solved'] else 2 * budget for k in X]))


def main():
	ap = argparse.ArgumentParser(); ap.add_argument('--root', default=R); ap.add_argument('--out', default=R + '/tables.md'); a = ap.parse_args()
	dirs = sorted(d for d in glob.glob(a.root + '/*_S?_*') if os.path.isdir(d))
	tags = {}
	for d in dirs:
		name = os.path.basename(d); i = name.index('_S'); tag, arm = name[:i + 3], name[i + 4:]
		tags.setdefault(tag, {})[arm] = load(d)
	lines = []
	for tag in sorted(tags):
		arms = tags[tag]; base = tag.split('_')[0]; budget = BUD.get(base, 600)
		ref = 'lp' if 'lp' in arms else sorted(arms)[0]
		lines.append(f"\n## {tag}  (budget {budget}s, reference arm: {ref})\n")
		lines.append("| arm | solved | median nodes | median sec | PAR-2 sec | refuted share | nodes vs ref: median ratio [CI] wins p | sec vs ref |")
		lines.append("|---|---|---|---|---|---|---|---|")
		for arm in sorted(arms, key=lambda x: (x != ref, x)):
			X = arms[arm]; sol = [r for r in X.values() if r['solved']]
			mn = np.median([r['nodes'] for r in sol]) if sol else float('nan'); ms = np.median([r['sec'] for r in sol]) if sol else float('nan')
			rs = np.median([r['refuted_nodes'] / max(r['nodes'], 1) for r in sol]) if sol else float('nan')
			cn = ratio(arms[ref], X, 'nodes', budget); cs = ratio(arms[ref], X, 'sec', budget)
			f = lambda c: '—' if c is None else f"{c['median_ratio']:.2f}x [{c['ci'][0]:.2f},{c['ci'][1]:.2f}] {c['wins']}/{c['nt']} p={c['p']:.2g}"
			lines.append(f"| {arm} | {len(sol)}/{len(X)} | {mn:,.0f} | {ms:.2f} | {par2(X, 'sec', budget):.1f} | {rs:.3f} | {f(cn)} | {f(cs)} |")
	open(a.out, 'w').write('\n'.join(lines)); print('\n'.join(lines))


if __name__ == '__main__':
	main()
