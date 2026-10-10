"""Density sweep table for the setup section: at each density and size, the strongest single-relaxation
classical rule against our Stage 1+2 arm (paired on commonly solved instances; median per-instance node
ratio, bootstrap 95% CI, sign test) plus median nodes of both. Learned model: trained at p=0.5 for the
p=0.5 row, at p=0.3 otherwise. Writes runs/v33/density_table.md and prints LaTeX rows.
"""
import glob, json, os
import numpy as np
from scipy.stats import binomtest

R = '../../runs'
TX = '\\times'; BS = '\\\\'
CL = [('most-fractional', 'lpfrac'), ('maxSD', 'maxsd'), ('dom/wdeg', 'domwdeg')]
# density -> size -> (classical dir base + tag, learned dir)
MAP = {
	0.2: {'21x60': (f'{R}/v30/d02/H_S0', f'{R}/v30/d02_m03/H_S0_pv_fast'), '24x70': (f'{R}/v30/d02/G70_S0', f'{R}/v30/d02_m03/G70_S0_pv_fast')},
	0.3: {'21x60': (f'{R}/v28/p05/H_S0', f'{R}/v29/H_S0_pv_fast'), '24x70': (f'{R}/v32/d03/G70_S0', f'{R}/v32/d03_m03/G70_S0_pv_fast'), '28x80': (f'{R}/v32/d03/G80_S0', f'{R}/v32/d03_m03/G80_S0_pv_fast')},
	0.4: {'21x60': (f'{R}/v30/d04/H_S0', f'{R}/v30/d04_m03/H_S0_pv_fast'), '24x70': (f'{R}/v32/d04/G70_S0', f'{R}/v32/d04_m03/G70_S0_pv_fast'), '28x80': (f'{R}/v32/d04/G80_S0', f'{R}/v32/d04_m03/G80_S0_pv_fast')},
	0.5: {'21x60': (f'{R}/v25/H_S0', f'{R}/v25/H_S0_pv_fast'), '24x70': (f'{R}/v25/G70_S0', f'{R}/v25/G70_S0_pv_fast'), '28x80': (f'{R}/v25/G80_S0', f'{R}/v24/G80_PV_S0_ahloff_pv')},
}


def load(d):
	out = {}
	for f in glob.glob(d + '/inst-*.json'):
		r = json.load(open(f)); out[r['instance'].replace('_r.json', '').replace('.json', '')] = dict(solved=bool(r['solved']), nodes=r['nodes'])
	return out


def pair(A, B):
	ks = [k for k in sorted(set(A) & set(B)) if A[k]['solved'] and B[k]['solved']]
	r = np.array([A[k]['nodes'] / max(B[k]['nodes'], 1) for k in ks]); w = int((r > 1).sum()); nt = int((r != 1).sum())
	rng = np.random.default_rng(0); bs = [np.median(rng.choice(r, len(r))) for _ in range(1000)]
	return dict(n=len(ks), ratio=float(np.median(r)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), wins=w, nt=nt, p=float(binomtest(w, nt, 0.5).pvalue) if nt else 1.0)


def main():
	md = ['# Density sweep: strongest single-relaxation classical rule vs Stage 1+2 (nodes)', '', '| density | size | n | strongest classical | its median nodes | Stage 1+2 median nodes | ratio [CI] | wins | p |', '|---|---|---|---|---|---|---|---|---|']
	tex = []
	for p in sorted(MAP):
		for size, (cb, ld) in MAP[p].items():
			L = load(ld)
			if not L: continue
			best = None
			for name, arm in CL:
				C = load(f'{cb}_{arm}')
				if not C: continue
				med = np.median([r['nodes'] for r in C.values() if r['solved']])
				if best is None or med < best[1]: best = (name, med, C)
			if best is None: continue
			c = pair(best[2], L); lm = np.median([r['nodes'] for r in L.values() if r['solved']])
			md.append(f"| {p} | {size} | {len(L)} | {best[0]} | {best[1]:,.0f} | {lm:,.0f} | {c['ratio']:.2f}x [{c['lo']:.2f}, {c['hi']:.2f}] | {c['wins']}/{c['nt']} | {c['p']:.2g} |")
			sz = size.replace('x', TX); tex.append(f"${p}$ & ${sz}$ & {len(L)} & {best[0]} & {best[1]:,.0f} & {lm:,.0f} & ${c['ratio']:.2f}{TX}$ [{c['lo']:.2f}, {c['hi']:.2f}] & {c['wins']}/{c['nt']} & ${c['p']:.2g}$ " + BS)
	open(f'{R}/v33/density_table.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md)); print('\n'.join(tex))


if __name__ == '__main__':
	main()
