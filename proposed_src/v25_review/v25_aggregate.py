"""v25 F1-F6: one aggregation with explicit definitions, over the v24/v23 result directories the
paper's tables came from (and, with --root runs/v25, over the v25 runs which share the record
format up to field names).

For every (set, seed) and arm pair it reports
  ratio of medians            median(nodes_A) / median(nodes_B)
  median per-instance ratio   median_i nodes_A(i) / nodes_B(i)          (the paper's multipliers)
  wins / ties, two-sided exact sign test with ties excluded
  95% bootstrap CI of the median per-instance ratio (instances resampled)
  PAR-2                       mean time with unsolved counted at 2 x budget
  common-solved paired        the same on the instances both arms solved
Output: runs/v25/aggregate_<root>.json and a printed table.
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import binomtest

BUDGET = dict(A=60, B=300, C=600, H=600, G70=1200, P70=1200, G80=1200, R=300, D=600, Crand=600)


def load(d, v25=False):
	out = {}
	for f in glob.glob(d + ('/inst-*.json' if v25 else '/inst-*_r.json')):
		r = json.load(open(f))
		key = r['instance'].replace('_r.json', '').replace('.json', '')
		out[key] = dict(solved=bool(r['solved']), nodes=r['nodes'], sec=r.get('sec', r.get('total_sec')), solved_by=r.get('solved_by'))
	return out


def compare(A, B, budget, key='nodes', boot=2000, seed=0):
	ks = sorted(set(A) & set(B))
	both = [k for k in ks if A[k]['solved'] and B[k]['solved']]
	a = np.array([A[k][key] for k in both], float); b = np.array([B[k][key] for k in both], float)
	if len(both) == 0:
		return dict(n=len(ks), common=0)
	r = a / np.maximum(b, 1e-9)
	w = int((r > 1).sum()); t = int((r == 1).sum()); nt = len(r) - t
	p = binomtest(w, nt, 0.5).pvalue if nt else 1.0
	rng = np.random.default_rng(seed)
	bs = np.array([np.median(rng.choice(r, len(r), replace=True)) for _ in range(boot)])
	par2 = lambda X: float(np.mean([X[k][key] if X[k]['solved'] else 2 * budget for k in ks])) if key == 'sec' else None
	return dict(n=len(ks), common=len(both), solved_A=sum(A[k]['solved'] for k in ks), solved_B=sum(B[k]['solved'] for k in ks),
	            median_A=float(np.median(a)), median_B=float(np.median(b)), ratio_of_medians=float(np.median(a) / np.median(b)),
	            median_ratio=float(np.median(r)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
	            wins_A_larger=w, ties=t, p_sign=float(p), par2_A=par2(A), par2_B=par2(B))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', default='../../runs/v24')
	ap.add_argument('--out', default='../../runs/v25/aggregate_v24.json')
	a = ap.parse_args()
	v25 = a.root.rstrip('/').endswith('v25')
	res = {}
	# arm directories: <tag>_<model>_S<seed>_ahl<mode>_<arm>  (v24) ; <tag>_S<seed>_<arm> (v25)
	pairs = [('lp', 'model', 'M0'), ('model', 'pv', 'PV'), ('model', 'pv', 'PV50')]
	for tag in ['B', 'C', 'H', 'G70', 'G80']:
		for seed in [0, 1, 2]:
			for armA, armB, mdl in pairs:
				if v25:
					dA = f"{a.root}/{tag}_S{seed}_{armA}"; dB = f"{a.root}/{tag}_S{seed}_{armB}"
				else:
					dA = f"{a.root}/{tag}_M0_S{seed}_ahloff_{armA}"; dB = f"{a.root}/{tag}_{mdl}_S{seed}_ahloff_{armB}"
				if not (os.path.isdir(dA) and os.path.isdir(dB)): continue
				A = load(dA, v25); B = load(dB, v25)
				if not A or not B: continue
				for key in ['nodes', 'sec']:
					c = compare(A, B, BUDGET[tag], key)
					res[f"{tag}|S{seed}|{armA}({'M0' if armA!='lp' else 'lp'}) vs {armB}({mdl if armB=='pv' else 'M0'})|{key}"] = c
					if c.get('common'):
						print(f"{tag:4s} S{seed} {armA:>5s} vs {armB}({mdl}) {key:5s}: n={c['n']} common={c['common']} solved {c['solved_A']}/{c['solved_B']} | "
						      f"median {c['median_A']:.1f} vs {c['median_B']:.1f} (ratio of medians {c['ratio_of_medians']:.2f}x) | median ratio {c['median_ratio']:.2f}x CI[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] | "
						      f"A larger on {c['wins_A_larger']}/{c['n'] - c['ties']} p={c['p_sign']:.2g}" + (f" | PAR-2 {c['par2_A']:.1f} vs {c['par2_B']:.1f}" if key == 'sec' else ''), flush=True)
	if not v25:
		# residual after the lattice stage, per seed (F3): lp vs model vs pv on solved_by != 'ahl'
		for seed in [0, 1, 2]:
			lp = load(f"{a.root}/C_M0_S{seed}_ahlon_lp"); m0 = load(f"{a.root}/C_M0_S{seed}_ahlon_model"); pv = load(f"{a.root}/C_PV_S{seed}_ahlon_pv")
			res_keys = [k for k in lp if lp[k]['solved_by'] != 'ahl']
			sub = lambda X: {k: X[k] for k in res_keys if k in X}
			for nameA, XA, nameB, XB in [('lp', lp, 'model', m0), ('model', m0, 'pv', pv)]:
				c = compare(sub(XA), sub(XB), 600, 'sec')
				res[f"residual|S{seed}|{nameA} vs {nameB}|sec"] = c
				if c.get('common'):
					print(f"residual S{seed} {nameA} vs {nameB} sec: residual n={c['n']} median ratio {c['median_ratio']:.2f}x, A slower on {c['wins_A_larger']}/{c['n']-c['ties']} p={c['p_sign']:.3g}", flush=True)
	json.dump(res, open(a.out, 'w'), indent=1)
	print('saved', a.out)


if __name__ == '__main__':
	main()
