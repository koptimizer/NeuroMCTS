"""Aggregate runs/v24 search results: per condition/seed solve counts, median time and nodes for
lp, M0, M1, M2, and paired comparisons (sign tests on time and nodes) of M0 vs lp and M1/M2 vs M0.
H_* is the 100-instance 21x60 set (AHL off only); C_*/B_* the 30-instance sets."""
import json, glob, numpy as np
from scipy.stats import binomtest

ROOT = '../../runs/v24'

def load(d): return {json.load(open(f))['instance']: json.load(open(f)) for f in glob.glob(d + '/inst-*_r.json')}
def arm(cond, tag, s, ahl, a): return load(f'{ROOT}/{cond}_{tag}_S{s}_ahl{ahl}_{a}')

def stats(R, resid_only):
	ks = [k for k in R if not resid_only or R[k]['solved_by'] != 'ahl']
	t = np.array([R[k]['total_sec'] for k in ks]); n = np.array([R[k]['nodes'] for k in ks])
	return sum(R[k]['solved'] for k in ks), len(ks), np.median(t), np.median(n), ks

def paired(Ra, Rb, ks):
	ta = np.array([Ra[k]['total_sec'] for k in ks]); tb = np.array([Rb[k]['total_sec'] for k in ks])
	na = np.array([Ra[k]['nodes'] for k in ks]); nb = np.array([Rb[k]['nodes'] for k in ks])
	w = int((tb < ta).sum()); wn = int((nb < na).sum()); tie = int((nb == na).sum())
	return (f"t {np.median(ta / np.maximum(tb, 1e-3)):.2f}x {w}/{len(ks)} p={binomtest(w, len(ks)).pvalue:.3f} | "
	        f"nodes {np.median(na / np.maximum(nb, 1)):.2f}x fewer {wn}/{len(ks) - tie} p={binomtest(wn, max(1, len(ks) - tie)).pvalue:.3f}")

for cond, ahls in [('C', ['off', 'on']), ('B', ['off', 'on']), ('H', ['off'])]:
	for ahl in ahls:
		print(f"\n===== {cond} ahl={ahl}")
		for s in [0, 1, 2]:
			arms = dict(lp=arm(cond, 'M0', s, ahl, 'lp'), M0=arm(cond, 'M0', s, ahl, 'model'),
			            M1=arm(cond, 'M1', s, ahl, 'model'), M2=arm(cond, 'M2', s, ahl, 'model'))
			if not all(arms.values()): continue
			ro = ahl == 'on'
			line = f"S{s}: "
			for nm, R in arms.items():
				sol, n, mt, mn, ks = stats(R, ro); line += f"{nm} {sol}/{n} t{mt:7.2f} n{mn:8.0f} | "
			ks = stats(arms['M0'], ro)[4]
			print(line)
			print(f"      M0 vs lp: {paired(arms['lp'], arms['M0'], ks)}")
			print(f"      M1 vs M0: {paired(arms['M0'], arms['M1'], ks)}")
			print(f"      M2 vs M0: {paired(arms['M0'], arms['M2'], ks)}")
