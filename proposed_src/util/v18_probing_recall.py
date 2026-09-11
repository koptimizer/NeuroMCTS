"""v18b: can classical LP-probing recover the FORCED variables that propagation misses?

v18 showed the residual gap at depth 5-8 is entirely on FORCED variables (logically
determined by the prefix) and that unit propagation detects only 0.9-7.5% of them, while
LP integrality already catches 66-77%. Probing is the classical tool aimed exactly at
this: tentatively assign the opposite value, and if the result is infeasible the original
value was forced.

Two probing strengths are measured against the ground-truth FORCED set:
  prop-probe : assign the opposite value, run constraint propagation, look for a wipeout
  LP-probe   : assign the opposite value, re-solve the LP relaxation, look for infeasibility

If LP-probe recall is high, the remaining information is classically extractable and the
answer to the depth 6-7 gap is a stronger propagator, not a better learned predictor.
Cost is reported too, since probing costs one solve per candidate variable.
"""
import argparse
import json
import time
import numpy as np

import LPneuroBLS_v7 as m

EPS = 1e-9


def prop_probe(A, b, j, val):
	# forced if assigning the opposite value wipes out under propagation
	env = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
	env.step_inplace(int(j), int(val))
	env.propagate_constraints().apply_probing()
	return bool(env.is_invalid)


def lp_probe(A, b, j, val):
	# forced if assigning the opposite value makes even the LP relaxation infeasible
	keep = np.setdiff1d(np.arange(A.shape[1]), [j])
	b_red = b - A[:, j] * val
	if np.any(b_red < 0) or np.any(b_red > A[:, keep].sum(axis=1)):
		return True
	_, ok = m.solve_lp_relaxation(A[:, keep].astype(np.int32), b_red.astype(np.int32))
	return not ok


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--test', required=True)
	ap.add_argument('--depths', type=int, nargs='+', default=[5, 6, 7, 8])
	ap.add_argument('--per_depth', type=int, default=60)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	recs = json.load(open(a.test))
	by_depth = {d: [r for r in recs if r['depth'] == d][:a.per_depth] for d in a.depths}

	print("=" * 96)
	print("v18b: recall on FORCED variables -- can probing find what propagation misses?")
	print("=" * 96)
	print(f"{'depth':>6}{'states':>8}{'%FORCED':>10}{'prop-probe':>13}{'LP-probe':>11}"
	      f"{'LP integral':>13}{'union':>9}{'LP-probe cost':>16}")
	table = {}
	for d in a.depths:
		rec_list = by_depth[d]
		if not rec_list:
			continue
		fr, rc_pp, rc_lp, rc_int, rc_un, costs = [], [], [], [], [], []
		for r in rec_list:
			A = np.array(r['A'], dtype=np.int64)
			b = np.array(r['b'], dtype=np.int64)
			p_true = np.array(r['marginals'])
			forced = (p_true < EPS) | (p_true > 1 - EPS)
			if forced.sum() == 0:
				continue
			forced_val = (p_true > 0.5).astype(np.int64)
			x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
			lp_int = ((x_lp < 1e-6) | (x_lp > 1 - 1e-6)) if ok else np.zeros(len(p_true), bool)

			det_pp = np.zeros(len(p_true), bool)
			det_lp = np.zeros(len(p_true), bool)
			t0 = time.time()
			for j in range(A.shape[1]):
				opp = 1 - forced_val[j]
				det_pp[j] = prop_probe(A, b, j, opp)
				det_lp[j] = lp_probe(A, b, j, opp)
			costs.append((time.time() - t0) / max(1, A.shape[1]))

			fr.append(forced.mean())
			rc_pp.append((det_pp & forced).sum() / forced.sum())
			rc_lp.append((det_lp & forced).sum() / forced.sum())
			rc_int.append((lp_int & forced).sum() / forced.sum())
			rc_un.append(((det_lp | lp_int) & forced).sum() / forced.sum())

		row = dict(frac_forced=float(np.mean(fr)), prop_probe=float(np.mean(rc_pp)),
		            lp_probe=float(np.mean(rc_lp)), lp_integral=float(np.mean(rc_int)),
		            union=float(np.mean(rc_un)), ms_per_var=float(1000 * np.mean(costs)))
		table[d] = row
		print(f"{d:>6}{len(fr):>8}{100*row['frac_forced']:>9.1f}%{100*row['prop_probe']:>12.1f}%"
		      f"{100*row['lp_probe']:>10.1f}%{100*row['lp_integral']:>12.1f}%"
		      f"{100*row['union']:>8.1f}%{row['ms_per_var']:>13.2f} ms")
	print("\nunion = LP-probe OR LP-integral; this is what a purely classical node")
	print("evaluator could flag as forced, with no learning involved.")
	json.dump(table, open(a.out, 'w'), indent=2)
	print(f"saved: {a.out}")


if __name__ == '__main__':
	main()
