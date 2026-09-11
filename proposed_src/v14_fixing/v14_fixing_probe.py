"""Pre-RL probe for the variable-fixing direction, answering two questions raised
against the proposal before any policy code is written:

Q1 "why not let a cheap algorithm do the fixing and RL solve the smaller problem?"
    -> measures how many variables constraint propagation + probing actually fix at
       the root on hard instances. If it fixes almost nothing, the 'cheap fixing'
       half of that design has no material to work with (and the 'RL solves the rest'
       half is exactly the v6-v9 architecture already measured at 0% contribution).

Q2 "how do you decide how many variables to fix?"
    -> sweeps k and reports, per k, how often AHL then solves the reduced instance
       under three fixing rules that isolate the two failure sources:
         oracle_lpset : LP-confidence picks WHICH k, the planted solution gives values
         lp_conf      : LP-confidence picks WHICH k, LP rounding gives values
         oracle_rand  : random k variables, planted values (pure dimension-reduction)
       oracle_lpset vs lp_conf isolates value errors on an identical variable subset;
       oracle_rand shows how much of any gain is just "smaller n" independent of choice.

Decision rule this probe is meant to settle:
  * lp_conf ~= oracle  -> no RL needed, plain LP-rounding fixing is the answer
  * oracle >> lp_conf  -> that gap is the learnable headroom
  * oracle flat/no gain -> the whole direction dies here, before it costs a training run
"""
# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

import LPneuroBLS_v7 as m
from v13_pipeline import ahl_try_with_perm


def reduce_instance(A, b, fixed_idx, fixed_val):
	# substitute the fixed assignment out: A_keep x_keep = b - A_fixed v
	keep = np.setdiff1d(np.arange(A.shape[1]), fixed_idx)
	b_red = b - A[:, fixed_idx].dot(np.asarray(fixed_val, dtype=np.int64))
	return A[:, keep], b_red, keep


def ahl_solve_reduced(A, b, block, tries, rng):
	n = A.shape[1]
	if n == 0:
		return bool(np.all(b == 0))
	if np.any(b < 0) or np.any(b > A.sum(axis=1)):
		return False   # substitution already made a row unsatisfiable
	for t in range(tries):
		perm = np.arange(n) if t == 0 else rng.permutation(n)
		if ahl_try_with_perm(A, b, perm, block) is not None:
			return True
	return False


def probe_one(task):
	idx, path, block, tries, ks, seed = task
	d = json.load(open(path))
	if not d.get('feasible', True) or d.get('x') is None:
		return idx, None
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	x_gt = np.array(d['x'], dtype=np.int64)
	n = A.shape[1]
	rng = np.random.default_rng(seed)

	# Q1: how much does the cheap symbolic stage fix on its own?
	env = m.FastBinaryEnv(A, b)
	env.propagate_constraints().apply_probing()
	n_prop_fixed = int((env.assignment != -1).sum())

	x_lp, lp_ok = m.solve_lp_relaxation(A, b)
	if not lp_ok:
		return idx, None
	conf = -np.abs(x_lp - np.round(x_lp))       # most confident first
	order_conf = np.argsort(-conf)

	rec = dict(instance=Path(path).name, n=n, n_prop_fixed=n_prop_fixed, by_k={})
	# k = 0 is the unmodified instance: the reference success rate
	for k in ks:
		if k >= n:
			continue
		sel = order_conf[:k]
		lp_vals = np.round(np.clip(x_lp[sel], 0, 1)).astype(np.int64)
		gt_vals = x_gt[sel]
		n_val_err = int((lp_vals != gt_vals).sum())

		Ar, br, _ = reduce_instance(A, b, sel, gt_vals)
		ok_oracle = ahl_solve_reduced(Ar, br, block, tries, np.random.default_rng(seed + 1))

		Ar, br, _ = reduce_instance(A, b, sel, lp_vals)
		ok_lp = ahl_solve_reduced(Ar, br, block, tries, np.random.default_rng(seed + 2))

		rsel = rng.choice(n, k, replace=False) if k > 0 else np.array([], dtype=np.int64)
		Ar, br, _ = reduce_instance(A, b, rsel, x_gt[rsel])
		ok_orand = ahl_solve_reduced(Ar, br, block, tries, np.random.default_rng(seed + 3))

		rec['by_k'][k] = dict(oracle_lpset=ok_oracle, lp_conf=ok_lp,
		                       oracle_rand=ok_orand, n_val_err=n_val_err)
	return idx, rec


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', required=True)
	ap.add_argument('--block', type=int, required=True)
	ap.add_argument('--tries', type=int, default=10)
	ap.add_argument('--ks', type=int, nargs='+', required=True)
	ap.add_argument('--limit', type=int, default=20)
	ap.add_argument('--jobs', type=int, default=8)
	ap.add_argument('--out_json', default=None)
	a = ap.parse_args()

	files = sorted(Path(a.data_dir).glob('*.json'))[:a.limit]
	tasks = [(i, str(f), a.block, a.tries, a.ks, 5000 + i) for i, f in enumerate(files)]
	print(f"probing {len(tasks)} instances, block={a.block}, tries={a.tries}, ks={a.ks}", flush=True)

	out = []
	t0 = time.time()
	with Pool(a.jobs) as pool:
		for idx, rec in pool.imap_unordered(probe_one, tasks, chunksize=1):
			if rec is None:
				continue
			out.append(rec)
			print(f"  [{len(out)}] {rec['instance']} n={rec['n']} "
			       f"prop_fixed={rec['n_prop_fixed']} ({time.time()-t0:.0f}s)", flush=True)

	if not out:
		print("no usable instances")
		return

	print("\n=== Q1: cheap symbolic fixing at the root ===")
	pf = np.array([r['n_prop_fixed'] for r in out])
	nn = np.array([r['n'] for r in out])
	print(f"  propagation+probing fixes {pf.mean():.1f} / {nn.mean():.0f} variables on average "
	       f"({100*pf.mean()/nn.mean():.1f}% of n); max={pf.max()}, instances with 0 fixed={int((pf==0).sum())}/{len(pf)}")

	print("\n=== Q2: does fixing k variables make AHL succeed? ===")
	print(f"  {'k':>4} {'oracle(LP-set)':>16} {'lp_conf':>10} {'oracle(rand)':>14} {'mean val errors':>16}")
	for k in a.ks:
		rows = [r['by_k'][k] for r in out if k in r['by_k']]
		if not rows:
			continue
		o = np.mean([x['oracle_lpset'] for x in rows])
		l = np.mean([x['lp_conf'] for x in rows])
		orr = np.mean([x['oracle_rand'] for x in rows])
		ve = np.mean([x['n_val_err'] for x in rows])
		print(f"  {k:>4} {100*o:>15.1f}% {100*l:>9.1f}% {100*orr:>13.1f}% {ve:>15.2f}")

	if a.out_json:
		with open(a.out_json, 'w') as f:
			json.dump(out, f, indent=2)
		print(f"\nsaved: {a.out_json}")


if __name__ == '__main__':
	main()
