"""Lower-bound |S| at n=100, where exact enumeration is out of reach.

v22_find_m.py showed that at n=100 no enumeration completes within 120s for any m tried
(0/6 at m=20, 24, 28), so |S| there cannot be verified the way it was at n=25 and n=50.
This settles for what is still measurable: how many solutions CP-SAT finds in a fixed
window, which is a valid LOWER bound on |S| and enough to tell "a handful" apart from
"thousands" when choosing m for the test-only size.

Reported as a lower bound, never as |S| -- the whole reason this file exists is that the
earlier numbers were truncated counts being read as exact ones.
"""
# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

import argparse
import time
import numpy as np
from ortools.sat.python import cp_model

from generate_hard_instances import gen_hard_feasible
from v15_bayes_limit import SolutionCollector


def count_within(A, b, cap, tl):
	mm, n = A.shape
	model = cp_model.CpModel()
	xv = [model.NewBoolVar(f"x{j}") for j in range(n)]
	for i in range(mm):
		idx = np.where(A[i] == 1)[0]
		model.Add(sum(xv[j] for j in idx) == int(b[i]))
	s = cp_model.CpSolver()
	s.parameters.enumerate_all_solutions = True
	s.parameters.num_search_workers = 1
	s.parameters.max_time_in_seconds = tl
	col = SolutionCollector(xv, cap)
	st = s.Solve(model, col)
	return len(col.sols), s.StatusName(st)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--n', type=int, default=100)
	ap.add_argument('--ms', type=int, nargs='+', required=True)
	ap.add_argument('--count', type=int, default=5)
	ap.add_argument('--cap', type=int, default=5000)
	ap.add_argument('--time_limit', type=float, default=20.0)
	a = ap.parse_args()

	print(f"n={a.n}: solutions found within {a.time_limit}s (LOWER bound on |S|)", flush=True)
	print(f"{'m':>5}{'median found':>15}{'proved exact':>14}{'mean sec':>10}", flush=True)
	for mm in a.ms:
		rng = np.random.default_rng(600 + mm)
		cs, opt, ts = [], 0, []
		for _ in range(a.count):
			A, b, x, _ = gen_hard_feasible(mm, a.n, rng, K=20)
			t0 = time.time()
			c, st = count_within(A, b, a.cap, a.time_limit)
			ts.append(time.time() - t0)
			cs.append(c)
			opt += (st == 'OPTIMAL')
		print(f"{mm:>5}{np.median(cs):>15.0f}{opt:>11}/{a.count}{np.mean(ts):>10.1f}", flush=True)


if __name__ == '__main__':
	main()
