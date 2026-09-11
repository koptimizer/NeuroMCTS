"""Find the m that puts |S| near a target at a given n, using only sound enumerations.

Uses the corrected completeness rule (OPTIMAL/INFEASIBLE only). Instances whose
enumeration is cut off by the time limit are excluded rather than counted, so a low
completion rate shows up as a small sample instead of a wrong median -- at n=100
enumeration frequently does not finish, and silently averaging truncated counts is
exactly the error this replaces.
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

from generate_hard_instances import gen_hard_feasible
from v15_bayes_limit import enumerate_solutions


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--n', type=int, required=True)
	ap.add_argument('--ms', type=int, nargs='+', required=True)
	ap.add_argument('--count', type=int, default=8)
	ap.add_argument('--cap', type=int, default=100000)
	ap.add_argument('--time_limit', type=float, default=120.0)
	ap.add_argument('--K', type=int, default=20)
	a = ap.parse_args()

	print(f"n={a.n}: |S| by m (sound completions only, limit {a.time_limit}s)", flush=True)
	print(f"{'m':>5}{'m/n':>7}{'|S| median':>13}{'complete':>11}{'mean sec':>10}", flush=True)
	for mm in a.ms:
		rng = np.random.default_rng(500 + mm + a.n)
		ns, oks, ts = [], 0, []
		for _ in range(a.count):
			A, b, x, _ = gen_hard_feasible(mm, a.n, rng, K=a.K)
			t0 = time.time()
			S, ok = enumerate_solutions(A, b, a.cap, a.time_limit)
			ts.append(time.time() - t0)
			if ok:
				ns.append(len(S)); oks += 1
		med = np.median(ns) if ns else float('nan')
		print(f"{mm:>5}{mm/a.n:>7.2f}{med:>13.0f}{oks:>8}/{a.count}{np.mean(ts):>10.1f}", flush=True)


if __name__ == '__main__':
	main()
