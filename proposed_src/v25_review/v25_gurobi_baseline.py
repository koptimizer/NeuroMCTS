"""v25 B4: Gurobi as a complete solver (feasibility MIP, single thread, same instances and budgets
as the other solvers; time counts Solve only, as in v22_classical_baseline.py).
Output: runs/v25/gurobi_<set>.json. Run from proposed_src/v25_review.
"""
import argparse, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def one(task):
	path, tl = task
	d = json.load(open(path)); A = np.array(d['A'], dtype=np.int64); b = np.array(d['b'], dtype=np.int64)
	env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.setParam('Threads', 1); env.setParam('TimeLimit', tl); env.start()
	mo = gp.Model(env=env); x = mo.addMVar(A.shape[1], vtype=GRB.BINARY); mo.addConstr(A.astype(float) @ x == b.astype(float)); mo.setObjective(0)
	t0 = time.time(); mo.optimize(); el = time.time() - t0
	ok = mo.Status == GRB.OPTIMAL and mo.SolCount > 0
	if ok:
		xs = np.rint(x.X).astype(np.int64); ok = bool(np.array_equal(A.dot(xs), b))
	mo.dispose(); env.dispose()
	return dict(instance=Path(path).name, solved=bool(ok), time_sec=round(el, 4), status=int(mo.Status if False else 0))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sets', nargs='+', default=['v22test_10x25', 'v22test_18x50', 'v24test_21x60_100', 'v24test_24x70', 'v24test_28x80'])
	ap.add_argument('--budgets', nargs='+', type=float, default=[60, 300, 600, 1200, 1200])
	ap.add_argument('--jobs', type=int, default=6)
	a = ap.parse_args()
	for s, tl in zip(a.sets, a.budgets):
		fs = sorted(Path(f'../../instances/{s}').glob('*.json'))
		with Pool(a.jobs) as pool:
			rows = pool.map(one, [(str(f), tl) for f in fs])
		sol = [r for r in rows if r['solved']]
		print(f"gurobi {s:20s} solved {len(sol)}/{len(rows)}  median {np.median([r['time_sec'] for r in sol]) if sol else float('nan'):.3f}s  max {max([r['time_sec'] for r in sol]) if sol else 0:.1f}s", flush=True)
		json.dump(rows, open(f'../../runs/v25/gurobi_{s}.json', 'w'), indent=1)


if __name__ == '__main__':
	main()
