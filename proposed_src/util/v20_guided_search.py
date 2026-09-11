"""v20: does the model's 20x cost advantage actually translate into better search?

v19 measured cost and accuracy at a single node. This runs the search itself and compares
guidance strategies under a MATCHED WALL-CLOCK budget, which is the only comparison that
respects the tradeoff: probing is sounder per node but pays ~33ms, the network answers in
~1.6ms, so in equal time the network gets to look at roughly 20x more nodes.

AHL is deliberately excluded. The pipeline's solve rate is normally dominated by lattice
reduction, which would swamp any difference in branching quality; stripping it down to
propagation + branching isolates exactly the thing under test.

Arms (identical search, different guidance):
  random   : control -- arbitrary variable, arbitrary value
  lp       : branch on the most integral LP value, try that value first   (~3 ms/node)
  lp_probe : LP-probe every free variable, soundly FIX the forced ones,
             then branch as in lp                                          (~33 ms/node)
  model    : v17 conditional network; branch on its most confident variable (~1.6 ms/node)

Reported: solve rate at the budget, and median nodes / seconds to solution among solved.
"""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Process
import numpy as np
import torch

import LPneuroBLS_v7 as m

torch.set_num_threads(1)


def guidance_random(A, b, rng, model, device):
	n = A.shape[1]
	j = int(rng.integers(n))
	return j, int(rng.integers(2)), []


def guidance_lp(A, b, rng, model, device):
	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if not ok:
		return None, None, []
	j = int(np.argmax(np.abs(x_lp - 0.5)))
	return j, int(x_lp[j] >= 0.5), []


def guidance_lp_probe(A, b, rng, model, device):
	from v18_probing_recall import lp_probe
	n = A.shape[1]
	x_lp, ok = m.solve_lp_relaxation(A.astype(np.int32), b.astype(np.int32))
	if not ok:
		return None, None, []
	forced = []
	for j in range(n):
		# if assigning 0 is impossible the variable is forced to 1, and vice versa
		if lp_probe(A, b, j, 0):
			forced.append((j, 1))
		elif lp_probe(A, b, j, 1):
			forced.append((j, 0))
	if forced:
		return None, None, forced
	j = int(np.argmax(np.abs(x_lp - 0.5)))
	return j, int(x_lp[j] >= 0.5), []


def guidance_model(A, b, rng, model, device):
	from v15_train_marginal import featurize
	rec = dict(A=A.tolist(), b=b.tolist(), x=[0] * A.shape[1], marginals=[0.5] * A.shape[1])
	f = featurize(rec, device)
	if f is None:
		return None, None, []
	with torch.no_grad():
		p = torch.sigmoid(model(f['xv'], f['xc'], f['ev2c'], f['ec2v'])).numpy()
	j = int(np.argmax(np.abs(p - 0.5)))
	return j, int(p[j] >= 0.5), []


GUIDES = dict(random=guidance_random, lp=guidance_lp,
               lp_probe=guidance_lp_probe, model=guidance_model)


def search(A0, b0, guide, model, device, time_limit, seed):
	"""DFS over the reduced instance; returns (solved, nodes, elapsed)."""
	rng = np.random.default_rng(seed)
	t0 = time.time()
	nodes = 0
	# each frame: (A, b, assignment-so-far as list of (orig_idx, val), alive original indices)
	stack = [(A0.copy(), b0.copy(), [], np.arange(A0.shape[1]))]
	while stack:
		if time.time() - t0 > time_limit:
			return False, nodes, time.time() - t0
		A, b, fixed, alive = stack.pop()
		nodes += 1

		if np.any(b < 0) or np.any(b > A.sum(axis=1)):
			continue
		# reduction can leave all-zero rows; they are vacuous once b hits 0 (the
		# bound check above already rejected the infeasible case) and SCIP rejects
		# the empty constraint outright, so drop them before any LP call
		live = A.sum(axis=1) > 0
		if not live.all():
			A, b = A[live], b[live]
			if A.shape[0] == 0:
				# no constraints remain, so any completion works
				return True, nodes, time.time() - t0
		env = m.FastBinaryEnv(A.astype(np.int32), b.astype(np.int32))
		env.propagate_constraints()
		if env.is_invalid:
			continue
		det = env.assignment
		if np.all(det != -1):
			if np.all(A.dot(det.astype(np.int64)) == b):
				return True, nodes, time.time() - t0
			continue
		if np.any(det != -1):
			# fold in whatever propagation settled before consulting the guide
			idx = np.where(det != -1)[0]
			vals = det[idx].astype(np.int64)
			keep = np.setdiff1d(np.arange(A.shape[1]), idx)
			stack.append((A[:, keep], b - A[:, idx].dot(vals),
			               fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep]))
			continue

		j, val, forced = guide(A, b, rng, model, device)
		if forced:
			idx = np.array([f[0] for f in forced])
			vals = np.array([f[1] for f in forced], dtype=np.int64)
			keep = np.setdiff1d(np.arange(A.shape[1]), idx)
			stack.append((A[:, keep], b - A[:, idx].dot(vals),
			               fixed + list(zip(alive[idx].tolist(), vals.tolist())), alive[keep]))
			continue
		if j is None:
			continue
		keep = np.setdiff1d(np.arange(A.shape[1]), [j])
		for v in (1 - val, val):        # push the preferred value last so it pops first
			stack.append((A[:, keep], b - A[:, j] * v,
			               fixed + [(int(alive[j]), int(v))], alive[keep]))
	return False, nodes, time.time() - t0


def _worker(arm, inst_path, out_path, ckpt, time_limit):
	from v15_train_marginal import MarginalNet
	device = torch.device('cpu')
	model = None
	if arm == 'model':
		model = MarginalNet()
		st = torch.load(ckpt, map_location=device, weights_only=False)
		model.load_state_dict(st['model_state_dict'])
		model.eval()
	d = json.load(open(inst_path))
	A = np.array(d['A'], dtype=np.int64)
	b = np.array(d['b'], dtype=np.int64)
	gt = bool(d.get('feasible', True))
	solved, nodes, el = search(A, b, GUIDES[arm], model, device, time_limit,
	                            abs(hash(Path(inst_path).name)) % (2 ** 31))
	json.dump(dict(instance=Path(inst_path).name, gt_feasible=gt, solved=bool(solved),
	                nodes=int(nodes), time_sec=round(el, 3)), open(out_path, 'w'))


def run_arm(arm, data_dir, out_dir, ckpt, time_limit, jobs):
	files = sorted(Path(data_dir).glob('*.json'))
	out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
	pending, running, done = list(files), [], 0
	t0 = time.time()
	while pending or running:
		while pending and len(running) < jobs:
			fp = pending.pop(0)
			op = out_dir / (fp.stem + '_r.json')
			p = Process(target=_worker, args=(arm, str(fp), str(op), ckpt, time_limit))
			p.start(); running.append(dict(p=p, fp=fp, op=op, s=time.time()))
		time.sleep(1)
		still = []
		for r in running:
			if r['p'].is_alive():
				if time.time() - r['s'] > time_limit + 60:
					r['p'].kill(); r['p'].join()
					json.dump(dict(instance=r['fp'].name, gt_feasible=True, solved=False,
					                nodes=-1, time_sec=time_limit), open(r['op'], 'w'))
					done += 1
				else:
					still.append(r)
			else:
				r['p'].join(); done += 1
		running = still
	print(f"  [{arm}] {done}/{len(files)} in {time.time()-t0:.0f}s", flush=True)


def summarize(out_dir):
	rows = [json.load(open(f)) for f in sorted(Path(out_dir).glob('*_r.json'))]
	feas = [r for r in rows if r['gt_feasible']]
	solved = [r for r in feas if r['solved']]
	return dict(n_feas=len(feas), n_solved=len(solved),
	             rate=len(solved) / max(1, len(feas)),
	             med_nodes=float(np.median([r['nodes'] for r in solved])) if solved else float('nan'),
	             med_time=float(np.median([r['time_sec'] for r in solved])) if solved else float('nan'))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', required=True)
	ap.add_argument('--time_limit', type=float, required=True)
	ap.add_argument('--arms', nargs='+', default=['random', 'lp', 'lp_probe', 'model'])
	ap.add_argument('--ckpt', default='../../runs/v17/conditional.pt')
	ap.add_argument('--jobs', type=int, default=6)
	ap.add_argument('--out_root', default='../../runs/v20')
	a = ap.parse_args()

	res = {}
	for arm in a.arms:
		od = f"{a.out_root}/{a.size}_{arm}"
		print(f"=== {arm} {a.size} (budget {a.time_limit}s/instance) ===", flush=True)
		run_arm(arm, f"../../instances/bench30_{a.size}", od, a.ckpt, a.time_limit, a.jobs)
		res[arm] = summarize(od)

	print("\n" + "=" * 78)
	print(f"v20 guided search, {a.size}, matched budget {a.time_limit}s/instance")
	print("=" * 78)
	print(f"{'arm':<12}{'solve rate':>14}{'median nodes':>15}{'median sec':>13}")
	for arm in a.arms:
		r = res[arm]
		print(f"{arm:<12}{100*r['rate']:>12.1f}% ({r['n_solved']}/{r['n_feas']})"
		      f"{r['med_nodes']:>13.0f}{r['med_time']:>13.2f}")
	json.dump(res, open(f"{a.out_root}/summary_{a.size}.json", 'w'), indent=2)


if __name__ == '__main__':
	main()
