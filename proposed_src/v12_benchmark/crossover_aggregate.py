"""
Aggregate CP-SAT-pure (crossover_cpsat_pure.py) and proposed-pure
(LPneuroBLS_v9_infer_pure.py) results into one crossover-comparison table.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def load_cpsat(json_path):
	with open(json_path) as f:
		d = json.load(f)
	return d['summary']


def load_pure(out_dir):
	files = sorted(Path(out_dir).glob('*_result.json'))
	total = len(files)
	if total == 0:
		return None
	tp = tn = fp = fn = sol_correct = unresolved = 0
	gt_f = gt_i = 0
	times = []
	for f in files:
		with open(f) as fp:
			r = json.load(fp)
		gt = r['gt_feasible']
		pred = r['pred_feasible']
		gt_f += int(gt)
		gt_i += int(not gt)
		times.append(r['time_sec'])
		if pred is None:
			unresolved += 1
		elif gt and pred:
			tp += 1
		elif not gt and not pred:
			tn += 1
		elif not gt and pred:
			fp += 1
		else:
			fn += 1
		if gt and r['solved']:
			sol_correct += 1
	resolved = total - unresolved
	return {
		'total': total, 'gt_feasible': gt_f, 'gt_infeasible': gt_i,
		'unresolved': unresolved, 'resolved': resolved,
		'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
		'feasibility_acc_resolved': (tp + tn) / resolved * 100 if resolved else 0.0,
		'sol_correct': sol_correct, 'sol_acc': sol_correct / gt_f * 100 if gt_f else 0.0,
		'avg_time': float(np.mean(times)), 'total_time': float(np.sum(times)),
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sizes', nargs='+', required=True, help='e.g. 10x25 20x50 40x100 60x150 80x200')
	ap.add_argument('--cpsat_dir', default='../../runs/crossover')
	ap.add_argument('--pure_dir', default='../../runs/crossover')
	a = ap.parse_args()

	print(f"{'size':<10}{'CPSAT_acc%':>12}{'CPSAT_unres%':>14}{'CPSAT_avgT(s)':>15} | "
	      f"{'OURS_acc%':>12}{'OURS_unres%':>14}{'OURS_avgT(s)':>15}")
	print('-' * 100)
	for sz in a.sizes:
		cpsat_json = Path(a.cpsat_dir) / f"cpsat_{sz}.json"
		pure_out = Path(a.pure_dir) / f"pure_{sz}"
		c = load_cpsat(cpsat_json) if cpsat_json.exists() else None
		p = load_pure(pure_out) if pure_out.exists() else None
		c_acc = f"{c['feasibility_acc_resolved']:.1f}" if c else 'N/A'
		c_unres = f"{100*c['unresolved']/c['total']:.1f}" if c else 'N/A'
		c_t = f"{c['avg_time']:.3f}" if c else 'N/A'
		p_acc = f"{p['feasibility_acc_resolved']:.1f}" if p else 'N/A'
		p_unres = f"{100*p['unresolved']/p['total']:.1f}" if p else 'N/A'
		p_t = f"{p['avg_time']:.3f}" if p else 'N/A'
		print(f"{sz:<10}{c_acc:>12}{c_unres:>14}{c_t:>15} | {p_acc:>12}{p_unres:>14}{p_t:>15}")


if __name__ == '__main__':
	main()
