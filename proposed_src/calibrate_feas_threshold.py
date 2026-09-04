#!/usr/bin/env python3
"""
Sweep the feasibility-head threshold offline from saved inference result JSONs.

Reads per-instance results produced by LPneuroBLS_v5_infer.py (run with the
default threshold 0.5 on a *training-derived* calibration set) and simulates
feasibility/solution metrics for thresholds t >= 0.5. Valid because at t=0.5
every instance that reaches the head is searched, so solved status is known;
raising t can only flip head-gated instances to 'declared infeasible'.

Usage:
  python calibrate_feas_threshold.py --results_dir runs/.../infer_calib_ep50
"""
import argparse
import json
import numpy as np
from pathlib import Path

SYMBOLIC_INFEASIBLE = {'propagation_infeasible', 'lp_hard_rule'}
CERTIFICATE = {'kernel_pump'}


def load_results(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob('*_result.json')):
        d = json.load(open(f))
        rows.append(d)
    return rows


def metrics_at(rows, t):
    tp = tn = fp = fn = sol = gt_f = gt_i = 0
    for r in rows:
        gt = r['gt_feasible']
        m = r['method']
        if m in SYMBOLIC_INFEASIBLE:
            pred = False
        elif m in CERTIFICATE:
            pred = True
        else:
            pred = r['feas_prob'] >= t
        gt_f += int(gt)
        gt_i += int(not gt)
        if gt and pred:
            tp += 1
            sol += int(r['solved'])
        elif not gt and not pred:
            tn += 1
        elif not gt and pred:
            fp += 1
        else:
            fn += 1
    total = len(rows)
    return {
        't': t,
        'feas_acc': (tp + tn) / total * 100,
        'sol_acc': sol / max(1, gt_f) * 100,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--min_sol_acc', type=float, default=75.0,
                    help='Only consider thresholds keeping solution accuracy above this')
    args = ap.parse_args()

    rows = load_results(args.results_dir)
    print(f"{len(rows)} calibration instances loaded from {args.results_dir}")

    best = None
    print(f"{'t':>6} {'feas%':>8} {'sol%':>8} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
    for t in np.arange(0.50, 1.00, 0.02):
        r = metrics_at(rows, float(t))
        print(f"{r['t']:>6.2f} {r['feas_acc']:>8.2f} {r['sol_acc']:>8.2f} "
              f"{r['tp']:>5} {r['tn']:>5} {r['fp']:>5} {r['fn']:>5}")
        if r['sol_acc'] >= args.min_sol_acc and (best is None or r['feas_acc'] > best['feas_acc']):
            best = r
    if best is None:
        print("\nNo threshold satisfies the solution-accuracy floor; keeping 0.5")
    else:
        print(f"\nBest threshold on calibration set: t={best['t']:.2f} "
              f"(feas {best['feas_acc']:.2f}%, sol {best['sol_acc']:.2f}%)")


if __name__ == '__main__':
    main()
