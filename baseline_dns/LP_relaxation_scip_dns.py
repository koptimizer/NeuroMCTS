"""
SCIP LP Relaxation baseline for DNS (Defect-Noise Scenario).

Infeasibility detection capability:
  - LP INFEASIBLE  → binary system is definitely infeasible
  - LP OPTIMAL     → round x_lp; if Ax̂=b declare feasible+found, else feasible-unsolved
                     (LP cannot prove binary infeasibility in this case)

Expect partial infeasibility detection: misses infeasible cases where the LP
relaxation still has a fractional feasible solution.
"""
import json
import time
import numpy as np
from pathlib import Path

try:
    from pyscipopt import Model, quicksum
except ImportError:
    raise SystemExit("pyscipopt not found. Install with: conda install -c conda-forge pyscipopt")


__DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = str(__DIR / "test_instances_dns_10x25_1000")


def load_data(data_dir):
    files = sorted(Path(data_dir).glob("*.json"))
    data = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        A           = np.array(d["A"], dtype=np.int32)
        b           = np.array(d["b"], dtype=np.int32)
        x_gt        = np.array(d["x"], dtype=np.int8) if d.get("x") is not None else None
        gt_feasible = bool(d.get("feasible", True))
        data.append((A, b, x_gt, gt_feasible))
    return data


def solve(A, b):
    """
    Returns (pred_feasible: bool, pred_solution: ndarray | None).

    LP infeasible        → (False, None)
    LP optimal, Ax̂=b    → (True,  x̂)
    LP optimal, Ax̂≠b    → (True,  None)
    """
    m, n = A.shape
    model = Model("dns_lp")
    model.hideOutput()
    x_vars = [model.addVar(name=f"x{j}", vtype="C", lb=0.0, ub=1.0) for j in range(n)]

    for i in range(m):
        idx = np.where(A[i] == 1)[0]
        model.addCons(quicksum(x_vars[j] for j in idx) == int(b[i]))

    model.optimize()
    status = model.getStatus()

    if status == "infeasible":
        result = (False, None)
    elif status == "optimal":
        x_val = np.array([model.getVal(v) for v in x_vars])
        x_round = np.round(x_val).astype(np.int8)
        if np.all(A.dot(x_round.astype(np.int32)) == b):
            result = (True, x_round)
        else:
            result = (True, None)
    else:
        result = (True, None)

    model.freeProb()
    return result


def print_summary(method, total, gt_f, gt_i, tp, tn, fp, fn,
                  sol_correct, times, note=""):
    feasibility_acc = (tp + tn) / total * 100
    sol_acc = sol_correct / gt_f * 100 if gt_f > 0 else 0.0
    avg_t = float(np.mean(times))
    std_t = float(np.std(times))
    w = 62
    print("\n" + "=" * w)
    print(f"Result Summary: {method}")
    if note:
        print(f"  NOTE: {note}")
    print("=" * w)
    print(f"  Total Instances   : {total:>6}")
    print(f"  GT Feasible       : {gt_f:>6}  ({100*gt_f/total:.1f}%)")
    print(f"  GT Infeasible     : {gt_i:>6}  ({100*gt_i/total:.1f}%)")
    print("-" * w)
    print(f"  [1] Feasibility Classification Accuracy")
    print(f"      Overall       : {tp+tn:>6} / {total}  ({feasibility_acc:.2f}%)")
    print(f"      TP  (F → F)   : {tp:>6} / {gt_f}")
    print(f"      TN  (I → I)   : {tn:>6} / {gt_i}")
    print(f"      FP  (I → F)   : {fp:>6} / {gt_i}")
    print(f"      FN  (F → I)   : {fn:>6} / {gt_f}")
    print("-" * w)
    print(f"  [2] Solution Accuracy  (GT Feasible instances only)")
    print(f"      Correct Sol   : {sol_correct:>6} / {gt_f}  ({sol_acc:.2f}%)")
    print("-" * w)
    print(f"  Avg Time / Inst   : {avg_t:.4f} sec")
    print(f"  Std Time / Inst   : {std_t:.4f} sec")
    print(f"  Total Time        : {sum(times):.4f} sec")
    print("=" * w)


def run(data_dir=DEFAULT_DATA_DIR):
    data = load_data(data_dir)
    total = len(data)
    if total == 0:
        raise SystemExit(f"No JSON files found in {data_dir}")

    print(f"\n[SCIP LP Relaxation] {total} instances  |  dir: {data_dir}")
    print("  Infeasibility detection: LP-infeasible cases only (partial coverage)")

    tp = tn = fp = fn = sol_correct = 0
    times = []

    for idx, (A, b, _, gt_feasible) in enumerate(data):
        t0 = time.perf_counter()
        pred_feasible, pred_sol = solve(A, b)
        times.append(time.perf_counter() - t0)

        if gt_feasible and pred_feasible:
            tp += 1
        elif not gt_feasible and not pred_feasible:
            tn += 1
        elif not gt_feasible and pred_feasible:
            fp += 1
        else:
            fn += 1

        if gt_feasible and pred_sol is not None and np.all(A.dot(pred_sol.astype(np.int32)) == b):
            sol_correct += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{total}]  feasibility acc: "
                  f"{(tp+tn)/(idx+1)*100:.1f}%  "
                  f"sol acc: {sol_correct/max(1,tp+fn)*100:.1f}%")

    gt_f = tp + fn
    gt_i = tn + fp
    print_summary(
        "SCIP LP Relaxation", total, gt_f, gt_i, tp, tn, fp, fn, sol_correct, times,
        note="FP expected when LP is feasible but binary system is not",
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = p.parse_args()
    run(args.data_dir)
