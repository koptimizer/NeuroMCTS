"""
LLL (fpylll) baseline for DNS (Defect-Noise Scenario).

Infeasibility detection:
  LLL finds short lattice vectors via embedding.  When no binary solution exists,
  no candidate x will satisfy Ax=b, so the solver returns None.
  We then declare infeasible — this is HEURISTIC: "could not find solution" ≠
  "provably infeasible".  Feasible instances where LLL fails also map to FN.
  Expect lower feasibility accuracy than exact solvers.
"""
import json
import time
import numpy as np
from pathlib import Path

try:
    from fpylll import IntegerMatrix, LLL
except ImportError:
    raise SystemExit("fpylll not found. Install with: pip install fpylll")


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

    Solution found → (True, solution)
    No solution    → (False, None)  [heuristic: may be wrong for hard feasible instances]
    """
    m, n = A.shape
    K = int(np.sqrt(n)) + 100

    top    = np.hstack([K * A.T,                np.eye(n, dtype=int)])
    bottom = np.hstack([-K * b.reshape(1, -1),  np.zeros((1, n), dtype=int)])
    basis  = np.vstack([top, bottom])

    M = IntegerMatrix.from_matrix(basis.tolist())
    LLL.reduction(M)

    for i in range(M.nrows):
        row = np.array(M[i])
        check, cand = row[:m], row[m:]
        if np.any(check):
            continue
        for x in (cand, -cand):
            if np.all(np.isin(x, [0, 1])) and np.array_equal(A.dot(x.astype(np.int32)), b):
                return True, x.astype(np.int8)

    return False, None  # heuristic infeasibility declaration


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

    print(f"\n[LLL (fpylll)] {total} instances  |  dir: {data_dir}")
    print("  Infeasibility detection: heuristic (no solution found → declared infeasible)")

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
        "LLL (fpylll)", total, gt_f, gt_i, tp, tn, fp, fn, sol_correct, times,
        note="infeasibility = heuristic (FN possible for hard feasible instances)",
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = p.parse_args()
    run(args.data_dir)
