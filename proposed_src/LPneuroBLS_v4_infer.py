#!/usr/bin/env python3
"""
Inference harness for LPneuroBLS v4 (DNS — Defect-Noise Scenario).

Decision flow per instance:
  1. Propagate constraints at root (apply_probing).
     If is_invalid → definitively infeasible, skip model.
  2. Run feasibility_head → P(feasible).
     If P(feasible) < feas_threshold → declare infeasible, skip MCTS.
  3. (Optional) LP shortcut: if LP solution is already binary and correct, use it.
  4. MCTS search with 1-ply backtracking (same as v3).

Two-metric evaluation output (matches baseline_dns format):
  [1] Feasibility classification accuracy (TP/TN/FP/FN)
  [2] Solution accuracy among GT-feasible instances

Usage:
  python LPneuroBLS_v4_infer.py --checkpoint runs/.../checkpoint_ep70_RL_FA90_RE76.pt
  python LPneuroBLS_v4_infer.py --checkpoint ... --instances_dir ./test_instances_dns_10x25_1000
  python LPneuroBLS_v4_infer.py --checkpoint ... --instance path/to/instance.json
"""
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import LPneuroBLS_v4 as m


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def matrix_to_graph(A, device):
    rows, cols = np.where(A == 1)
    ev = torch.tensor(np.array([cols, rows]), dtype=torch.long, device=device)
    ec = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device)
    return ev, ec


def prepare_lp_tensors(A, b, device):
    x_lp, lp_is_feasible = m.solve_lp_relaxation(A, b)
    row_degrees = np.maximum(np.sum(A != 0, axis=1), 1)
    x_lp_check = (b - A.dot(x_lp)) / row_degrees
    return (
        torch.tensor(x_lp, dtype=torch.float, device=device),
        torch.tensor(x_lp_check, dtype=torch.float, device=device),
        x_lp,
        lp_is_feasible,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core inference function
# ──────────────────────────────────────────────────────────────────────────────

def infer_instance(model, mcts, A, b, x_lp_var, x_lp_check, device,
                   ev2c=None, ec2v=None, deterministic=True, feas_threshold=0.5,
                   lp_is_feasible=True):
    """
    Returns
    -------
    assignment    : list[int]  — final variable assignment (may contain -1 for unset)
    solved        : bool       — True if Ax=b is exactly satisfied
    pred_feasible : bool       — model's feasibility prediction
    feas_prob     : float      — P(feasible) from feasibility head
    trajectory    : list       — MCTS decision steps (empty when declared infeasible)
    is_invalid    : bool       — propagation detected contradiction
    """
    mcts.cache.clear()
    env = m.FastBinaryEnv(A, b)
    if ev2c is None or ec2v is None:
        ev2c, ec2v = matrix_to_graph(A, device)

    env.propagate_constraints().apply_probing()

    # Propagation proved infeasibility — no need to consult the model.
    if env.is_invalid:
        return env.assignment.tolist(), False, False, 0.0, [], True

    # Hard rule: LP infeasible → binary definitely infeasible (skip GNN entirely).
    # LP is a relaxation of the binary problem; LP infeasible implies no binary solution exists.
    if not lp_is_feasible:
        return env.assignment.tolist(), False, False, 0.0, [], env.is_invalid

    # ── Step 1: Feasibility prediction (GNN) ──────────────────────────────────
    with torch.no_grad():
        xv, xc = env.get_tensor_state(device)
        _, _, _, feas_logit = model(xv, xc, x_lp_var, x_lp_check, ev2c, ec2v)
        feas_prob = torch.sigmoid(feas_logit).item()
    pred_feasible = feas_prob >= feas_threshold

    if not pred_feasible:
        # Declared infeasible — skip MCTS entirely.
        return env.assignment.tolist(), False, pred_feasible, feas_prob, [], env.is_invalid

    # ── Step 2: Propagation may have resolved everything ──────────────────────
    if env.is_terminal():
        solved = bool(env.get_reward() == 1.0)
        return env.assignment.tolist(), solved, pred_feasible, feas_prob, [], env.is_invalid

    # ── Step 3: MCTS with 1-ply backtracking ─────────────────────────────────
    bt_stack  = []   # (assignment_snapshot, var, action)
    trajectory = []
    while True:
        out = mcts.run(env, ev2c, ec2v, x_lp_var, x_lp_check)
        if out[0] is None:
            break
        pi_var_full, pi_act, chosen_var, _, _, _ = out

        if deterministic:
            tgt_var = int(np.argmax(pi_var_full))
            tgt_act = int(np.argmax(pi_act))
        else:
            tgt_var = int(chosen_var)
            tgt_act = int(np.random.choice([0, 1], p=pi_act))

        pre_snap = env.assignment.copy()
        bt_stack.append((pre_snap, tgt_var, tgt_act))
        trajectory.append((tgt_var, tgt_act, pi_var_full.tolist(), pi_act.tolist()))

        env.step_inplace(tgt_var, tgt_act)
        env.propagate_constraints().apply_probing()

        if env.is_invalid:
            if bt_stack:
                prev_snap, bt_var, bt_act = bt_stack.pop()
                trajectory.pop()
                alt_act = 1 - bt_act
                env_alt = m.FastBinaryEnv(A, b, prev_snap.copy())
                env_alt.step_inplace(bt_var, alt_act)
                env_alt.propagate_constraints().apply_probing()
                if not env_alt.is_invalid:
                    # Recovered: continue from the flipped state.
                    env = env_alt
                    mcts.cache.clear()
                else:
                    break  # both values lead to contradiction
            else:
                break

    env.apply_local_search()
    solved = bool(env.get_accuracy() == 1.0)
    return env.assignment.tolist(), solved, pred_feasible, feas_prob, trajectory, env.is_invalid


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer  (matches baseline_dns format)
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(checkpoint, total, gt_f, gt_i,
                  tp, tn, fp, fn, sol_correct, times,
                  lp_hard_rule, lp_shortcut, mcts_invalid, feas_threshold):
    feasibility_acc = (tp + tn) / total * 100
    sol_acc         = sol_correct / gt_f * 100 if gt_f > 0 else 0.0
    avg_t           = float(np.mean(times))
    std_t           = float(np.std(times))
    w = 66
    print("\n" + "=" * w)
    print(f"Result Summary: LPneuroBLS v4")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Feas threshold: {feas_threshold}")
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
    print(f"  [3] Breakdown")
    print(f"      LP hard rule  : {lp_hard_rule:>6}  (LP infeasible → declared infeasible before GNN)")
    print(f"      LP shortcut   : {lp_shortcut:>6}  (LP solution directly binary and correct)")
    print(f"      MCTS invalid  : {mcts_invalid:>6}  (propagation contradiction during search)")
    print("-" * w)
    print(f"  Avg Time / Inst   : {avg_t:.4f} sec")
    print(f"  Std Time / Inst   : {std_t:.4f} sec")
    print(f"  Total Time        : {sum(times):.4f} sec")
    print("=" * w)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./runs/260624-2225_LP_Hybrid_v4/checkpoint_ep50_RE_FA80_RE63.pt')

    parser.add_argument('--instance',     type=str, default=None,
                        help='Single JSON instance file (DNS format)')
    parser.add_argument('--instances_dir', type=str, default='./test_instances_dns_10x25_1000',
                        help='Directory of JSON instances (DNS format)')
    parser.add_argument('--hidden_dim',   type=int,   default=256)
    parser.add_argument('--num_layers',   type=int,   default=8)
    parser.add_argument('--device',       type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_simulations', type=int, default=100)
    parser.add_argument('--c_puct',       type=float, default=1.5)
    parser.add_argument('--feas_threshold', type=float, default=0.5,
                        help='P(feasible) threshold for feasibility head (default: 0.5)')
    parser.add_argument('--stochastic',   action='store_true',
                        help='Use stochastic MCTS selection (default: deterministic argmax)')
    parser.add_argument('--out_dir',      type=str,   default='./inference_out_v4')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = m.BipartiteGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)

    ckpt = None
    for load_fn in [
        lambda: torch.load(args.checkpoint, map_location=device),
        lambda: torch.load(args.checkpoint, map_location=device, weights_only=False),
    ]:
        try:
            ckpt = load_fn()
            break
        except Exception:
            pass

    if ckpt is None:
        raise SystemExit(f"Failed to load checkpoint: {args.checkpoint}")

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Loaded: {args.checkpoint}")

    mcts = m.MCTS(model, device,
                  num_simulations=args.num_simulations,
                  c_puct=args.c_puct, training=False)
    deterministic = not args.stochastic

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect instance paths ──────────────────────────────────────────────────
    if args.instance:
        inst_paths = [Path(args.instance)]
    elif args.instances_dir:
        inst_paths = sorted(Path(args.instances_dir).glob('*.json'))
    else:
        raise SystemExit('Provide --instance or --instances_dir')

    if not inst_paths:
        raise SystemExit(f"No JSON files found in {args.instances_dir}")

    # ── Precompute LP relaxations and graphs ────────────────────────────────────
    print(f"Precomputing LP relaxations for {len(inst_paths)} instances...")
    precomputed = []
    for inst_path in tqdm(inst_paths):
        with open(inst_path, 'r') as f:
            d = json.load(f)
        A  = np.array(d['A'],  dtype=np.int32)
        b  = np.array(d['b'],  dtype=np.int32)
        gt_feasible = bool(d.get('feasible', True))
        x_gt = np.array(d['x'], dtype=np.int8) if d.get('x') is not None else None

        x_lp_var_ts, x_lp_check_ts, x_lp, lp_is_feasible = prepare_lp_tensors(A, b, device)
        ev2c, ec2v = matrix_to_graph(A, device)

        # LP shortcut: LP feasible AND solution is already binary and correct
        x_lp_rounded = np.round(x_lp)
        lp_direct = (
            lp_is_feasible and
            np.all(np.isin(x_lp_rounded, [0, 1])) and
            np.all(A.dot(x_lp_rounded.astype(np.int32)) == b)
        )

        precomputed.append({
            'path':           inst_path,
            'A':              A,
            'b':              b,
            'x_gt':           x_gt,
            'gt_feasible':    gt_feasible,
            'x_lp_var':       x_lp_var_ts,
            'x_lp_check':     x_lp_check_ts,
            'x_lp_rounded':   x_lp_rounded,
            'ev2c':           ev2c,
            'ec2v':           ec2v,
            'lp_direct':      lp_direct,
            'lp_is_feasible': lp_is_feasible,
        })

    # ── Inference loop ──────────────────────────────────────────────────────────
    total = len(precomputed)
    tp = tn = fp = fn = sol_correct = 0
    lp_hard_rule_count = 0
    lp_shortcut_count = 0
    mcts_invalid_count = 0
    times = []

    mode_str = 'stochastic' if args.stochastic else 'deterministic'
    print(f"\nRunning inference on {total} instances  (mode: {mode_str}, feas_threshold: {args.feas_threshold})")

    for info in tqdm(precomputed):
        t0 = time.time()
        A, b     = info['A'], info['b']
        gt_feasible = info['gt_feasible']

        # ── Run inference ────────────────────────────────────────────────────
        assignment, solved, pred_feasible, feas_prob, traj, is_invalid = infer_instance(
            model, mcts, A, b,
            info['x_lp_var'], info['x_lp_check'], device,
            ev2c=info['ev2c'], ec2v=info['ec2v'],
            deterministic=deterministic,
            feas_threshold=args.feas_threshold,
            lp_is_feasible=info['lp_is_feasible'],
        )

        # Determine method label and update counters
        if not info['lp_is_feasible'] and not pred_feasible:
            method = 'lp_hard_rule'        # LP proved infeasible — GNN skipped
            lp_hard_rule_count += 1
        elif pred_feasible:
            if not solved and info['lp_direct']:
                lp_shortcut_count += 1
                assignment = info['x_lp_rounded'].astype(np.int8).tolist()
                solved = True
                method = 'lp_shortcut'
            else:
                method = 'mcts'
                if is_invalid:
                    mcts_invalid_count += 1
        else:
            method = 'declared_infeasible'  # LP feasible but GNN predicted infeasible

        elapsed = time.time() - t0
        times.append(elapsed)

        # ── Feasibility classification ────────────────────────────────────────
        if gt_feasible and pred_feasible:
            tp += 1
        elif not gt_feasible and not pred_feasible:
            tn += 1
        elif not gt_feasible and pred_feasible:
            fp += 1
        else:
            fn += 1

        # ── Solution accuracy (GT feasible instances only) ────────────────────
        if gt_feasible and solved:
            assign_arr = np.array(assignment, dtype=np.int32)
            assign_arr = np.maximum(assign_arr, 0)
            if np.all(A.dot(assign_arr) == b):
                sol_correct += 1

        # ── Save per-instance result ──────────────────────────────────────────
        result = {
            'instance':     str(info['path']),
            'gt_feasible':  bool(gt_feasible),
            'pred_feasible': bool(pred_feasible),
            'feas_prob':    round(float(feas_prob), 6),
            'solved':       bool(solved),
            'is_invalid':   bool(is_invalid),
            'method':       method,
            'assignment':   assignment if pred_feasible else None,
            'trajectory_len': len(traj),
        }
        save_path = out_dir / (info['path'].stem + '_result.json')
        with open(save_path, 'w') as wf:
            json.dump(result, wf, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    gt_f = tp + fn
    gt_i = tn + fp
    print_summary(
        args.checkpoint, total, gt_f, gt_i,
        tp, tn, fp, fn, sol_correct, times,
        lp_hard_rule_count, lp_shortcut_count, mcts_invalid_count,
        args.feas_threshold,
    )


if __name__ == '__main__':
    main()
