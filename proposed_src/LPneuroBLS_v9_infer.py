#!/usr/bin/env python3
"""
Inference harness for LPneuroBLS v9 (adds AHL lattice-reduction stage) (DNS — Defect-Noise Scenario).

Decision flow per instance:
  1. Propagate constraints at root (apply_probing).
     If is_invalid → definitively infeasible, skip model.
  2. LP hard rule: LP infeasible → binary infeasible (relaxation argument).
  3. Kernel pump: alternate rounding with projection onto {x : Ax = b}
     (null-space walk from the LP point). Success is a constructive
     feasibility certificate — model is skipped entirely.
  4. Feasibility head → P(feasible). Below threshold → declared infeasible.
  5. MCTS search with 1-ply backtracking; on failure, one pump repair
     attempt from the MCTS terminal point.

Two-metric evaluation output (matches baseline_dns format):
  [1] Feasibility classification accuracy (TP/TN/FP/FN)
  [2] Solution accuracy among GT-feasible instances

Reported per-instance time includes LP relaxation + pseudo-inverse +
graph construction (prep), so comparisons against baselines are fair.

Usage:
  python LPneuroBLS_v5_infer.py --checkpoint runs/.../checkpoint_ep50_RL_FA90_RE76.pt
  python LPneuroBLS_v5_infer.py --checkpoint ... --instances_dir ./test_instances_dns_10x25_1000
  python LPneuroBLS_v5_infer.py --checkpoint ... --instance path/to/instance.json
"""
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import LPneuroBLS_v7 as m
from lattice_enum import ahl_solve


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
    col_deg = (A != 0).sum(axis=0) / max(1, A.shape[0])
    xlp_v_arr = np.stack([x_lp, col_deg], axis=-1)
    return (
        torch.tensor(xlp_v_arr, dtype=torch.float, device=device),
        torch.tensor(x_lp_check, dtype=torch.float, device=device),
        x_lp,
        lp_is_feasible,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core inference function
# ──────────────────────────────────────────────────────────────────────────────

def infer_instance(model, mcts, A, b, x_lp_var, x_lp_check, x_lp, A_pinv, device,
                   ev2c, ec2v, deterministic=True, feas_threshold=0.5,
                   lp_is_feasible=True, pump_iters=200, dfs_budget=20000, mp_iters=None,
                   ahl_block=40, ahl_tries=10):
    """
    Returns
    -------
    assignment    : list[int]  — final variable assignment (may contain -1 for unset)
    solved        : bool       — True if Ax=b is exactly satisfied
    pred_feasible : bool       — feasibility decision (constructive or model)
    feas_prob     : float      — P(feasible); 1.0 when a certificate was found
    trajectory    : list       — MCTS decision steps (empty when MCTS skipped)
    is_invalid    : bool       — propagation detected contradiction
    method        : str        — which stage produced the decision
    """
    mcts.cache.clear()
    env = m.FastBinaryEnv(A, b)
    env.propagate_constraints().apply_probing()

    # Propagation proved infeasibility — no need to consult the model.
    if env.is_invalid:
        return env.assignment.tolist(), False, False, 0.0, [], True, 'propagation_infeasible'

    # Hard rule: LP infeasible → binary definitely infeasible (skip GNN entirely).
    if not lp_is_feasible:
        return env.assignment.tolist(), False, False, 0.0, [], False, 'lp_hard_rule'

    # Kernel pump from the LP point: constructive feasibility certificate on success.
    x_pump, pumped, pstats = m.kernel_pump_stats(x_lp, A, b, A_pinv, max_iter=pump_iters)
    if pumped:
        return x_pump.tolist(), True, True, 1.0, [], False, 'kernel_pump'

    # AHL lattice-reduction attack: BKZ on the solution-embedding lattice; a
    # verified 0/1 solution is a constructive feasibility certificate
    x_ahl = ahl_solve(A.astype(np.int64), b.astype(np.int64), block=ahl_block, tries=ahl_tries)
    if x_ahl is not None:
        return x_ahl.tolist(), True, True, 1.0, [], False, 'ahl_bkz'
    pump_ts = torch.tensor(pstats, dtype=torch.float, device=device).unsqueeze(0)

    # ── Feasibility prediction (GNN); policy logits reused for DFS ordering ──
    with torch.no_grad():
        xv, xc, cx = env.get_tensor_state(device)
        select_logits, assign_logits, _, feas_logit = model(xv, xc, x_lp_var, x_lp_check, ev2c, ec2v, pump_feats=pump_ts, check_extra=cx, mp_iters=mp_iters)
        feas_prob = torch.sigmoid(feas_logit).item()
        var_order = np.argsort(-select_logits.cpu().numpy())
        val_order = assign_logits.cpu().numpy().argmax(axis=1)
    pred_feasible = feas_prob >= feas_threshold
    root_assign = env.assignment.copy()

    if not pred_feasible:
        return env.assignment.tolist(), False, False, feas_prob, [], False, 'declared_infeasible'

    # ── Propagation may have resolved everything ──────────────────────────────
    if env.is_terminal():
        solved = bool(env.get_reward() == 1.0)
        return env.assignment.tolist(), solved, True, feas_prob, [], False, 'propagation_solved'

    # ── MCTS with 1-ply backtracking ──────────────────────────────────────────
    bt_stack = []
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
                    env = env_alt
                    mcts.cache.clear()
                else:
                    break
            else:
                break

    env.apply_local_search()
    solved = bool(env.get_accuracy() == 1.0)
    if solved:
        return env.assignment.tolist(), True, True, feas_prob, trajectory, env.is_invalid, 'mcts'

    # Pump repair from the MCTS terminal point (different seed for perturbations).
    x0 = np.maximum(env.assignment, 0).astype(np.float64)
    x_pump, pumped, _ = m.kernel_pump_stats(x0, A, b, A_pinv, max_iter=pump_iters, seed=1)
    if pumped:
        return x_pump.tolist(), True, True, feas_prob, trajectory, env.is_invalid, 'pump_repair'

    # Policy-guided complete backtracking from the propagated root: either finds
    # a solution, proves infeasibility, or exhausts its node budget.
    scaled_budget = int(dfs_budget * A.shape[1] / 25)
    sol, status, _ = m.policy_dfs(A, b, root_assign, var_order, val_order, node_budget=scaled_budget)
    if status == 'solved':
        return sol.tolist(), True, True, feas_prob, trajectory, env.is_invalid, 'dfs_solved'
    if status == 'infeasible':
        return env.assignment.tolist(), False, False, feas_prob, trajectory, env.is_invalid, 'dfs_proved_infeasible'

    return env.assignment.tolist(), False, True, feas_prob, trajectory, env.is_invalid, 'mcts'


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer  (matches baseline_dns format)
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(checkpoint, total, gt_f, gt_i,
                  tp, tn, fp, fn, sol_correct, times, prep_times,
                  method_counts, feas_threshold):
    feasibility_acc = (tp + tn) / total * 100
    sol_acc         = sol_correct / gt_f * 100 if gt_f > 0 else 0.0
    avg_t           = float(np.mean(times))
    std_t           = float(np.std(times))
    avg_prep        = float(np.mean(prep_times))
    w = 66
    print("\n" + "=" * w)
    print(f"Result Summary: LPneuroBLS v9")
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
    print(f"  [3] Method Breakdown")
    for name in ['propagation_infeasible', 'lp_hard_rule', 'kernel_pump', 'ahl_bkz',
                 'declared_infeasible', 'propagation_solved', 'mcts', 'pump_repair',
                 'dfs_solved', 'dfs_proved_infeasible']:
        print(f"      {name:<22}: {method_counts.get(name, 0):>6}")
    print("-" * w)
    print(f"  Avg Time / Inst   : {avg_t:.4f} sec  (incl. LP/pinv prep {avg_prep:.4f} sec)")
    print(f"  Std Time / Inst   : {std_t:.4f} sec")
    print(f"  Total Time        : {sum(times):.4f} sec")
    print("=" * w)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
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
    parser.add_argument('--dfs_budget',   type=int,   default=20000,
                        help='Node budget for policy-guided complete backtracking')
    parser.add_argument('--mp_scale', type=float, default=8.0,
                        help='Message-passing iterations = ceil(mp_scale * n / 25), capped at 48')
    parser.add_argument('--ahl_block', type=int, default=40)
    parser.add_argument('--ahl_tries', type=int, default=10)
    parser.add_argument('--ahl_tries_scale', type=float, default=0.0,
                        help='If >0, tries = round(ahl_tries_scale * n / 25) capped at 40; overrides ahl_tries')
    parser.add_argument('--pump_iters',   type=int,   default=200,
                        help='Max iterations for the kernel pump')
    parser.add_argument('--stochastic',   action='store_true',
                        help='Use stochastic MCTS selection (default: deterministic argmax)')
    parser.add_argument('--out_dir',      type=str,   default='./inference_out_v9')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = m.BipartiteGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    # ── Precompute LP relaxations, pinv, and graphs (timed per instance) ────────
    print(f"Precomputing LP relaxations for {len(inst_paths)} instances...")
    precomputed = []
    for inst_path in tqdm(inst_paths):
        with open(inst_path, 'r') as f:
            d = json.load(f)
        A  = np.array(d['A'],  dtype=np.int32)
        b  = np.array(d['b'],  dtype=np.int32)
        gt_feasible = bool(d.get('feasible', True))

        t_prep = time.time()
        x_lp_var_ts, x_lp_check_ts, x_lp, lp_is_feasible = prepare_lp_tensors(A, b, device)
        A_pinv = np.linalg.pinv(A.astype(np.float64))
        ev2c, ec2v = matrix_to_graph(A, device)
        prep_time = time.time() - t_prep

        precomputed.append({
            'path':           inst_path,
            'A':              A,
            'b':              b,
            'gt_feasible':    gt_feasible,
            'x_lp_var':       x_lp_var_ts,
            'x_lp_check':     x_lp_check_ts,
            'x_lp':           x_lp,
            'A_pinv':         A_pinv,
            'ev2c':           ev2c,
            'ec2v':           ec2v,
            'lp_is_feasible': lp_is_feasible,
            'prep_time':      prep_time,
        })

    # ── Inference loop ──────────────────────────────────────────────────────────
    total = len(precomputed)
    tp = tn = fp = fn = sol_correct = 0
    method_counts = {}
    times, prep_times = [], []

    mode_str = 'stochastic' if args.stochastic else 'deterministic'
    print(f"\nRunning inference on {total} instances  (mode: {mode_str}, feas_threshold: {args.feas_threshold})")

    for info in tqdm(precomputed):
        t0 = time.time()
        A, b = info['A'], info['b']
        gt_feasible = info['gt_feasible']

        mp_iters = min(48, int(np.ceil(args.mp_scale * A.shape[1] / 25.0)))
        mcts.mp_iters = mp_iters
        eff_tries = args.ahl_tries if args.ahl_tries_scale <= 0 else min(40, max(1, int(round(args.ahl_tries_scale * A.shape[1] / 25.0))))
        assignment, solved, pred_feasible, feas_prob, traj, is_invalid, method = infer_instance(
            model, mcts, A, b,
            info['x_lp_var'], info['x_lp_check'], info['x_lp'], info['A_pinv'], device,
            ev2c=info['ev2c'], ec2v=info['ec2v'],
            deterministic=deterministic,
            feas_threshold=args.feas_threshold,
            lp_is_feasible=info['lp_is_feasible'],
            pump_iters=args.pump_iters,
            dfs_budget=args.dfs_budget,
            mp_iters=mp_iters,
            ahl_block=args.ahl_block,
            ahl_tries=eff_tries,
        )

        elapsed = (time.time() - t0) + info['prep_time']
        times.append(elapsed)
        prep_times.append(info['prep_time'])
        method_counts[method] = method_counts.get(method, 0) + 1

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
            assign_arr = np.maximum(np.array(assignment, dtype=np.int32), 0)
            if np.all(A.dot(assign_arr) == b):
                sol_correct += 1

        result = {
            'instance':      str(info['path']),
            'gt_feasible':   bool(gt_feasible),
            'pred_feasible': bool(pred_feasible),
            'feas_prob':     round(float(feas_prob), 6),
            'solved':        bool(solved),
            'is_invalid':    bool(is_invalid),
            'method':        method,
            'time_sec':      round(elapsed, 6),
            'assignment':    assignment if pred_feasible else None,
            'trajectory_len': len(traj),
        }
        save_path = out_dir / (info['path'].stem + '_result.json')
        with open(save_path, 'w') as wf:
            json.dump(result, wf, indent=2)

    gt_f = tp + fn
    gt_i = tn + fp
    print_summary(
        args.checkpoint, total, gt_f, gt_i,
        tp, tn, fp, fn, sol_correct, times, prep_times,
        method_counts, args.feas_threshold,
    )


if __name__ == '__main__':
    main()
