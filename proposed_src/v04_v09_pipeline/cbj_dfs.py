#!/usr/bin/env python3
"""
Conflict-directed backjumping (CBJ) DFS for binary linear systems.

Propagation records, for every forced assignment, the row that forced it.
On contradiction the violated rows' supports are traced back through the
forcing chain to decision variables (the conflict reason); when both values
of a decision fail, search jumps to the deepest decision in the accumulated
reason instead of backtracking chronologically, and conflict variables get
activity bumps that dynamically re-rank the policy-given branching order.
"""
import numpy as np


def propagate_reasons(A, b, assignment, forced_by):
    # Fixed-point propagation recording the forcing row of every implied
    # assignment; returns (ok, violated_rows)
    while True:
        ones = np.maximum(assignment, 0) * (assignment == 1)
        rem = b - A.dot(ones)
        n_un = A.dot((assignment == -1).astype(np.int32))
        viol = np.where((rem < 0) | (n_un < rem))[0]
        if viol.size:
            return False, viol
        active = n_un > 0
        fz = np.where(active & (rem == 0))[0]
        fo = np.where(active & (rem == n_un))[0]
        if fz.size == 0 and fo.size == 0:
            return True, np.empty(0, dtype=np.int64)
        for rows, val in ((fz, 0), (fo, 1)):
            for i in rows:
                for j in np.where((A[i] == 1) & (assignment == -1))[0]:
                    assignment[j] = val
                    forced_by[j] = i


def conflict_reason(A, violated_rows, assignment, forced_by, decision_set):
    # Trace violated rows' supports back through forcing rows to decisions
    reason = set()
    seen = set()
    stack = list(violated_rows)
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        for j in np.where(A[i] == 1)[0]:
            if assignment[j] == -1:
                continue
            if j in decision_set:
                reason.add(int(j))
            elif forced_by[j] >= 0 and forced_by[j] not in seen:
                stack.append(int(forced_by[j]))
    return reason


def cbj_dfs(A, b, root_assignment, var_order, val_order, node_budget=20000,
            activity_decay=0.97, activity_bump=1.0):
    """
    Returns (solution|None, status, nodes) with status in
    {'solved', 'infeasible', 'budget'}.
    """
    n = A.shape[1]
    rank = np.empty(n, dtype=np.int64)
    rank[np.asarray(var_order)] = np.arange(n)
    activity = np.zeros(n, dtype=np.float64)
    nodes = 0

    assignment = root_assignment.copy()
    forced_by = np.full(n, -2, dtype=np.int64)
    forced_by[assignment != -1] = -1
    ok, _ = propagate_reasons(A, b, assignment, forced_by)
    if not ok:
        return None, 'infeasible', 1

    def solved(a):
        return np.array_equal(A.dot(np.maximum(a, 0)), b)

    if not (assignment == -1).any():
        return (assignment.copy(), 'solved', 1) if solved(assignment) else (None, 'infeasible', 1)

    def pick_var(a):
        un = np.where(a == -1)[0]
        return int(un[np.argmin(rank[un] - activity[un] * n)])

    # trail level: var, values left to try, state snapshots, accumulated reason
    trail = [{'var': pick_var(assignment), 'tried': 0,
        'snap_a': assignment.copy(), 'snap_f': forced_by.copy(), 'reason': set()}]

    while True:
        nodes += 1
        if nodes > node_budget:
            return None, 'budget', nodes
        top = trail[-1]
        v = top['var']
        val = int(val_order[v]) if top['tried'] == 0 else 1 - int(val_order[v])
        assignment = top['snap_a'].copy()
        forced_by = top['snap_f'].copy()
        assignment[v] = val
        forced_by[v] = -1
        ok, viol = propagate_reasons(A, b, assignment, forced_by)
        if ok and not (assignment == -1).any():
            if solved(assignment):
                return assignment.copy(), 'solved', nodes
            ok = False
            viol = np.where(A.dot(np.maximum(assignment, 0)) != b)[0]
        if ok:
            trail.append({'var': pick_var(assignment), 'tried': 0,
                'snap_a': assignment.copy(), 'snap_f': forced_by.copy(), 'reason': set()})
            continue

        # Conflict: accumulate reason at this level, bump activity
        decision_set = {t['var'] for t in trail}
        reason = conflict_reason(A, viol, assignment, forced_by, decision_set)
        activity *= activity_decay
        for j in reason:
            activity[j] += activity_bump
        top['reason'] |= (reason - {v})

        if top['tried'] == 0:
            top['tried'] = 1
            continue

        # Both values failed: backjump to the deepest responsible decision
        while True:
            dead = trail.pop()
            carry = dead['reason']
            if not trail:
                return None, 'infeasible', nodes
            # unwind levels whose decision variable is not in the carried reason
            while trail and trail[-1]['var'] not in carry:
                skipped = trail.pop()
                carry |= skipped['reason']
            if not trail:
                return None, 'infeasible', nodes
            tgt = trail[-1]
            tgt['reason'] |= (carry - {tgt['var']})
            if tgt['tried'] == 0:
                tgt['tried'] = 1
                break
            # target already tried both values: continue jumping with its reason
        # loop back to retry at the jump target (second value)
