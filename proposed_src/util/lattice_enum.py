#!/usr/bin/env python3
"""
Lattice-reformulated enumeration for binary linear systems (Aardal-style).

Column Hermite normal form with a recorded unimodular transform yields an
integer particular solution x_p and a saturated basis K of the integer kernel
lattice {z : Az = 0}; the basis is LLL-reduced (fpylll) and solutions are
enumerated in coefficient space, x = x_p + K^T u, with interval-arithmetic
pruning against the box 0 <= x <= 1. Every leaf is integral and satisfies
Ax = b by construction, so exhausting the tree proves infeasibility.
"""
import numpy as np
from fpylll import IntegerMatrix, LLL


def hnf_with_transform(A):
    # Column-style HNF via exact integer column operations; returns (H, U, r)
    # with A @ U = H, U unimodular, r = rank; kernel basis = columns U[:, r:]
    m = len(A)
    n = len(A[0])
    H = [[int(x) for x in row] for row in A]
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def col_op(j1, j2, q):
        # col_j2 -= q * col_j1
        for i in range(m):
            H[i][j2] -= q * H[i][j1]
        for i in range(n):
            U[i][j2] -= q * U[i][j1]

    def col_swap(j1, j2):
        for i in range(m):
            H[i][j1], H[i][j2] = H[i][j2], H[i][j1]
        for i in range(n):
            U[i][j1], U[i][j2] = U[i][j2], U[i][j1]

    def col_neg(j):
        for i in range(m):
            H[i][j] = -H[i][j]
        for i in range(n):
            U[i][j] = -U[i][j]

    r = 0
    for i in range(m):
        piv = None
        while True:
            nz = [j for j in range(r, n) if H[i][j] != 0]
            if not nz:
                break
            if len(nz) == 1:
                piv = nz[0]
                break
            nz.sort(key=lambda j: abs(H[i][j]))
            j1 = nz[0]
            for j2 in nz[1:]:
                q = H[i][j2] // H[i][j1]
                if q != 0:
                    col_op(j1, j2, q)
        if piv is None:
            continue
        col_swap(r, piv)
        if H[i][r] < 0:
            col_neg(r)
        r += 1
    return H, U, r


def lattice_data(A, b):
    # Returns (x_p, K) with A x_p = b (integer) and K the LLL-reduced saturated
    # integer kernel basis (rows are basis vectors); (None, None) if no integer
    # solution exists
    m, n = A.shape
    H, U, r = hnf_with_transform(A.tolist())
    # forward-solve H y = b on the pivot structure (columns 0..r-1)
    y = [0] * r
    resid = [int(v) for v in b]
    pivot_rows = []
    ci = 0
    for i in range(m):
        if ci < r and H[i][ci] != 0:
            pivot_rows.append(i)
            ci += 1
    if ci < r:
        return None, None
    for k, i in enumerate(pivot_rows):
        acc = resid[i] - sum(H[i][j] * y[j] for j in range(k))
        if acc % H[i][k] != 0:
            return None, None
        y[k] = acc // H[i][k]
    # consistency on non-pivot rows
    x_p = [sum(U[i][j] * y[j] for j in range(r)) for i in range(n)]
    if not all(int(v) == int(w) for v, w in zip(A.dot(np.array(x_p, dtype=object)), b)):
        return None, None
    K = [[U[i][j] for i in range(n)] for j in range(r, n)]
    if K:
        M = IntegerMatrix.from_matrix(K)
        LLL.reduction(M)
        K = [[M[i, j] for j in range(n)] for i in range(M.nrows)]
    return np.array(x_p, dtype=np.int64), np.array(K, dtype=np.int64)


def recenter(x_p, K, target=0.5, passes=6):
    # One-shot least-squares rounding followed by Babai coordinate passes:
    # translates x_p by lattice vectors to bring it near the box center
    x = x_p.astype(np.float64).copy()
    u0, *_ = np.linalg.lstsq(K.T.astype(np.float64), (target - x), rcond=None)
    x = x + K.T.dot(np.rint(u0))
    for _ in range(passes):
        moved = False
        for i in range(K.shape[0]):
            ki = K[i].astype(np.float64)
            c = int(np.rint(np.dot(x - target, ki) / max(1e-9, np.dot(ki, ki))))
            if c != 0:
                x -= c * ki
                moved = True
        if not moved:
            break
    return np.rint(x).astype(np.int64)


def coeff_boxes(x_p, K):
    # Per-coefficient integer bounds for the box 0 <= x_p + K^T u <= 1.
    # Provably valid initial bounds: any binary solution x gives u = P(x - x_p)
    # with P = pinv(K^T), so |u_i| <= sum_j |P_ij| * max(|x_pj|, |1 - x_pj|);
    # interval fixed-point tightening below only shrinks this valid superset,
    # so an empty box is a genuine infeasibility proof
    k, n = K.shape
    P = np.linalg.pinv(K.T.astype(np.float64))
    cj = np.maximum(np.abs(x_p), np.abs(1 - x_p)).astype(np.float64)
    init = np.abs(P).dot(cj)
    lo = -np.ceil(init + 1e-9).astype(np.int64)
    hi = np.ceil(init + 1e-9).astype(np.int64)
    for _ in range(30):
        changed = False
        pos = np.where(K > 0, K, 0)
        neg = np.where(K < 0, K, 0)
        for i in range(k):
            others = [t for t in range(k) if t != i]
            base_min = x_p.astype(np.float64) + sum(pos[t] * lo[t] + neg[t] * hi[t] for t in others)
            base_max = x_p.astype(np.float64) + sum(pos[t] * hi[t] + neg[t] * lo[t] for t in others)
            for j in range(n):
                if K[i, j] == 0:
                    continue
                # need 0 <= x_j <= 1 achievable: K[i,j]*u_i in [ -base_max_j, 1-base_min_j ]
                ub = (1.0 - base_min[j]) / K[i, j] if K[i, j] > 0 else (0.0 - base_max[j]) / K[i, j]
                lb = (0.0 - base_max[j]) / K[i, j] if K[i, j] > 0 else (1.0 - base_min[j]) / K[i, j]
                nhi = int(np.floor(ub + 1e-9))
                nlo = int(np.ceil(lb - 1e-9))
                if nhi < hi[i]:
                    hi[i] = nhi
                    changed = True
                if nlo > lo[i]:
                    lo[i] = nlo
                    changed = True
            if hi[i] < lo[i]:
                return lo, hi, False
        if not changed:
            break
    return lo, hi, True


def lattice_enumerate(A, b, node_budget=200000, order=None):
    """
    Returns (solution|None, status, nodes), status in
    {'solved', 'infeasible', 'budget', 'no_integer_solution'}.
    """
    x_p, K = lattice_data(A, b)
    if x_p is None:
        return None, 'infeasible', 1
    k, n = (K.shape if K.size else (0, A.shape[1]))
    if k == 0:
        ok = np.all(x_p >= 0) and np.all(x_p <= 1)
        return (x_p.astype(np.int8), 'solved', 1) if ok else (None, 'infeasible', 1)

    x_p = recenter(x_p, K)
    lo, hi, feas = coeff_boxes(x_p, K)
    if not feas:
        return None, 'infeasible', 1
    # enumerate short (tightly bounded) coefficients first unless caller orders
    dim_order = order if order is not None else np.argsort(hi - lo)
    Kord = K[dim_order]
    loo, hio = lo[dim_order], hi[dim_order]
    pos = np.where(Kord > 0, Kord, 0)
    neg = np.where(Kord < 0, Kord, 0)
    # suffix min/max contributions for pruning
    suf_min = np.zeros((k + 1, n))
    suf_max = np.zeros((k + 1, n))
    for i in range(k - 1, -1, -1):
        suf_min[i] = suf_min[i + 1] + pos[i] * loo[i] + neg[i] * hio[i]
        suf_max[i] = suf_max[i + 1] + pos[i] * hio[i] + neg[i] * loo[i]

    nodes = 0
    x_partial = x_p.astype(np.float64)

    def dfs(depth):
        nonlocal nodes, x_partial
        nodes += 1
        if nodes > node_budget:
            return 'budget'
        if depth == k:
            x = np.rint(x_partial).astype(np.int64)
            if np.all(x >= 0) and np.all(x <= 1):
                return x
            return None
        if np.any(x_partial + suf_max[depth] < -1e-9) or np.any(x_partial + suf_min[depth] > 1 + 1e-9):
            return None
        # dynamic bounds for the branching coefficient: exact interval reasoning
        # against every variable given the fixed prefix and remaining suffixes
        kv = Kord[depth].astype(np.float64)
        hi_req = 1.0 + 1e-9 - x_partial - suf_min[depth + 1]
        lo_req = -1e-9 - x_partial - suf_max[depth + 1]
        u_lo, u_hi = float(loo[depth]), float(hio[depth])
        pos = kv > 0
        neg = kv < 0
        if pos.any():
            u_hi = min(u_hi, np.floor(np.min(hi_req[pos] / kv[pos]) + 1e-9))
            u_lo = max(u_lo, np.ceil(np.max(lo_req[pos] / kv[pos]) - 1e-9))
        if neg.any():
            u_hi = min(u_hi, np.floor(np.min(lo_req[neg] / kv[neg]) + 1e-9))
            u_lo = max(u_lo, np.ceil(np.max(hi_req[neg] / kv[neg]) - 1e-9))
        if u_hi < u_lo:
            return None
        vals = sorted(range(int(u_lo), int(u_hi) + 1), key=abs)
        for u in vals:
            x_partial += Kord[depth] * u
            res = dfs(depth + 1)
            x_partial -= Kord[depth] * u
            if isinstance(res, np.ndarray) or res == 'budget':
                return res
        return None

    res = dfs(0)
    if isinstance(res, np.ndarray):
        assert np.array_equal(A.dot(res), b)
        return res.astype(np.int8), 'solved', nodes
    if res == 'budget':
        return None, 'budget', nodes
    return None, 'infeasible', nodes


def ahl_solve_trace(A, b, block=40, tries=10, N=1000, seed=0):
    # ahl_solve variant that records per-try statistics for hazard-model
    # training and budget scheduling: (solution|None, hit_try|None, trace)
    # where trace[t] = dict(min_norm, n_zero_tail, gs_slope, t_sec)
    import time as _time
    from fpylll import BKZ, GSO
    m, n = A.shape
    rng = np.random.default_rng(seed)
    trace = []
    for t in range(tries):
        t0 = _time.time()
        perm = rng.permutation(n) if t else np.arange(n)
        Ap = A[:, perm]
        dim = n + 1
        M = IntegerMatrix(dim, n + 1 + m)
        for j in range(n):
            M[j, j] = 2
            for i in range(m):
                M[j, n + 1 + i] = int(N * Ap[i, j])
        for j in range(n):
            M[n, j] = 1
        M[n, n] = 1
        for i in range(m):
            M[n, n + 1 + i] = int(N * b[i])
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=min(block, dim)))
        sol = None
        n_zero_tail = 0
        min_norm = float('inf')
        for r in range(dim):
            v = np.array([M[r, c] for c in range(n + 1 + m)], dtype=np.int64)
            nrm = float(np.sqrt((v.astype(np.float64) ** 2).sum()))
            min_norm = min(min_norm, nrm)
            if np.all(v[n + 1:] == 0):
                n_zero_tail += 1
            for sgn in (1, -1):
                w = sgn * v
                if np.any(w[n + 1:] != 0) or abs(w[n]) != 1:
                    continue
                x = (w[:n] * w[n] * -1 + 1)
                if np.all((x == 0) | (x == 2)):
                    xs = (x // 2).astype(np.int64)
                    xo = np.empty(n, dtype=np.int64)
                    xo[perm] = xs
                    if np.array_equal(A.dot(xo), b):
                        sol = xo
        g = GSO.Mat(M)
        g.update_gso()
        lognorms = [0.5 * float(np.log(max(1e-9, g.get_r(i, i)))) for i in range(min(dim, 20))]
        gs_slope = float(np.polyfit(range(len(lognorms)), lognorms, 1)[0]) if len(lognorms) > 2 else 0.0
        trace.append({'min_norm': min_norm, 'n_zero_tail': int(n_zero_tail),
            'gs_slope': gs_slope, 't_sec': _time.time() - t0, 'profile': lognorms})
        if sol is not None:
            return sol, t + 1, trace
    return None, None, trace


# Sound but empirically useless: lambda_1(L) >= min_i ||b_i*|| always holds,
# but this lattice's determinant is dominated by the diagonal-2 rows, so the
# trailing GSO vector is always far shorter than the target norm regardless
# of block size (confirmed: bound ~3 vs target ~7-10 across 20x50/40x100) --
# kept only as a documented dead end, see neurosat-paired-dualnode-findings.
def ahl_infeas_certificate(A, b, block=40, tries=10, N=1000, seed=0, margin=1e-6):
    from fpylll import BKZ, GSO, FPLLL
    m, n = A.shape
    target = float(np.sqrt(n + 1))
    rng = np.random.default_rng(seed)
    FPLLL.set_precision(128)
    best_bound = 0.0
    best_try = None
    for t in range(tries):
        perm = rng.permutation(n) if t else np.arange(n)
        Ap = A[:, perm]
        dim = n + 1
        M = IntegerMatrix(dim, n + 1 + m)
        for j in range(n):
            M[j, j] = 2
            for i in range(m):
                M[j, n + 1 + i] = int(N * Ap[i, j])
        for j in range(n):
            M[n, j] = 1
        M[n, n] = 1
        for i in range(m):
            M[n, n + 1 + i] = int(N * b[i])
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=min(block, dim)))
        g = GSO.Mat(M, float_type='mpfr')
        g.update_gso()
        min_r = min(g.get_r(i, i) for i in range(dim))
        bound = float(np.sqrt(max(0.0, min_r)))
        if bound > best_bound:
            best_bound = bound
            best_try = t + 1
        if bound > target * (1.0 + margin):
            return True, bound, target, t + 1
    return False, best_bound, target, best_try


def ahl_solve(A, b, block=40, tries=10, N=1000, seed=0):
    # Aardal-Hurkens-Lenstra-style embedding: a lattice is built whose short
    # vectors of the form (2x-1 | -1 | 0) encode 0/1 solutions of Ax=b; BKZ
    # reduction with randomized column permutations reveals them directly.
    # Only verified solutions are returned, so infeasible input cannot yield
    # a false positive
    from fpylll import BKZ
    m, n = A.shape
    rng = np.random.default_rng(seed)
    for t in range(tries):
        perm = rng.permutation(n) if t else np.arange(n)
        Ap = A[:, perm]
        dim = n + 1
        M = IntegerMatrix(dim, n + 1 + m)
        for j in range(n):
            M[j, j] = 2
            for i in range(m):
                M[j, n + 1 + i] = int(N * Ap[i, j])
        for j in range(n):
            M[n, j] = 1
        M[n, n] = 1
        for i in range(m):
            M[n, n + 1 + i] = int(N * b[i])
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=min(block, dim)))
        for r in range(dim):
            v = np.array([M[r, c] for c in range(n + 1 + m)], dtype=np.int64)
            for sgn in (1, -1):
                w = sgn * v
                if np.any(w[n + 1:] != 0) or abs(w[n]) != 1:
                    continue
                x = (w[:n] * w[n] * -1 + 1)
                if np.all((x == 0) | (x == 2)):
                    xs = (x // 2).astype(np.int64)
                    xo = np.empty(n, dtype=np.int64)
                    xo[perm] = xs
                    if np.array_equal(A.dot(xo), b):
                        return xo
    return None
