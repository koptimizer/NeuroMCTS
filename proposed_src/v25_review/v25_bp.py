"""v25 P5: sum-product belief propagation marginals on the 0/1 equality factor graph
(variables x_j in {0,1}, one cardinality factor per row: sum_{j in row i} x_j = b_i), the
learning-free inference baseline behind NSNet. Factor-to-variable messages are computed exactly
by prefix/suffix count convolutions (O(r_i * b_i) per row), with damping; returns p_j = P(x_j = 1).
"""
import numpy as np


def _conv(P, q):
	# P: distribution over partial sums (length L); q: (q0, q1) of one more 0/1 variable
	R = np.zeros(len(P) + 1); R[:-1] += P * q[0]; R[1:] += P * q[1]
	return R


def bp_marginals(A, b, iters=300, damp=0.9, tol=1e-5):
	mm, n = A.shape
	rows = [np.flatnonzero(A[i]) for i in range(mm)]
	# messages factor -> variable, stored as P(x_j = 1) under that message (0.5 = uninformative)
	M = np.full((mm, n), 0.5)
	for it in range(iters):
		Mold = M.copy()
		lp1 = np.log(np.clip(M, 1e-12, 1)); lp0 = np.log(np.clip(1 - M, 1e-12, 1))
		tot1 = lp1.sum(axis=0); tot0 = lp0.sum(axis=0)
		for i in range(mm):
			J = rows[i]; r = len(J); bi = int(b[i])
			if r == 0: continue
			# variable -> factor messages (exclude factor i), normalised
			q1 = np.exp(tot1[J] - lp1[i, J]); q0 = np.exp(tot0[J] - lp0[i, J]); Z = q0 + q1; q0 /= Z; q1 /= Z
			L = min(bi, r) + 1
			pre = [np.array([1.0])]
			for t in range(r):
				pre.append(_conv(pre[-1], (q0[t], q1[t]))[:L])
			suf = [np.array([1.0])]
			for t in range(r - 1, -1, -1):
				suf.append(_conv(suf[-1], (q0[t], q1[t]))[:L])
			suf = suf[::-1]                               # suf[t] = distribution of sum over positions t..r-1
			for t in range(r):
				P = pre[t]; S = suf[t + 1]; Sr = S[::-1]            # Sr[k] = S[len(S)-1-k]
				# m_v = sum_c P[c] S[bi - v - c]  ->  dot of P with the reversed suffix shifted by bi - v
				m = []
				for v in (0, 1):
					lo = bi - v - (len(S) - 1); hi = bi - v     # c range where 0 <= bi-v-c < len(S)
					c0 = max(lo, 0); c1 = min(hi, len(P) - 1)
					m.append(float(P[c0:c1 + 1].dot(Sr[(len(S) - 1) - (hi - c0):(len(S) - 1) - (hi - c1) + 1])) if c1 >= c0 else 0.0)
				z = m[0] + m[1]
				new = 0.5 if z <= 0 else m[1] / z
				M[i, J[t]] = damp * Mold[i, J[t]] + (1 - damp) * new
		if np.abs(M - Mold).max() < tol:
			break
	lp1 = np.log(np.clip(M, 1e-12, 1)).sum(axis=0); lp0 = np.log(np.clip(1 - M, 1e-12, 1)).sum(axis=0)
	p = 1 / (1 + np.exp(lp0 - lp1))
	return p, it + 1
