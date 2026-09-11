"""Cross-arm report for the v22 factorial: every condition x AHL setting x arm in one table.

v22_cascade_search.py prints a table per invocation, but conditions C and D are separate
invocations (different checkpoints), so the comparisons that matter -- model vs lp within
a size, and model_C vs model_D on the same test set -- span invocations. This reads the
per-instance files directly and assembles them.

Times are search_sec (AHL time excluded) when comparing guidance, since AHL runs
identically for every arm and including it would dilute the difference under test.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path('../../runs/v22')


def load(tag, mode, arm):
	p = ROOT / f"{tag}_ahl{mode}_{arm}"
	if not p.exists():
		return None
	return {json.load(open(f))['instance']: json.load(open(f)) for f in sorted(p.glob('*_r.json'))}


def stats(d):
	feas = [r for r in d.values() if r['gt_feasible']]
	sol = [r for r in feas if r['solved']]
	by_ahl = sum(1 for r in sol if r['solved_by'] == 'ahl')
	by_srch = sum(1 for r in sol if r['solved_by'] == 'search')
	return dict(n=len(feas), solved=len(sol), rate=len(sol) / max(1, len(feas)),
	             by_ahl=by_ahl, by_search=by_srch,
	             med_search=float(np.median([r['search_sec'] for r in sol])) if sol else float('nan'),
	             med_total=float(np.median([r['total_sec'] for r in sol])) if sol else float('nan'),
	             med_nodes=float(np.median([r['nodes'] for r in sol if r['nodes'] > 0])) if sol else float('nan'),
	             solved_set=set(k for k, r in d.items() if r['solved'] and r['gt_feasible']))


def mcnemar(a, b):
	oa, ob = len(a - b), len(b - a)
	n = oa + ob
	p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(max(oa, ob), n + 1)) / 2 ** n)
	return oa, ob, p


def pairwise(da, db, la, lb):
	# speed comparison restricted to instances both arms solved by search
	com = [k for k in da if k in db and da[k]['solved_by'] == 'search' and db[k]['solved_by'] == 'search']
	if not com:
		print(f"    {la} vs {lb}: 양쪽 모두 탐색으로 푼 인스턴스 없음")
		return
	sp = np.array([db[k]['search_sec'] / max(1e-9, da[k]['search_sec']) for k in com])
	print(f"    {la} vs {lb} ({len(com)}개 공통): {la}가 빠른 경우 {(sp>1).sum()}/{len(com)}, "
	      f"속도배수 중앙={np.median(sp):.2f}x  총시간 {sum(da[k]['search_sec'] for k in com):.1f}s "
	      f"vs {sum(db[k]['search_sec'] for k in com):.1f}s")


def main():
	rows = [('A_n25to25', 'n=25모델→n=25', ['lp', 'model']),
	         ('B_n25to50', 'n=25모델→n=50', ['lp', 'model']),
	         ('R_n50to50', 'n=50모델→n=50', ['model']),
	         ('C_n25to60', 'n=25모델→n=60', ['lp', 'model']),
	         ('D_n50to60', 'n=50모델→n=60', ['model'])]
	print("=" * 104)
	print("v22 전체: |S| 통제 전이 실험 (조건 x AHL x arm)")
	print("=" * 104)
	print(f"{'조건':<18}{'AHL':>5}{'arm':>8}{'해결율':>10}{'AHL해결':>9}{'탐색해결':>10}"
	      f"{'탐색중앙':>11}{'전체중앙':>11}{'노드중앙':>11}")
	for tag, label, arms in rows:
		for mode in ('off', 'on'):
			for arm in arms:
				d = load(tag, mode, arm)
				if not d:
					continue
				s = stats(d)
				print(f"{label:<18}{mode:>5}{arm:>8}{100*s['rate']:>9.1f}%{s['by_ahl']:>9}"
				      f"{s['by_search']:>10}{s['med_search']:>11.2f}{s['med_total']:>11.2f}"
				      f"{s['med_nodes']:>11.0f}")
		print("-" * 104)

	print("\n### arm 간 대조 (AHL off, 안내 품질만) ###")
	for tag, label in [('B_n25to50', 'n=50 테스트'), ('C_n25to60', 'n=60 테스트')]:
		lp, mo = load(tag, 'off', 'lp'), load(tag, 'off', 'model')
		if lp and mo:
			print(f"  [{label}]")
			pairwise(mo, lp, 'model', 'lp')
			oa, ob, p = mcnemar(stats(mo)['solved_set'], stats(lp)['solved_set'])
			print(f"    해결 여부: model만 {oa}개, lp만 {ob}개, McNemar p={p:.3f}")

	print("\n### 전이 손실 (같은 테스트, 학습 크기만 다름, AHL off) ###")
	for tt, tc, label in [('R_n50to50', 'B_n25to50', 'n=50 테스트'),
	                       ('D_n50to60', 'C_n25to60', 'n=60 테스트')]:
		same, tr = load(tt, 'off', 'model'), load(tc, 'off', 'model')
		if same and tr:
			print(f"  [{label}]")
			pairwise(tr, same, 'n=25학습(전이)', 'n=50학습(동일크기)')
			oa, ob, p = mcnemar(stats(tr)['solved_set'], stats(same)['solved_set'])
			print(f"    해결 여부: 전이모델만 {oa}개, 동일크기모델만 {ob}개, McNemar p={p:.3f}")

	print("\n### AHL on (실제 운용): AHL이 실패한 잔여에서의 안내 ###")
	for tag, label in [('B_n25to50', 'n=50'), ('C_n25to60', 'n=60')]:
		for arm in ('lp', 'model'):
			d = load(tag, 'on', arm)
			if d:
				s = stats(d)
				resid = s['n'] - s['by_ahl']
				print(f"  {label} {arm:6s}: AHL {s['by_ahl']}/{s['n']} 해결, "
				      f"잔여 {resid}개 중 탐색이 {s['by_search']}개 해결")


if __name__ == '__main__':
	main()
