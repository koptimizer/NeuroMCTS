# CP-SAT 파라미터 선택 정책 실험

## 1. 배경

[AHL block 선택 실험](AHL_BLOCK_POLICY_REPORT.md)에서 같은 sweep+MLP 방법론으로 "학습보다
기본값 교체가 더 큰 승리"라는 결론을 얻었다. 이어서 CP-SAT 자체의 solver 파라미터
(`linearization_level`, `search_branching` 등)도 인스턴스별로 다르게 고르면 이득이
있는지 동일한 방법론으로 확인했다. `cpsat_param_sweep.py`(고정 config 비교)에서 이미
"3종 튜닝 전부 무효"였던 파라미터들이지만, AHL 사례처럼 "고정 튜닝은 무효 ≠ 인스턴스별
적응도 무효"일 수 있어 재검증했다.

## 2. 방법

40×100(n=40)/60×150(n=40)/80×200(n=30) — CP-SAT가 이미 어려움을 겪는 크기만 대상
(10×25/20×50은 모든 config에서 즉시 풀려 퇴화 케이스). 6개 config
(`default, linearization0, linearization2, portfolio_search, lp_search, no_symmetry`,
`cpsat_param_sweep.py`의 CONFIGS 재사용) × 각 인스턴스, cap=20초.

**방법론 함정 하나 발견 및 수정**: 처음 jobs=16으로 병렬 스윕했더니, 단독 실행하면
14~15초에 풀리는 인스턴스가 병렬 CPU 경합 때문에 20초 cap을 넘겨 unresolved로
잘못 기록되는 걸 발견(직접 재현 확인). CP-SAT는 시간 압박에 민감한 솔버라 다수
워커의 동시 실행 자체가 측정을 오염시킨다. jobs=4로 낮춰 재스윕(47분 소요)해서
해결.

## 3. 결과

### Config별 marginal 해결 수 (jobs=4, 정제된 데이터)

| config | 40×100 (n=40) | 60×150 (n=40) | 80×200 (n=30) |
|---|---|---|---|
| default | 17 | 8 | 6 |
| linearization0 | 0 | 0 | 0 |
| linearization2 | 12 | 4 | 2 |
| portfolio_search | 12 | 8 | 5 |
| lp_search | 17 | 8 | 6 |
| no_symmetry | 17 | 8 | 6 |
| **oracle(6개 중 최적)** | **18** | **8** | **6** |

### 학습된 정책 vs default (5개 시드 평균, held-out)

| 크기 | 정책 | default |
|---|---|---|
| 40×100 | 34.0% | 36.0% |
| 60×150 | 14.0% | 14.0% |
| 80×200 | 22.9% | 25.7% |

## 4. 해석 — AHL block 실험과 정반대 결론

1. **오라클(6개 config 중 인스턴스별 최적 선택)이 default와 사실상 동일하다.**
   40×100에서 딱 1개(18 vs 17), 60×150/80×200은 0개 차이. 즉 이 6개 config 사이에는
   "이 인스턴스는 A가, 저 인스턴스는 B가 낫다"는 상호보완적 구조가 애초에 없다 —
   `default`/`lp_search`/`no_symmetry`가 항상 정확히 같은 인스턴스 집합을 풀었다
   (이 문제의 단순 등식 제약 구조에서는 symmetry-breaking이나 LP 기반 분기가
   기본 전략과 다른 탐색 경로를 만들지 않는 것으로 보임).
2. **학습된 정책은 이득이 없고, 오히려 약간 손해다.** 모든 크기에서 policy ≤ default
   (동률이거나 최대 -2.8pp). 배울 신호가 없는 곳에서 억지로 패턴을 찾으려 하면
   노이즈를 학습해 오히려 살짝 나빠진다 — 표본이 작을수록 이 위험이 크다.
3. **AHL block 사례와의 대조가 핵심 시사점이다.** 같은 방법론을 똑같이 적용했는데
   AHL block은 진짜 신호(오라클이 default보다 확실히 높음, 실제 이득 존재)를 찾았고
   CP-SAT config는 신호가 없음(오라클≈default)을 정확히 찾아냈다. 즉 이 sweep+MLP
   방법론 자체는 "아무거나 학습된 것처럼 보이게 하는" 방법론적 아티팩트가 아니라,
   신호가 있을 때는 잡아내고 없을 때는 정직하게 무(無)를 보고한다는 뜻이다.

## 5. 권고

- CP-SAT 파라미터는 인스턴스별로 조정할 가치가 없다. `linearization0`만 확실히
  피하면 되고, 나머지는 `default`(=`lp_search`=`no_symmetry`) 그대로 두는 게 최선.
- 이번 결과로 "CP-SAT를 어떻게든 조정해서 이기자"는 방향의 여지가 더 좁아졌다 —
  hint 주입(유일한 세션 내 긍정, [[cpsat-hint-and-param-findings]])을 제외하면
  CP-SAT 자체를 건드려서 얻을 수 있는 추가 이득은 거의 소진된 것으로 보인다.
- 병렬 스윕 시 시간 cap 근처 결과는 CPU 경합에 취약하다는 점은 향후 유사 실험
  설계 시 반드시 감안할 것(jobs를 낮추거나 격리 실행 필요).

## 6. 재현 및 파일

- `proposed_src/cpsat_hparam_sweep.py` — sweep 수집 (jobs=4 권장, 16은 경합으로 왜곡됨)
- `proposed_src/cpsat_hparam_policy.py` — 정책 학습·평가 (`--seed`로 다중 시드)
- `runs/cpsat_hparam/{sweep_v2.json, policy.pt}` — 정제된 데이터·학습된 정책
  (`sweep.json`은 jobs=16 오염 버전, 참고용으로만 보존)
