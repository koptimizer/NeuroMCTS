# v23 사이클 종료 검증 감사 (2026-09-16)

목적: 다음 사이클 전에 (1) 테스트 데이터 누출, (2) 라벨(정답 분포) 오염, (3) 파이프라인이 실제로 보고대로 실행되었는지를 관련 파일 전수로 점검한다.
모든 검사는 저장된 파일과 코드에 대해 독립 재계산으로 수행했다.

## 1. 데이터 누출 (train ↔ test)

| 검사 | 방법 | 결과 |
|---|---|---|
| 학습 풀 vs 평가 풀 인스턴스 겹침 | (A,b)의 md5로 10개 데이터 파일 × 3 평가 디렉토리 전수 교차 | **학습 파일(full_*_train, cond_*_train)과 평가 디렉토리(v22test_*) 겹침 0** |
| v22test_10x25 / 18x50의 출처 | 해시 대조 | full_*_test.json(test split)의 **앞 30개**와 정확히 일치 — 학습 풀에 없음 |
| v22test_21x60 | 생성 seed 700000+i (v22_make_testset.py), 학습 seed 900000+i | 크기·seed 모두 분리 |
| train/test split 재현 | 대화 기록에서 복구한 split 명령(seed 0 shuffle, 80/20)을 재실행해 저장 파일과 비교 | **바이트 단위 동일** → `v22_transfer/v22_split.py`로 저장 |
| in-tree 학습 상태의 출처 | cond_10x25_train의 depth-0 상태 28개가 전부 full_10x25_train에 속함, test에는 0 | 학습 상태는 train split에서만 파생 |
| 분석 표(2/4/5/4b/6/13)의 평가 데이터 | 대화 기록에서 실제 호출 인자 복구 | 전부 `--test cond_10x25_test.json`(held-out), 학습은 `cond_10x25_train.json` |
| checkpoint 선택 | `v17_train_conditional.py` 확인 | 마지막 epoch 저장, test 성능 기반 선택 없음; hyperparameter는 v15 이후 고정(hidden 64, 4층, lr 1e-3, 20 epoch), sweep 기록 없음 |
| feature에 라벨 유입 | `featurize()` / `features()` 확인 | 입력은 A, b, LP 해뿐; `x`·`marginals`는 target으로만 사용 |

## 2. 라벨(정확한 marginal) 검증

| 검사 | 방법 | 결과 |
|---|---|---|
| root 라벨 | 학습 풀 6개 인스턴스를 2^25 전수 brute force(CP-SAT와 독립)로 재열거 | 해집합 **완전 일치**, marginal L1 오차 0, 심어진 해 ∈ S |
| in-tree 라벨 | cond_train 무작위 12개 상태 중 n_free ≤ 22인 9개를 축소 인스턴스에서 brute force | n_surviving·marginal **완전 일치**, 저장된 x ∈ S |
| 열거 완료 판정 | `enumerate_solutions`: OPTIMAL/INFEASIBLE만 완료, cap 도달 시 제외 | v22에서 수정된 규칙이 모든 생성 스크립트에 적용됨 |

## 3. 탐색 결과의 진위

| 검사 | 방법 | 결과 |
|---|---|---|
| "solved" 판정의 건전성 | 탐색 루프를 전체 배정 복원판으로 재실행해 A₀x=b₀ 검증 (21×60 4개 × lp/model, 10×25 5개 × lp/model/random) | **전부 검증 통과**; model arm의 node 수가 기록값과 정확히 일치(527/1,457/1,647/1,831) |
| AHL 단계 해 | `ahl_try_with_perm`는 `A·x==b` 검증 후에만 반환; 18×50 3개에서 재확인 | 통과 |
| ground truth 사용 | `v23_cascade_warm._worker`: `gt`는 보고용 필드에만, 탐색에 미전달 | 누출 경로 없음 |
| 두 arm의 조건 동일성 | seed별 AHL이 닫은 집합이 lp/model/random에서 동일한지 | 3 seed 모두 동일(16/14/10) |
| 결과 파일 완전성·예산 | 51 디렉토리 × 30 = 1,530 파일, 예산 초과·0-node 해결·조기 미해결 이상 | **이상 0건** |
| checkpoint | `runs/v22/model_n25/conditional.pt` md5 b0c62ff1…, 2026-09-10 18:14 생성(모든 탐색 실험 이전) | 체인 스크립트가 이 파일을 사용 |
| CPU 경합 | 사용자의 다른 학습 6개(train_v5_*)가 00:45에 시작 — 21×60 C 조건은 3 seed 모두 그 이전(≤00:32) 완료 | 헤드라인 수치 영향 없음; seed 2의 D/Crand만 경합 하에 실행(D S2 24.6 s vs 23.2/23.3, 결론 불변) |
| classical baseline | CP-SAT/SCIP 단일 스레드, 동일 인스턴스, jobs=6 | 프로토콜 일치 |

## 4. 발견된 문제와 조치

| # | 문제 | 심각도 | 조치 |
|---|---|---|---|
| 1 | **표 11(where), 표 4b(ctrl) 데이터 생성, "80.2% 범위 밖" 계측, train/test split의 스크립트가 저장소에 없었음** (대화 중 heredoc으로 실행) | 재현성 결함 (결과 자체는 유효) | 대화 기록에서 원문 복구 → `v23_ablation/{v23_where_analysis.py, v23_ctrl_data.sh, v23_query_range.py}`, `v22_transfer/v22_split.py`. split은 저장 파일과 동일함을 검증, where는 재실행으로 표 11 재현 확인(§5) |
| 2 | 논문 재현성 절에 split 단계가 빠져 있었고 경로가 `runs/data/`로 잘못 적힘 | 문서 | 4개 판본 모두 `v22_add_solutions.py` + `v22_split.py` 단계 추가, 경로 `runs/v22/`로 수정 |
| 3 | v22test_10x25/18x50 생성 명령(test split 앞 30개 복사)이 스크립트가 아님 | 경미 | `v22_split.py` docstring과 ROLES에 명시 |

부정 소견 없음: 테스트 데이터 학습 사용, 라벨 조작, 결과 파일 위조·누락, 보고 수치와 원시 파일 불일치 — 어느 것도 발견되지 않았다.

## 5. 표 11 재실행 (복구 스크립트 검증)

복구한 `v23_where_analysis.py`를 그대로 재실행한 결과(`runs/revision/where_rerun.log`):

```
n_c 구간별: 모델이 LP를 넘어서는가 (21x60 in-tree 상태, 전수열거 가능분)
  n_c     학습범위?    상태수  model FORCED  LP FORCED      차이   model~LP 상관
   20        안     14        100.0%     100.0%   +0.0%        1.000
   25        안     14        100.0%     100.0%   +0.0%        0.998
   30        밖     14         95.7%      95.0%   +0.7%        0.954
   40        밖     14         75.5%      75.2%   +0.4%        0.747
   50        밖     14         70.2%      65.0%   +5.3%        0.664
   60        밖      9         93.3%      88.9%   +4.4%        0.658
```

논문 표 11의 수치(20/25: +0.0, corr 1.000/0.998; 30: +0.7; 40: +0.4; 50: +5.3; 60: +4.4)와 **동일**하다.
