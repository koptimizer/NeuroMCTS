# 261001 비평 대응 계획 (v25 사이클 제안)

입력: `docs/tex/261001_Mnet_비평.pdf` (Gemini / Claude / ChatGPT 세 심사 의견). 대상 원고: `docs/tex/261001_paper.tex`.

## 레벨 정의
- 심각도 S1: 논문의 전제·핵심 주장을 흔듦 (실험 없이는 방어 불가) / S2: 주요 보완 (실험 또는 상당한 재서술) / S3: 글 수정으로 해결 / S4: 반박 가능 또는 선택 사항
- 비용 C1: 문장 수정 (<1시간) / C2: 재집계·스크립트 (1~3시간) / C3: 소규모 신규 실험 (반나절~1일, 학습 없음) / C4: 학습 포함 신규 실험 (계산 1~3일) / C5: 시스템 재설계 (수 주) — 이번 사이클에서 제외

## 사전 확인한 사실 (계획의 근거)
- 배수의 정의 불일치는 사실: 논문의 5.5×, 2.3~2.4×는 **instance별 비율의 중앙값**(5.48, 2.29/2.38)이고, 표의 중앙값끼리 나누면 5.95, 2.44/2.71. 정의를 명시하면 해결(F1).
- p값: 21×60 100개에서 Stage 1+2 vs Stage 1 시간 66/100은 양측 부호검정 p=0.0018 (논문 "p<0.001"은 node 기준 7.9e-5와 혼용). 재집계 필요(F2).
- "forward pass 90% 이상"은 root 노드(1.9 ms) 기준. 트리 평균으로는 LP rule 1,855 node/s 대 Stage 1 794 node/s → forward 비중 약 57%. 심사자 계산이 맞음(A2).
- "20×50 random" 수치는 v8/v9 시절의 다른 instance 계열 결과. 현재 계열에서 random arm 재실행 필요(F8).
- Stage 2 rollout이 90초 예산에 걸리면 완료된 하위 subtree의 결정만 기록되고 중단된 조상 결정은 기록되지 않음 (`v24_rl_core.dfs_collect`). 서술만 추가하면 됨(E4).
- 코드의 bound check는 빈 행 제거 전에 실행됨 → 알고리즘 1의 서술 순서만 고치면 됨(E8).

## 항목별 정리

| ID | 지적 (출처) | 심각도 | 비용 | 대응 |
|---|---|---|---|---|
| A1 | CP-SAT/SCIP 대비 9~15배 느림; compiled inference 실측 요구 (G, C, O) | S1 | C4 | dense-tensor GAT(순수 torch, n≤80이라 dense adjacency) + TorchScript/torch.compile로 forward 재구현 → node당 시간 측정. 3배 이상 빨라지면 21×60(100개×3 seed×3 arm)과 24×70 시간 재측정. node 수는 불변. 결론은 "구현 병목이며, 최적화 후 격차 X배"로 한정 |
| A2 | forward 90% 주장이 표와 충돌 (C) | S3 | C2 | 노드당 비용 프로파일(propagation/LP/forward/overhead)을 트리 전체에서 측정해 표로 제시. root 1.9 ms, 평균 ~57%로 정정 |
| A3 | branch≠node이므로 "품질 아닌 처리량" 결론 불가 (C, O) | S3 | C1 | "CP-SAT는 presolve·clause learning·restart로 탐색 자체가 다르다; 크기 차수 비교일 뿐"으로 한정. A2 프로파일이 "우리 루프 안의 병목"을 뒷받침 |
| A4 | Table 8의 4,040~8,873 범위 설명 없음 (C) | S3 | C1 | 학습 seed 두 개(30개 세트)의 중앙값임을 명시 |
| B1 | 고전 CP 분기 규칙(dom/wdeg, impact, counting-based, probing)과 같은 루프에서 비교 없음 (C, O) | S2 | C3 | guide 함수 4종 추가: (i) maxSD counting — 0/1 행에서는 행 내 solution density가 정확히 b'_i/r_i라 계산이 즉시 가능, (ii) dom/wdeg — 노드 폐기를 유발한 행에 가중치 누적, (iii) propagation impact — 각 후보를 고정했을 때 propagation 고정 수 최대, (iv) LP-probing(기존 `g_lp_probe`). 18×50·21×60(100개)에서 3 seed |
| B2 | 학습 baseline 없음: random init Stage 2(FMSTS식), LP 규칙 초기화, trunk 비동결, 0/1 hard label Stage 1 (C, O) | S1 | C4 | RL 3회(각 ~2시간 학습 + 1시간 평가) + hard-label Stage 1 학습(20분) + 평가. 기여 2("posterior 초기화 + 동결")의 직접 검증 |
| B3 | lattice 단계가 너무 약함(BKZ-12 스캔); Wassermann식 열거면 n=80도 풀릴 수 있어 전제가 무너짐 (C, O) | S1 | C3 | 21×60/24×70/28×80 30개에서 block 20/30/40, 순열 30회, fpylll 열거(kernel lattice CVP/SVP)를 시간 제한 60~600초로 실행해 coverage-시간 곡선 제시. 많이 풀리면 "잔여" 정의를 그 단계 뒤로 옮기고 branching 기여는 잔여에서 주장 |
| B4 | Gurobi MIP·PB solver 미비교 (C) | S3 | C2 | Gurobi feasibility MIP를 같은 instance·예산에서 실행(보유). RoundingSat은 설치 가능하면 추가, 아니면 명시적으로 제외 사유 기재 |
| C1 | 값×변수 교차 ablation(6조합) 없음 — 두 단계 필요성의 핵심 검증 (C, O) | S1 | C3 | guide 4종 추가(LP 변수+S1 값, S1 변수+LP 값, S2 변수+LP 값, LP 변수+S2 없음). 21×60 100개×3 seed |
| C2 | M1 실험 교란(상태 2,691 vs 10,000) (C, O) | S2 | C3 | 10×25 상태 2,691개로만 학습한 크기 대조 Stage 1 추가 |
| C3 | soft label vs sampled 0/1 label 같은 예산 비교, oracle marginal 탐색 (O) | S2 | C3 | hard-label 학습은 B2에 포함. oracle marginal 탐색은 10×25·18×50에서 노드마다 열거(18×50은 시간 제한)로 상한 곡선 제시 |
| D1 | 학습 분포(feasible 상태)와 탐색 분포(대부분 반증 subtree) 불일치; feasible/infeasible 노드 분리 분석 (C) | S2 | C3 | DFS 계측: 해 경로 노드 vs 반증 subtree 노드로 분해, arm별 반증 subtree 크기 분포·정책 행동(fail-first 해석 검증) 보고. `pv_root_diag`의 dead-child 반증 비용(규칙 3,495 vs policy 2,143)을 확장 |
| D2 | LP-confidence 순서로 만든 학습 상태 = 분포 이동, DAgger 없음 (C) | S3 | C4 | 이번 사이클은 서술(한계)로 처리, 선택 사항으로 network 자체 순서 prefix 상태 재생성 |
| D3 | τ=1 학습 vs greedy 배포 불일치 (C) | S3 | C1 (+C2) | 서술 추가; 선택적으로 τ∈{0.5,1}로 배포 평가 |
| E1 | "생성기가 유도한 posterior" 오류: planting은 weight n/2, 선택 단계도 분포 왜곡 (C, O) | S2 | C1 | (2)를 "S 위의 균등 측도로 정의한 marginal"로 재정의, posterior 표현 삭제, 제목 "Exact Posteriors" → "Exact Marginals" 검토. n=25의 planting 개수(n//2=12) 명시 |
| E2 | "규칙 (5)는 첫 자식이 해를 포함할 확률 최대화" 오류 (O) | S2 | C1 | "다수 값은 해를 보존하며 confidence는 보존되는 해의 비율"로 고침. 무작위 해 일치 확률 / 자식 feasible 여부 / 그 자식의 비용을 분리 서술, OPEN 변수의 값 선택도 비용에 영향 있음을 명시 |
| E3 | −log c 목적함수 정당화 부족, 전역 목적·조기 종료·미완료 rollout 처리 미기재 (O) | S2 | C1 (+C4 선택) | 전역 목적 c(root)=경로상 결정 비용의 합 관계와 DFS 성질 명시, −log c는 크기 불변 surrogate임을 인정, advantage 표준화·detach·timeout 처리 기술. 선택: raw cost 학습 1회 비교 |
| E4 | rollout timeout 처리 (O) | S3 | C1 | 위 사실 서술 |
| E5 | Proposition 1은 자명 (C, O) | S3 | C1 | 기여 목록에서 내리고 "기초 성질"로 서술 |
| E6 | LP-probing이 모든 FORCED를 찾는다는 서술 오류 (O) | S3 | C1 | "LP로 판별되는 일부만"으로 정정 |
| E7 | kernel 길이 4k는 ±1 성분일 때만 (O) | S3 | C1 | 조건 추가 |
| E8 | 알고리즘 1의 빈 행 제거 순서 (O) | S3 | C1 | 코드 순서(bound check → 행 제거)대로 서술 |
| E9 | LP 목적함수·tie-breaking·상태 정의, 재현성 정보(N, 버전, CPU, 학습 비용) (O) | S3 | C1~C2 | 부록 절 추가 |
| F1 | 배수 정의 불일치 (C, O) | S3 | C2 | 모든 배수를 "instance별 비율의 중앙값"으로 통일하고 정의 문장 추가, 표에는 중앙값 병기 |
| F2 | p값 오류 (C, O) | S3 | C2 | 전 비교 재계산(동률 처리 명시), 초록 p값 정정 |
| F3 | 잔여 50개 pooled 부호검정은 유사 반복 (C) | S3 | C2 | seed별 검정으로 교체 |
| F4 | 28×80 solved-only 중앙값 편향; PAR-2·공통 해결 paired·시간별 해결 곡선 (O) | S2 | C2 | 집계 스크립트 추가 |
| F5 | 28×80 seed 1개 (C, O) | S2 | C4 | seed 1·2 실행 (약 13시간 계산) |
| F6 | instance 단위 신뢰구간 (C, O) | S3 | C2 | bootstrap CI |
| F7 | Stage 1·M1·M2 학습 seed 1개 (C) | S2 | C3 | Stage 1 seed 2개 추가 학습·평가, M1 seed 1개 추가 |
| F8 | "20×50" random (C, O) | S3 | C2 | 18×50·21×60에서 random arm 재실행 |
| F9 | |S| 중앙값 21 vs 23 (C) | S3 | C1 | 풀 차이 명시 |
| F10 | held-out 상태 수 100/150/300 혼재 (C) | S3 | C1 | 각각의 용도 명시 |
| F11 | 학습·열거 비용 미보고 (C, O) | S3 | C2 | 로그에서 수집해 표 |
| G1 | 단일 합성 분포; 밀도·planting 비율·선택 절차 변경 검증 (C, O) | S2 | C3 | 21×60 30개씩: 밀도 0.3/0.7, planting 비율 0.3, vertex-spread 선택 없음. 재학습 없이 LP/S1/S1+2 평가 |
| G2 | 실제 비파괴검사 투영 행렬 없음 (G, C, O) | S2 | C5 | 이번 사이클 제외. 서론 응용 주장을 "동기"로 한정하고 한계로 명시. 선택: 투영 기하 흉내 낸 band 구조 행렬 1종(C3) |
| G3 | infeasible instance 배제 (G, C, O) | S2 | C4 | 범위 결정(2026-09-21)은 유지하되, 부록 실험 1건: 18×50 infeasible 30개(`gen_hard_infeasible`)에서 LP/S1/S1+2의 반증 트리 크기 비교. 사용자 결정 필요 |
| G4 | n≥70 multiplicity 미검증, 60의 27/30 (O) | S3 | C2 | 미검증 3개가 어려운 쪽인지 확인해 각주 |
| H | 참신성이 NSNet+FMSTS 결합 수준; 차별점 서술 (C, O) | S3 | C1 | 기여를 "검증된 설계 선택"(초기화·동결·신호 분리의 실험적 근거)으로 재서술 — B2·C1 결과가 근거 |
| I | 각주 placeholder, 오타, "a open", 코드 공개 선언 (C, O) | S3 | C1 | 일괄 수정 |

## 반박 논리 (실험 없이 답할 수 있는 것)
- "실용적 우위 없음": 논문의 주장은 solver 대체가 아니라 branching 방법론이며, 같은 루프의 비교는 branching 품질만 분리한다. 다만 A1·A2로 "구현 병목" 주장을 실측으로 바꾼다.
- "Proposition 1 자명": 동의. 기여가 아니라 "하나의 network를 모든 node에 쓰는 근거"로만 사용.
- "M1은 주요 구간에서 더 나쁜 모델": 동의하되, 논문의 결론은 "값 정확도를 마진 구간에서 올려도 트리가 줄지 않는다"이므로 C2의 크기 대조 실험으로 교란을 제거한 뒤 같은 결론이면 유지.
- "3 evaluation seed는 시간 잡음 반복": 맞음. node 수는 instance 표본이 통계 단위임을 명시하고 CI를 instance 단위로 제시(F6). 학습 seed는 F7로 보강.
- "격자가 약함": 우리 v9~v12 기록에서 block 40 등 강한 설정도 시도했고 60×150 이상은 붕괴했다는 기록이 있으나 현재 instance 계열에서는 재측정이 없으므로 B3로 답한다.

## 작업 순서 (레벨 → 비용 순)
- **Phase 0 (즉시)**: v24 사이클 commit·push, 논문 데이터 아카이브(아래), 새 tex 사본(날짜)에 C1 항목 일괄 반영(E1~E9, F9, F10, A3, A4, H, I).
- **Phase 1 (C2, 1일)**: F1, F2, F3, F4, F6, F8, F11, A2, B4(Gurobi), G4.
- **Phase 2 (C3, 2~3일 계산, 연쇄 스크립트)**: C1(교차 ablation) → B1(고전 규칙) → B3(강한 lattice) → D1(feasible/infeasible 분해) → C2(크기 대조) → G1(분포 변형) → C3(oracle 상한).
- **Phase 3 (C4, 3~5일 계산)**: B2(학습 baseline 4종) → A1(dense inference + 시간 재측정) → F7(학습 seed) → F5(28×80 seed).
- **Phase 4 (선택, 사용자 결정)**: G3(infeasible 부록), D2(DAgger), E3 raw-cost 비교, D3 τ 평가.
- 각 Phase 종료 시 version.md 기록, 결과가 나오는 대로 tex 반영(날짜 사본 규칙).

## 데이터 보관
- 대상: 평가 instance 6세트(`instances/v22test_*`, `v24test_*`, 약 2 MB), 학습 상태 파일(`runs/v22/cond_10x25_*`, `runs/v24/cond_*`, `pool_*`, 약 60 MB), 체크포인트(`runs/v22/model_n25`, `runs/v24/{M1,M2}`, `runs/v24/r2*`, `runs/revision/ctrl_*`), 결과 JSON(`runs/v24/summary_*`, `runs/v24/*_ahl*_*/inst-*_r.json`, `runs/revision/*.json`, `runs/v23/aggregate.json`), 학습 로그.
- 방법: `proposed_src/util/archive_paper_data.py`가 위 파일을 `archive/261001_paper_data/`로 복사하고 `manifest.json`(경로·sha256·생성 스크립트·용도(표/그림 번호))을 만든 뒤 tar.gz로 묶는다. tar는 git 밖(gitignore), manifest는 git에 커밋. 재구성 절차(생성기 seed, 학습 명령)를 `docs/data_manifest_261001.md`에 기록.
- 크기: runs 전체 7.4 GB 중 논문 관련 약 350 MB 예상 → tar.gz 약 100 MB.

## 승인 요청 사항
1. 위 순서로 v25 사이클 시작 (Phase 0~3 기본, Phase 4는 선택).
2. v24 사이클 종료로 간주해 commit + push 실행 여부.
3. G3(infeasible 부록 실험) 포함 여부.
4. 제목의 "Exact Posteriors" → "Exact Marginals" 변경 여부.
