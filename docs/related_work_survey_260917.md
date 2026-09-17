# 선행연구 조사 (2026-09-17): 우리와 유사한 접근, 인용 반영, 노벨티 영향

조사 방법: 웹 검색(arXiv·NeurIPS·AAAI·JAIR)과 원문 PDF 텍스트 추출로 확인. "확인" 열은 원문에서 직접 확인한 사실만 적음.

## 1. Stage 1(정확한 marginal을 학습 target으로)과 겹치는 연구

| 연구 | 하는 일 | 우리와 겹치는 점 | 다른 점 | 노벨티 영향 |
|---|---|---|---|---|
| **NSNet** (Li & Si, NeurIPS 2022) | BP 구조의 GNN으로 SAT 해집합의 marginal을 추론. **ALLSAT solver로 모든 만족 배정을 열거해 평균 낸 정확한 marginal을 ground truth로 KL 손실 학습**(원문 확인). 추론 시 rounding + local search 초기값, decimation도 언급 | **정확한 열거 기반 marginal 라벨**이라는 Stage 1의 핵심 아이디어가 이미 있음 | root 상태만; complete search 안의 conditional marginal 아님; 변수 확정(rounding/decimation)에 사용, branching guidance 아님; SAT | **큼** — "exact marginals as supervision"을 우리 발명으로 쓸 수 없음. 우리 novelty는 (i) reduction 항등식으로 tree 내부 conditional marginal로 옮긴 것, (ii) FORCED/OPEN 분할과 in-tree vs root 통제 비교, (iii) 확정 없는 guidance |
| **Counting-Based Search / maxSD** (Pesant, Quimper, Zanarini, JAIR 2012; Zanarini & Pesant 2009) | 제약별로 해 밀도(solution density: 그 제약의 해 중 x=v인 비율)를 정확/근사 계산해 밀도 최대인 변수-값으로 분기 | "가장 편향된 marginal의 변수로 분기"라는 Stage 1 규칙의 고전적 원형. 제약별 exact counting | 제약 단위(local) 밀도; 학습 없음; 전역 해집합의 conditional marginal 아님 | 중간 — 규칙 자체는 고전. 우리는 전역 posterior를 amortize한 것 |
| **BP-guided decimation** (Montanari, Ricci-Tersenghi, Semerjian 2007; Coja-Oghlan 2011) | 근사 marginal(BP)로 가장 편향된 변수를 고정하고 marginal을 재계산하며 반복 | "conditional marginal을 매 단계 다시 보고 가장 확신하는 변수부터" 구조가 동일 | backtracking 없는 고정(불건전); 근사 marginal; random k-SAT 이론 | 중간 — 개념적 조상으로 인용 필요 |
| **MIP-GNN** (Khalil, Morris, Lodi, AAAI 2022) | (near-)optimal 해들의 성분 평균(variable bias)을 GNN으로 예측해 node selection·warm start에 사용 | 해집합 통계를 GNN으로 예측해 solver를 안내 | 최적화 문제, 수집된 해의 편향(정확한 posterior 아님), branching 아님 | 작음 |
| **Neural Diving** (Nair et al. 2020) | 목적값으로 가중한 해 분포를 조건부 독립 Bernoulli로 학습, SelectiveNet으로 일부 변수 고정 후 SCIP | 변수별 marginal 예측 + 탐색 결합 | 변수 확정(불건전), MIP 최적화 | 작음 |
| **Predict-and-Search** (Han et al., ICLR 2023) | GNN이 marginal 확률 예측 → trust region 안에서 탐색 | 동일 | 최적화, 수집 해 기반 marginal, 확정/제한 | 작음 |
| **Neuro#** (Vaezipoor et al., AAAI 2021) | #DPLL 기반 #SAT solver의 branching literal을 GNN(**residual formula 입력**)으로 선택, **evolution strategies**로 branching 횟수 최소화(원문 확인: 보상 = 해결 시 1, 아니면 −penalty) | "축소된 잔여 공식을 입력으로 tree 내부에서 예측" — 우리 reduction 항등식과 같은 실천 | 지도 marginal 없음; ES(episodic); #SAT | 중간 — Stage 2의 tree-size 목적과도 겹침 |

## 2. Stage 2(subtree 비용으로 branching 순서 RL)와 겹치는 연구

| 연구 | 하는 일 | 겹치는 점 | 다른 점 | 노벨티 영향 |
|---|---|---|---|---|
| **FMSTS** (Etheve et al., CPAIOR 2020) | **Q-value = subtree 크기**(관측 가능), **DFS에서는 각 subtree 최소화가 전체 tree 최소화와 동치(Proposition 2)**, approximate Q-learning, **from scratch**(원문 확인) | 우리 critic(log subtree 비용)과 DFS 논거가 동일 | 초기화 없음(from scratch), MILP B&B, Q-learning | **큼** — critic 아이디어와 DFS 정당화는 FMSTS 것. 우리 것은 posterior 초기화 + 동결 + 순서만 학습 |
| **Tree MDP** (Scavuzzo et al., NeurIPS 2022) | B&B를 tree MDP로 형식화, tree policy gradient, 결정별 subtree return | 결정 단위 credit assignment 동일 | from scratch, MILP | 큼 (이미 인용) |
| **Retro branching** (Parsonson et al., AAAI 2023) | search tree를 subtree별 경로로 회고적으로 분해해 RL | subtree 단위 학습 | MILP, from scratch, DQN | 중간 |
| **SORREL** (AAAI 2025), PPO/IL hybrids (2025), PEBSI(2026), ReviBranch(2025) | 준최적 시연 + RL, IL 초기화 후 PPO | "지도(모방) 초기화 후 RL"은 표준 관행 | expert = strong branching 등; 우리 expert = 정확한 posterior 규칙 | 중간 — "초기화 후 fine-tuning" 자체는 새롭지 않음 |
| **Impact-based search** (Refalo, CP 2004), **fail-first** (Haralick & Elliott 1980) | 탐색공간 축소량(impact)이 큰 변수부터 / 가장 빨리 실패할 변수부터 | 우리 RL이 배운 것으로 추정되는 목적의 고전적 진술 | 학습 없음 | 인용 필요(해석 근거) |
| **Cappart et al.** (AAAI 2021) | CP 안에서 GNN+DQN으로 **value ordering** 학습 | CP 탐색 내부 학습, 값 선택 | 값 선택을 RL로; 우리는 값은 posterior, 변수는 RL | 작음 |
| **Chu & Stuckey** (CPAIOR 2015) | 해로부터 value heuristic 학습 | 값 선택 학습 | 선형 모델 | 작음 |
| **Song et al.** (2022), RL for FDS tree-size (2025) | CSP 변수 순서 DQN / FDS를 MAB로 tree 크기 최소화 | tree 크기 목적 | 스케줄링 | 작음 |

## 3. 문제 계열(market split)의 고전적 SOTA

| 연구 | 내용 | 영향 |
|---|---|---|
| **Wassermann 2025** (arXiv 2508.08702, MPC 2026), solvediophant | lattice enumeration으로 QOBLIB market split을 **m=14까지** 해결(단일 CPU, (14,130,50) 39시간). branch-and-cut은 m=7, GPU Schroeppel–Shamir(2025)는 m=11 | 정식 벤치마크에서는 lattice enumeration이 SOTA. 우리 21×60(|S| 통제, m/n≈0.35)은 다른 영역이며 우리 기여는 branching 학습 방법론이지 기록 갱신이 아님을 명시해야 함 |
| Aardal et al. 2000, Cornuéjols & Dawande 1999 | (이미 인용) | — |

## 4. 노벨티 판정

**떨어지는 부분 (정직하게 적을 것)**
1. "정확한 열거 marginal을 지도 신호로" — NSNet이 SAT에서 이미 함(ALLSAT 열거, KL). → 기여 (1)의 문구를 "target은 NSNet과 같고, 새로운 것은 conditional·in-tree·guidance-only 사용과 FORCED/OPEN 분석"으로 고침.
2. "subtree 비용을 value로, DFS에서 국소=전역" — FMSTS·tree MDP가 이미 확립. → Stage 2의 기여를 "posterior 규칙에서 출발해 marginal을 동결하고 순서만 fine-tuning; 그래서 개선을 순서에 귀속시킬 수 있음"으로 좁힘.
3. "가장 편향된 marginal로 분기" — counting-based search(maxSD)·BP-guided decimation의 고전 규칙. → 인용하고 amortization 관점으로 위치.
4. "지도 초기화 후 RL" — 표준 관행(SORREL 등).

**남는 노벨티**
- reduction 항등식으로 conditional marginal을 tree 전체에 전달 + FORCED/OPEN 분할 + in-tree vs root 통제 비교 (같은 pool·label·수로 root 학습이 rounding보다 못함을 보인 것).
- 두 신호가 다른 것을 잡는다는 실증: 분포 정렬 부정 결과, R0 잔차, 95% 불일치, 덜 확신하는 변수 선호 — 이것이 논문의 중심 주장이며 선행연구에 없음.
- integer equality feasibility(BLS/market-split) 영역에서의 학습 branching + lattice cascade + 100개·3 seed·CP-SAT 대조의 통제 평가.

## 5. 논문 반영 사항
- 관련연구 절: "Marginals of the solution set as a learning target" 단락 신설(NSNet, maxSD, BPGD, MIP-GNN, Neural Diving, Predict-and-Search), "Learning to branch" 단락에 FMSTS 정확 기술·retro branching·Neuro#·SORREL·impact-based·Cappart 추가, "Complete solvers" 단락에 Wassermann 2025 추가.
- 기여 (1)·(2) 문구 수정(위 4.), 한계 절에 "target의 novelty는 정의가 아니라 사용에 있다" 단락 추가.
- 참고문헌 11건 추가. 4개 판본(260917 EN/KR, 상세판 EN/KR) 동기화.
