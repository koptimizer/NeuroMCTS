# 문헌 검토: 인스턴스 크기에 대한 Zero-Shot 일반화 — RL·Neural CO·GNN 이론

**Slug:** `zero-shot-size-generalization-rl` · **작성일:** 2026-07-10 · **상태:** 인용 URL 검증(researcher 접속 확인) — 정량 수치는 초록 수준 확인이며 본문 정독 필요 항목은 표기. 본 파일은 v7 설계의 근거 문서로, 표준 워크플로우 중 verifier URL 검증까지 수행하고 reviewer 검토는 v7 보고서 단계로 이월함.

## 0. 맥락과 질문

v6 사이클에서 확인된 사실: (a) self-play RL이 in-distribution에서도 사전학습 대비 test 성능을 악화시켰고, (b) 크기 증가에 따라 성능이 붕괴했다(10×25 포화 → 20×50 sol 9.6% → 40×100 2.3%). 본 검토의 질문: **학습하지 않은 크기로 zero-shot 전이되는 학습 신호와 구조는 무엇인가?**

## 1. 핵심 교훈 5가지 (합의점)

**L1. 학습 신호는 self-play가 아니라 모방이 크기 전이의 실증 경로다.** learn2branch(Gasse et al. 2019)는 strong branching **모방만으로** 학습보다 훨씬 큰 MILP에 전이했고, RL 분기(tree MDP, Scavuzzo et al. 2022)조차 sparse reward/credit assignment 때문에 소형 문제에 한정됨을 인정한다. v6의 self-play 실패는 이 문헌 패턴과 정합적이다.

**L2. Test-time 반복 스케일링은 파라미터 변경 없는 크기 대응 수단이다.** NeuroSAT(Selsam et al. 2019)은 학습 32회 반복의 메시지 패싱을 추론 시 늘리는 것만으로 더 크고 어려운 문제로 일반화했다. LEHD(Luo et al. 2023)의 RRC도 같은 원리(추론 반복 증가). 우리 GNN은 층 간 가중치를 공유하는 재귀 구조이므로 추론 시 반복 수를 크기에 비례해 늘릴 수 있다.

**L3. 크기 외삽은 국소 통계의 불변성에 달려 있다.** Yehudai et al. (2021): 국소 구조(차수 분포)가 크기에 의존하면 "작은 그래프 성공/큰 그래프 실패"의 나쁜 global minima가 존재. Xu et al. (2021): ReLU 네트워크는 분포 밖에서 선형으로 수렴하므로, 비선형 관계(잔차 크기 등)는 학습에 맡기지 말고 **특징으로 명시 주입**해야 외삽이 된다.

**L4. 구조-알고리즘 정렬(algorithmic alignment)이 외삽 표본복잡도를 결정한다.** Xu et al. (2020): GNN이 대상 알고리즘(DP/고정점 반복)과 정렬될수록 일반화가 좋다. 제약 전파는 고정점 반복이므로, 메시지 패싱이 전파 연산을 흉내 내도록 특징·구조를 맞추는 것이 유리하다.

**L5. 분포 다양성 + 우선순위 커리큘럼이 ZSG의 기본기다.** Procgen(Cobbe et al. 2020)·Kirk et al. (2023) 서베이: 다양한 절차 생성 분포 없이는 일반화 정책이 나오지 않는다. PLR(Jiang et al. 2021)은 학습 잠재력 기반 레벨 우선 재생으로 테스트 리턴 +76%(Procgen) — 크기를 "레벨"로 보는 커리큘럼 설계의 근거.

## 2. 이견/한계

- LEHD·BQ-NCO의 크기 외삽은 지도학습(+구조 선택) 기반이며 100→1000노드급 라우팅 문제에서의 결과다. 제약충족(BLS)으로의 이식은 검증되지 않았다(추론).
- Yehudai의 반례는 "국소 통계 통제 없이는 실패 가능"이지 "항상 실패"가 아니다 — 우리 생성기는 밀도 0.5·행차수 분포가 크기와 함께 변하므로(행차수 ≈ n/2 증가) 정확히 위험 조건에 해당한다. 차수 정규화 특징이 이를 완화하는지가 실험 질문이다.
- 정량 수치(각 논문의 gap %, 전이 크기)는 초록 수준 확인 — 보고서 인용 시 본문 재확인 필요.

## 3. v7 적용 설계 (분석 및 계획)

| 교훈 | v7 반영 |
|---|---|
| L1 | self-play 제거. **전파 영향(propagation-impact) 전문가**(변수별로 양쪽 값을 probe해 고정 변수 수/모순 검출을 점수화 — strong branching의 feasibility 대응물)를 소형 인스턴스에서 라벨 생성, 정책 head가 모방 |
| L2 | 메시지 패싱 반복 수를 추론 인자로 분리(`--mp_iters`), 크기 비례 스케일(기본 8×⌈n/25⌉); MCTS/DFS 예산도 크기 비례 |
| L3 | 행차수 정규화 잔차(v6 도입)에 더해: 행·열 차수, n/m 비, 위반량의 명시적 비선형 특징(정규화 절대 잔차) 주입 |
| L4 | check 노드 특징을 전파 규칙의 중간량(잔여 b, 미할당 수, slack)으로 구성해 전파와 정렬 |
| L5 | 고정 3크기 대신 **연속 크기 분포** m~U[6,20], n=2.5m로 학습(전 인스턴스 fresh); 실패율 기반 크기 재가중(PLR-lite) |
| 검증 | **zero-shot 프로토콜**: 학습은 m≤20까지만, 테스트는 20×50(경계)·40×100(외삽)·가능 시 60×150 — CLRS의 OOD 평가 관례 준용 |

## 4. 참고문헌 (URL은 researcher 접속 확인)

1. Kirk, Zhang, Grefenstette, Rocktäschel (2023). A Survey of Zero-shot Generalisation in Deep RL. JAIR 76. https://arxiv.org/abs/2111.09794
2. Jiang, Grefenstette, Rocktäschel (2021). Prioritized Level Replay. ICML. https://arxiv.org/abs/2010.03934
3. Cobbe, Hesse, Hilton, Schulman (2020). Leveraging Procedural Generation to Benchmark RL. ICML. https://arxiv.org/abs/1912.01588
4. Luo et al. (2023). Neural CO with Heavy Decoder (LEHD). NeurIPS. https://arxiv.org/abs/2310.07985
5. Drakulic et al. (2023). BQ-NCO: Bisimulation Quotienting. NeurIPS. https://arxiv.org/abs/2301.03313
6. Yehudai et al. (2021). From Local Structures to Size Generalization in GNNs. ICML. https://arxiv.org/abs/2010.08853
7. Xu et al. (2020). What Can Neural Networks Reason About? ICLR. https://arxiv.org/abs/1905.13211
8. Xu et al. (2021). How Neural Networks Extrapolate. ICLR. https://arxiv.org/abs/2009.11848
9. Gasse et al. (2019). Exact Combinatorial Optimization with GCNN (learn2branch). NeurIPS. https://arxiv.org/abs/1906.01629
10. Scavuzzo et al. (2022). Learning to Branch with Tree MDPs. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2022/file/756d74cd58592849c904421e3b2ec7a4-Paper-Conference.pdf
11. Selsam et al. (2019). NeuroSAT. ICLR. https://arxiv.org/abs/1802.03685
12. Bengio, Lodi, Prouvost (2021). ML for CO: A Methodological Tour d'Horizon. EJOR 290. https://arxiv.org/abs/1811.06128
13. Manchanda et al. (2022). On the Generalization of Neural CO Heuristics. https://arxiv.org/abs/2206.00787
14. Veličković et al. (2022). The CLRS Algorithmic Reasoning Benchmark. ICML. https://arxiv.org/abs/2205.15659
15. Georgiev et al. (2023). SALSA-CLRS. https://arxiv.org/abs/2309.12253
