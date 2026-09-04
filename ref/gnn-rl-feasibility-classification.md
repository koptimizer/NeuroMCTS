# 문헌 검토: GNN/RL 기반 Feasibility(결정문제) 판별 — BLS 프로젝트 적용 가능성

**Slug:** `gnn-rl-feasibility-classification` · **작성일:** 2026-07-22 · **상태:** researcher 1차 조사(주요 출처 직접 확인) + reviewer 검토 완료, 지적사항(3.1절 논리 오류 등) 반영

## 0. 맥락과 질문

본 프로젝트는 지금까지 solution 구성(construction)에 학습을 결합하는 설계를 시도해왔으나, feasibility 판별(0/1 해가 존재하는가라는 예/아니오 결정) 자체를 GNN/RL로 직접 푸는 시도는 7회 독립적으로 실패했다(단발 스칼라 특징, 다회 프로브 집계 특징, 사전학습 GNN 임베딩, NeuroSAT식 짝 데이터, NeuroSAT식 이중노드 아키텍처 — 전부 AUC 0.50~0.60에서 정체, sanity check로 구현 버그 배제 완료; [[v10-attribution-findings]] [[neurosat-paired-dualnode-findings]]).

사용자는 이제 프로젝트의 핵심 novelty를 "학습"에 명확히 두기로 하고, 다음 조건 하에 feasibility 판별을 학습의 1차 목표로 재설정하고자 한다:
1. Solution 구성은 잠시 보류.
2. **필수조건 A(속도)**: 기존 휴리스틱/수리적 최적화(MILP, CP-SAT, 격자/BKZ)보다 추론이 빨라야 한다.
3. **필수조건 B(zero-shot 크기 강건성)**: 학습하지 않은(더 큰) 인스턴스 크기에서도 성능이 유지되어야 한다.
4. RL/GNN으로 결정문제 feasibility를 판별한 선행 사례를 조사하고 우리 문제 적용 가능성을 정리.

본 조사는 위 4개 질문에 답하기 위해 researcher 에이전트가 수집한 1차 출처(arXiv/DOI, 대부분 초록·본문 HTML 직접 확인)를 종합한다.

## 1. 핵심 발견

### 1.1 NeuroSAT 이후 계보와 정량적 벤치마크

**NeuroSAT**(Selsam et al., ICLR 2019)[1]은 단일 비트 지도(SAT/UNSAT)만으로 학습된 메시지패싱 GNN이 SAT 분류와 해 디코딩을 동시에 수행함을 보였다. 다만 원 논문이 강조하는 "일반화"는 **추론 시 메시지패싱 반복 횟수를 늘려 더 크고 어려운 문제의 만족 배정(witness)을 찾는 것**이지, held-out 대형 인스턴스에서의 **분류 정확도**가 유지된다는 의미가 아니다 — 이 구분이 본 조사에서 가장 중요한 포인트다.

**G4SATBench**(Li, Guo, Si, TMLR 2024)[2]는 여러 GNN-SAT 아키텍처를 통제된 조건에서 비교한 벤치마크로, 본 조사에서 가장 직접적으로 유용한 정량 근거를 제공한다(arXiv HTML에서 직접 확인):
- 학습분포 내(easy SR/3-SAT) 분류 정확도: NeuroSAT ≈ 96.0~96.3%, GGNN ≈ 96.25~96.75%.
- 동일 분포 내에서도 medium 난이도로 가면 SR ≈78%, 3-SAT ≈85%로 하락.
- **크기 전이(easy→medium, 변수 수 약 4~5배)에서 분류 정확도가 15~25%p 추가로 하락**(저자 결과표에서 읽은 값, 원문에 이 구간이 요약 문장으로 명시되어 있지는 않음 — 재인용 시 본문 표 재확인 권고)하며, 저자들은 "GNN 모델은 학습 데이터보다 큰 인스턴스로의 일반화 능력이 제한적"이라고 명시적으로 결론짓는다.
- 저자들의 해석: GNN은 "greedy local search와 유사한 것을 학습하지만 latent space에서 backtracking을 학습하지 못한다."

NeuroCore[9]·NLocalSAT[10] 등 후속작은 분류기의 출력을 **최종 판별이 아니라 CDCL/SLS 솔버의 휴리스틱 신호(변수 활동도, 초기화)로 재사용**하는 쪽으로 방향을 틀었다 — 조사된 사례가 이 둘뿐이라 "업계의 합의"라 부르긴 이르지만, 적어도 이 두 설계는 "학습된 분류기 자체를 최종 답으로 쓰지 않는다"는 선택을 했다.

### 1.2 결정적 근거 — GNN의 표현력 한계 (MILP feasibility)

**Chen, Liu, Wang, Lu, Yin (ICLR 2023)**[3]은 우리 문제와 가장 직접적으로 관련된 이론 결과를 제공한다: **"모든 GNN이 동일하게 취급할 수밖에 없는 feasible MILP와 infeasible MILP 쌍이 존재한다"**는 것을 Weisfeiler-Leman 동형성 논증으로 증명한다. 이는 구현이나 학습량의 문제가 아니라 **GNN 아키텍처 자체의 표현력 한계**이며, 문제가 "unfoldable"(대칭이 없는 구조)이거나 랜덤 노드 특징을 주입하지 않는 한 근본적으로 회피할 수 없다.

밀도 0.5의 이진 계수 행 다수로 구성된 market-split류 인스턴스(본 프로젝트의 A)는 계수가 대부분 0/1로 반복되고 행/열 간 대칭이 높아 이 정리가 지목하는 "foldable" 구조에 해당할 가능성이 있다는 **가설**을 세울 수 있다 — 단, [3]은 그런 쌍이 *존재*함을 증명했을 뿐 market-split류 분포의 몇 %가 실제로 foldable인지는 정량화하지 않으므로, 이는 검증되지 않은 추론이다.

중요한 것은 [3]이 순수한 불가능성 정리가 아니라는 점이다: 저자들은 "unfoldable MILP로 제한하거나 **랜덤 노드 특징을 주입하면**, 지정된 정밀도까지 feasibility를 신뢰성 있게 예측하는 GNN이 존재한다"는 **구성적(긍정) 결과**도 함께 제시한다. 즉 [3]은 "GNN은 안 된다"가 아니라 "표준 GNN은 특정 대칭 하에서 한계가 있고, 대칭을 깨는 특징을 추가하면 그 한계가 원리적으로 해소된다"는 처방까지 포함한 결과다 — 우리가 아직 시도하지 않은 처방이다(§4 참고).

### 1.3 왜 CNF는 되고 dense 0/1 선형계는 안 되는가

- [6][7]: NeuroSAT류 GNN이 실제로 배우는 것은 belief propagation/SDP 완화에 가까운 근사, 또는 "support"(저지지 변수 뒤집기) 같은 **얕은 고전 휴리스틱**이다 — CNF의 리터럴·절 단위 국소 구조가 이런 얕은 신호를 명시적으로 제공하기 때문이다.
- 조밀한 선형 등식(각 행이 수십~수백 개 변수의 합)은 이런 국소·이산적 단서를 제공하지 않는다 — [3]의 이론과 정합적인 정성적 설명이다.
- [4] Geometric Perspective on GNN-SAT(2025, arXiv 2508.21513)는 k-SAT 그래프가 음의 곡률(negative curvature)을 가지며 난이도가 커질수록 곡률이 더 음수화되어 **oversquashing**(장거리 의존성이 고정 크기 임베딩에 압축되지 못함)이 발생함을 보이고, 이 곡률이 일반화 오차를 예측한다고 보고한다. [5] StructureSAT도 독립적으로 그래프 구조 지표와 일반화 실패의 상관관계를 보고한다.

### 1.4 RL이 feasibility 판정 자체를 출력하는 사례 — 사실상 부재

Graph-Q-SAT[8](NeurIPS 2020, 분기 정책, 5배 큰 변수 수로 정책 전이), NLocalSAT[10](SLS 초기화), RL-MILP[13](해 구성) 등 RL-for-CO 문헌을 조사했으나(3편 표본, 전수조사 아님), **RL 에이전트의 출력이 곧 feasibility 판정(예/아니오)인 사례는 발견하지 못했다.** 전부 완전한 솔버 내부의 탐색/분기 정책이며, 최종 판정은 여전히 심볼릭 솔버가 내린다. 즉 "예/아니오를 RL이 직접 판별"하는 문헌은 조사 범위 내에서 공백이다 — 이는 시도할 가치가 없다는 뜻이라기보다, 선례로부터 얻을 수 있는 설계 지침이 거의 없다는 뜻이다.

### 1.5 속도 비교 — 문헌에 직접적 근거 없음

GNN feasibility 분류기가 MILP/SAT/격자 솔버 대비 **동일 인스턴스에서 wall-clock으로 더 빠르다**는 것을 여러 크기에 걸쳐 직접 보고한 1차 문헌을 찾지 못했다. NeuroCore는 오히려 온라인 GNN 추론 비용이 무겁다고 보고하며, NeuroSAT·G4SATBench 자체가 고전 솔버와 속도로 경쟁하는 것을 목표로 하지 않는다고 명시한다. **필수조건 A(속도 우위)는 기존 문헌에서 확인도 반증도 되지 않은 공백**이다. (참고: 우리 문제의 경우 LP relaxation이 infeasible이면 Farkas 보조정리로 즉시·저비용·엄밀하게 infeasibility를 증명할 수 있다는 것이 이미 파이프라인에 내장돼 있다 — 학습 기반 분류기가 속도로 이겨야 할 상대는 "무거운 MILP 솔버"만이 아니라 이 값싼 규칙도 포함된다는 점을 감안해야 한다.)

## 2. 종합 표

| 축 | 문헌 근거 | 본 프로젝트 요구조건 대비 |
|---|---|---|
| 학습분포 내 분류 정확도 | NeuroSAT/G4SATBench 78~96% (난이도 의존) | 참고 가능하나 CNF 특화 |
| **크기 zero-shot 분류 정확도** | G4SATBench(CNF 한정): 15~25%p 하락, "제한적 일반화" 명시 | 필수조건 B에 부정적 근거(단, CNF→우리 문제 외삽은 미검증) |
| 표현력 이론 | GNN이 구분 못 하는 feasible/infeasible MILP 쌍 존재 증명 + 랜덤 특징 처방(효과 있음) | 우리 구조가 사각지대일 가설 + **미시도 처방 존재** |
| RL이 판정 자체를 출력 | 사실상 문헌 부재(전부 솔버 내부 정책) | 참고할 선례 없음 — 백지에서 설계해야 함 |
| 속도(고전 솔버 대비) | 직접 비교 문헌 없음 | **필수조건 A는 미검증 공백** — 우리가 직접 만들어야 할 증거 |

## 3. 우리 프로젝트에 대한 시사점

1. **7번의 부정 결과는 우연이 아니라 이론적으로 설명 가능하다(단, 반증은 아니다).** [3]의 WL-표현력 정리는 표준 GNN 아키텍처를 아무리 정교하게 바꿔도(우리가 시도한 이중노드 등) 대칭이 있는 한 넘을 수 없는 한계가 있음을 보인다 — 우리의 부정 결과들과 정성적으로 정합적이다. 다만 [3]은 처방(랜덤 노드 특징 주입)도 함께 제시하며 그 처방이 통했다고 보고한다. **이 처방은 우리가 아직 시도하지 않았다.** v10의 "다회 프로브 집계 특징"은 GNN 내부에 랜덤/비대칭 노드 특징을 주입한 것이 아니라, 별도의 집계 통계를 MLP/로지스틱 분류기에 입력한 것이므로 [3]이 말하는 대칭 파괴 메커니즘과 다르다 — 우리의 무신호 결과는 [3]의 처방에 대한 반증이 아니라, 아직 검증되지 않은 별개의 질문으로 남아 있다.
2. **필수조건 B(zero-shot 크기 강건성)는 문헌 전체에서 "분류" 과제에 대해 지지되지 않는다.** 가장 체계적인 벤치마크(G4SATBench)가 정반대(15~25%p 하락)를 보고한다. "크기에 강건하다"는 유명한 주장들(NeuroSAT의 일반화, LEHD 등, [[zero-shot-size-generalization-rl]] 참고)은 전부 **construction/search** 과제에 대한 것이며 **classification**에 대한 것이 아니다 — 이 구분을 흐리면 안 된다.
3. **필수조건 A(속도)는 검증도 반증도 안 된 진짜 공백이다.** 우리가 이걸 직접 측정해 보고하면, 성공하든 실패하든 (문헌에 없는) 새로운 증거가 된다.

## 4. 결론 및 권고

**정직한 평가**: 문헌 중 유일한 체계적 벤치마크(G4SATBench)는 사용자가 제시한 필수조건 B(zero-shot 크기 강건성)와 반대 방향(분류 정확도 15~25%p 하락)을 보고한다 — 다만 이는 **CNF/SAT 한정** 벤치마크이며, 우리 문제(조밀 0/1 선형계)에 대한 직접 증거는 아니다. 필수조건 A(속도)는 문헌에 확인도 반증도 없는 순수 공백이다. 표현력 이론[3]은 우리 구조가 GNN의 사각지대일 수 있다는 **가설**을 뒷받침하지만, 동시에 그 사각지대를 벗어나는 처방(랜덤 노드 특징)도 제시하며 그 처방은 우리가 아직 시도하지 않았다 — 즉 "이미 시도해서 실패했다"고 말할 수 있는 근거는 없다.

요약하면 문헌은 (a) 가장 가까운 벤치마크(SAT 한정)가 필수조건 B에 부정적이고, (b) 필수조건 A는 검증된 바 없으며, (c) 우리 구조에 특화된 처방(랜덤 특징 주입)은 이론상 존재하지만 미시도라는 세 가지를 알려준다. "학습으로 feasibility 판별은 불가능하니 포기하라"는 결론은 문헌에서 도출되지 않는다 — 오히려 시도해볼 구체적이고 아직 검증되지 않은 지점이 있다는 것이 더 정확한 요약이다. 다만 "NeuroSAT처럼 하면 될 것"이라는 막연한 전제만으로 재시도하는 것은 근거가 약하며, 아래처럼 구체적 처방을 겨냥해야 한다.

**권고**:
- [3]이 실제로 검증한 처방(랜덤 노드 특징 주입, 혹은 행/열 순서 의존 위치 인코딩)을 GNN 내부에 직접 적용해보는 것 — 우리가 시도한 7가지 중 어느 것도 이 정확한 메커니즘을 테스트하지 않았다.
- 속도(필수조건 A)를 실제로 측정해 문헌의 공백을 직접 메우는 것 — 실패하더라도 그 자체로 새로운 정보다.
- 우리 문제가 CNF가 아니라는 점을 살려, G4SATBench의 부정적 결과가 우리 문제에도 그대로 이전되는지 별도로 확인(현재는 확인되지 않은 외삽).

## 5. 참고문헌

1. Selsam, Lamm, Bünz, Liang, de Moura, Dill (2019). NeuroSAT: Learning a SAT Solver from Single-Bit Supervision. ICLR. https://arxiv.org/abs/1802.03685
2. Li, Guo, Si (2024). G4SATBench: Benchmarking and Advancing SAT Solving with Graph Neural Networks. TMLR. https://arxiv.org/abs/2309.16941
3. Chen, Liu, Wang, Lu, Yin (2023). On Representing Mixed-Integer Linear Programs by Graph Neural Networks. ICLR. https://arxiv.org/abs/2210.10759
4. (2025). A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers. arXiv. https://arxiv.org/abs/2508.21513 *(초록 수준 확인)*
5. StructureSAT: A Structure-based SAT Dataset for GNN Generalisation Study. arXiv. https://arxiv.org/abs/2502.11410 *(초록 수준 확인)*
6. Understanding GNNs for Boolean Satisfiability through Approximation Algorithms. arXiv. https://arxiv.org/abs/2408.15418 *(초록 수준 확인)*
7. Concept Learning for Algorithmic Reasoning: SAT-solving GNNs (2025). Information Sciences. https://www.sciencedirect.com/science/article/pii/S0020025525008904 *(스니펫 수준 확인)*
8. Kurin, Godil, Whiteson, Catanzaro (2020). Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver? (GQSAT). NeurIPS. https://arxiv.org/abs/1909.11830
9. Selsam, Bjørner (2019). Guiding High-Performance SAT Solvers with Unsat-Core Predictions (NeuroCore). SAT 2019. https://arxiv.org/abs/1903.04671
10. Zhang et al. (2020). NLocalSAT: Boosting Local Search with Solution Prediction. arXiv. https://arxiv.org/abs/2001.09398 *(스니펫 수준 확인)*
11. Neural Approaches to SAT Solving: Design Choices and Interpretability. arXiv. https://arxiv.org/abs/2504.01173 *(스니펫 수준 확인)*
12. Learning Better Representations From Less Data for Propositional Satisfiability. arXiv. https://arxiv.org/abs/2402.08365 *(스니펫 수준 확인, 정량 수치 미추출)*
13. RL-MILP Solver: A Reinforcement Learning Approach for Solving Mixed-Integer Linear Programs with Graph Neural Networks. arXiv. https://arxiv.org/abs/2411.19517 *(스니펫 수준 확인)*

## 검증 로그

- researcher 에이전트가 [1][2][3][4][6][8][9]를 초록/본문 HTML 수준에서 직접 확인(arXiv 접속), dblp/OpenReview/ar5iv로 존재 여부 교차 검증.
- [5][7][10][11][12][13]은 존재는 확인했으나 본문 정독은 못 함(스니펫 수준) — 정량 수치 재확인 필요 시 본문 재조사 권고.
- 검색 중 발견됐으나 존재를 확인할 수 없었던 arXiv ID(2602.18419, 2603.07176, 2604.15448, 2602.14772, 2512.04475)는 인용에서 배제.
- 미해결: 필수조건 A(속도)에 대한 1차 문헌 근거는 결국 찾지 못함 — 이 조사의 결론이자 한계.
