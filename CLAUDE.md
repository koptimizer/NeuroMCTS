# 프로젝트 개요
본 프로젝트는 이진연립방정식을 neuro-symbolic 강화학습 기반의 접근법으로 해결하고자 하며, 기존의 수리적 최적화 및 휴리스틱 대비 문제 인스턴스 사이즈에 강건하고, 빠른 추론 속도를 강점으로 하는 모델을 연구/개발하는 것을 목표로 한다.  

이때 우리가 목표로 하는 타겟 문제는 레이저 기반 비파괴 검사를 위한 이미지 복원 문제로써, 이를 가장 일반화한 형태의 이진연립방정식 문제를 다음과 같이 정의된다.

## 문제 정의: 이진 연립방정식 (Binary Linear Systems, BLS)

본 정의는 실수 체계의 선형대수학적 성질과 이진 변수 제약 조건이 결합된 이산 최적화 문제를 형식화합니다.

## 1. 수학적 공식화 (Mathematical Formulation)

계수 행렬 $A \in \{0, 1\}^{m \times n}$와 타겟 벡터 $\mathbf{b} \in \mathbb{Z}^m$가 주어졌을 때, 다음 연립 방정식을 만족하는 이진 해 벡터 $\mathbf{x}^* \in \{0, 1\}^n$를 찾는 문제를 정의합니다.

### 최적화 목표 (Optimization Objective)
$$
\text{Find } \mathbf{x}^* \in \{0, 1\}^n \quad \text{s.t.} \quad A\mathbf{x}^* = \mathbf{b}
$$

### 구조적 정의 (Structural Definitions)
* **제약 조건 집합:** 본 시스템은 $m$개의 선형 방정식으로 구성되며, 각 행 $i$는 제약 조건 $\sum_{j=1}^{n} a_{ij}x_j = b_i$를 나타냅니다.
* **허용 가능 해 집합 (Feasible Set):** $\mathcal{S} = \{ \mathbf{x} \in \{0, 1\}^n \mid A\mathbf{x} = \mathbf{b} \}$
    * $\mathcal{S} \neq \emptyset$ 인 경우 시스템은 일관성(Consistent)이 있습니다. $|\mathcal{S}|$는 1일 수도, 여러 개일 수도 있습니다 — **$\mathbf{x}^*$의 유일성은 가정하지 않습니다**(2026-08 실측: 인스턴스 생성 시 유일성을 검증하지 않으면 $|\mathcal{S}|>1$인 경우가 흔함, 특히 소규모 인스턴스). 목표는 $\mathcal{S}$에 속하는 해 하나를 찾는 것이며, 심어놓은 특정 해와 일치할 필요는 없습니다.
    * $\mathcal{S} = \emptyset$ 인 경우, 만족하는 이진 해가 존재하지 않습니다. 먼지 등의 노이즈로 $\mathbf{b}$가 실제 스캔 결과와 어긋나면 이 경우가 발생합니다.

---

## 2. 대수적 완화 및 해석 (Algebraic Relaxation and Analysis)

이산 시스템의 복잡도를 이해하기 위해 실수 공간 $\mathbb{R}^n$에서의 완화(Relaxation) 시스템을 분석합니다.

### 2.1. 실수 완화 시스템 ($\mathbb{R}^n$)
$\tilde{\mathcal{S}} = \{ \mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{b} \}$ 라 정의합니다.

만약 $\text{rank}(A) = r < n$ (부족 제약 경우)이라면, 해의 집합은 다음과 같은 아핀 부분 공간(Affine subspace)을 형성합니다.
$$
\mathbf{x} = \mathbf{x}_p + \sum_{j=1}^{n-r} c_j \mathbf{v}_j
$$
여기서:
* $\mathbf{x}_p \in \mathbb{R}^n$은 $A\mathbf{x} = \mathbf{b}$를 만족하는 특수 해(Particular solution)입니다.
* $\{\mathbf{v}_1, \dots, \mathbf{v}_{n-r}\}$은 영공간(Null space) $\mathcal{N}(A)$의 기저(Basis)입니다.
* $c_j \in \mathbb{R}$은 스칼라 계수입니다.



### 2.2. 정수성 간극 (Integrality Gap)
연구의 핵심 과제는 $\tilde{\mathcal{S}} \cap \{0, 1\}^n$ 교집합 내에 존재하는 $\mathbf{x}^*$를 찾는 것입니다. 
실수 공간 $\tilde{\mathcal{S}}$에는 무수히 많은 해가 존재할 수 있지만, 부분 집합 $\mathcal{S}$는 유한하며 공집합일 가능성이 큽니다. 아핀 부분 공간과 $n$차원 하이퍼큐브(Hypercube) $\{0, 1\}^n$의 교차점을 찾는 것이 탐색 알고리즘의 주요 목표입니다.

---

## 3. 계산적 시사점 (Computational Implications)

* **일관성 검증:** 임의의 이진 행렬 $A$에 대해 $\mathcal{S} = \emptyset$ 여부를 판별하는 것은 NP-Complete 문제입니다.
* **탐색 전략:** 신경-기호(Neuro-symbolic) 프레임워크에서 영공간 기저 $\{\mathbf{v}_j\}$는 탐색을 위한 구조적 방향성을 제공합니다. MCTS 에이전트는 잔차(Residual) $\|A\mathbf{x} - \mathbf{b}\|_2^2$의 기울기를 활용하여 하이퍼큐브의 정점들을 허용 가능 해 집합 $\mathcal{S}$ 방향으로 탐색할 수 있습니다.
* **제약 만족:** 연속적인 해 공간 $\tilde{\mathcal{S}}$와 이산적인 격자 $\{0, 1\}^n$ 사이의 불일치는 '정수성 간극(Integrality Gap)'으로 정의되며, 이는 해당 BLS 인스턴스의 난이도를 결정합니다.

## 4. 프로젝트 구조 (File Tree)
neuroMCTS
├── .claude: agent 및 setting 용 json 파일
│   └── agents/
│       ├── researcher.md
│       ├── reviewer.md
│       └── verifier.md
├── baseline_dns: 제안 알고리즘과 비교하기 위한 baseline 코드
│     ├── gurobi_dns.py : gurobi를 이용한 MILP
│     ├── scip_dns.py : scip를 이용한 MILP
│     ├── LP_relaxation_gurobi_dns.py : gurobi를 이용한 LP relxation의 반올림 정수 해
│     ├── LP_relaxation_scip_dns.py : scip를 이용한 LP relaxation의 반올림 정수 해
│     ├── CPSAT_ORtools_dns.py : Google OR tools를 이용한 CPSAT 솔버 해
│     ├── LLL_with_fpylll_dns.py : fpylll 패키지를 이용한 LLL 알고리즘 해
│     ├── BKZ_with_fpylll_dns.py : fpylll 패키지를 이용한 BKZ 알고리즘 해
├── docs
│     ├── tex: 구현 사항, 실험 결과, 결과 insight 줄 글 정리 (영논문 형식)
│     ├── pptx: 학회 또는 연구 미팅 자료 생성
├── figures: 시각화의 결과 저장 파일
├── instances: 학습 및 테스트용 이진연립방정식 문제 인스턴스 파일 (.json)
│     ├── {train,test}_instances_dns_&mx&n_*: 초기 생성기(legacy), 유일해 강제 — 상대적으로 쉬움
│     ├── {train,test}_instances_dnsms_&mx&n_*: market-split 생성기, 유일성 미검증
│     └── {train,test}_instances_hard_&mx&n_*: CP-SAT 고난도 생성기(2026-08~), 동일 A에 대해 후보 해 K개를 심어 LP 완화 정점 간 거리(vertex_spread)가 가장 큰 것을 채택 — LP 완화 폴리토프를 직접 크게 만듦, 유일해 가정 폐기 (proposed_src/generate_hard_instances.py)
├── legacy: 사용되지는 않지만, 재사용성을 위해 저장해둔 legacy 코드 모음들
├── proposed_src: 제안 강화학습 알고리즘의 학습 및 추론 코드
├── ref: 참고 레퍼런스 조사
├── runs: 학습 시 성능 지표, 가중치, 시각화 저장 폴더
└── CLAUDE.md: 본 파일

## 5. 코딩 컨벤션 합의
- 구현언어: python
- 들여쓰기: 1 tab
- 세미콜론 불가
- 주석: class = 용도 3줄, 함수 = 1줄 
- 변수명: 최대한 간결한 단어로 (ex. double C, S, M)
- Try and Exception: 쓰레딩 동기화 보호 목적 제외 사용 불가

## 6. 검증 과정
- baseline vs 제안 알고리즘 비교 실험
- 우위 이유 분석 및 Insight 도출
- figures, docs에 한 눈에 분석되도록 업데이트

## 7. 주의 사항
- 베이스라인 알고리즘을 과소 평가하거나, 내 알고리즘을 올려치기 하지 말 것

## 8. 응답 규칙
- Claude는 매 응답 맨 앞에 응답 시각을 `YY-MM-DD HH:MM:SS` 형식으로 표기할 것 (예: `26-09-04 16:23:45`)
- 대답할때 물결표를 ("~") 잘못 붙이면 취소선이 섞여서 나오니 이를 유의해서 답변할 것. 

## 9. 버전 관리 규칙
- 버전 히스토리는 `docs/version.md`에 실시간으로 기록한다 (v13부터, 새 실험/변경마다 추가).
- **GitHub push는 각 버전 사이클이 끝났을 때 한 번에 모아서 진행한다** — 사이클 중간의 개별 커밋마다 push하지 말 것. 사이클 종료 시점은 사용자가 명시적으로 알려줌.

## 10. 실험 실행 규칙
* **다단계 실험은 연쇄 실행 스크립트로 묶을 것.** 앞 단계가 끝나면 다음 단계와 최종 평가까지
  사람 개입 없이 이어지도록 한 번에 걸어둔다. "끝나면 다음 단계를 돌리겠다"고 말만 하고
  실제로 걸지 않아 학습이 멈춘 채 며칠이 지난 사고가 있었다. 단계 사이를 수동으로 잇지 말 것.
* **보고를 요청받은 실험은 종료 신호를 받는 즉시 보고할 것.** 사용자가 다시 묻기를 기다리지
  않는다. 학습 종료·평가 완료·크래시 모두 해당한다.
* 실험을 시작할 때는 종료를 감지할 수단(Monitor 또는 종료 조건 대기)을 함께 걸어둔다.
  걸지 않은 채 "완료되면 보고하겠다"고 답하지 말 것.