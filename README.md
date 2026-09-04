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
    * $\mathcal{S} \neq \emptyset$ 인 경우 시스템은 일관성(Consistent)이 있습니다.
    * $\mathcal{S} = \emptyset$ 인 경우, 만족하는 이진 해가 존재하지 않습니다.
    * 이러한 가정은 이미지 복원이 단일의 물체를 스캔하기 때문에 단일해가 존재하는 것이며, 먼지 등의 노이즈가 발생하면 이미지 복원이 되지 않기 때문에 해가 존재하지 않는 경우가 발생하는 것이다.

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