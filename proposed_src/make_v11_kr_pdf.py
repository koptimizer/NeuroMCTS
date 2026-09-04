#!/usr/bin/env python3
"""Generates the Korean PDF of the v11 report via reportlab (xelatex/lualatex
Korean support is broken in this environment; reportlab + NotoSansKR is the
established workaround for this project)."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

pdfmetrics.registerFont(TTFont('NotoKR', '/home/kopt/.fonts/NotoSansKR.ttf'))

styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleKR', fontName='NotoKR', fontSize=16, leading=22, alignment=TA_CENTER, spaceAfter=6)
meta_style = ParagraphStyle('MetaKR', fontName='NotoKR', fontSize=10, leading=14, alignment=TA_CENTER, spaceAfter=14, textColor=colors.grey)
h1_style = ParagraphStyle('H1KR', fontName='NotoKR', fontSize=13, leading=18, spaceBefore=16, spaceAfter=8)
h2_style = ParagraphStyle('H2KR', fontName='NotoKR', fontSize=11.5, leading=16, spaceBefore=10, spaceAfter=6)
body_style = ParagraphStyle('BodyKR', fontName='NotoKR', fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=8)
caption_style = ParagraphStyle('CapKR', fontName='NotoKR', fontSize=9, leading=13, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=12)
abstract_style = ParagraphStyle('AbsKR', fontName='NotoKR', fontSize=9.5, leading=14.5, alignment=TA_JUSTIFY,
                                 leftIndent=20, rightIndent=20, spaceAfter=12)

def T(txt, style=body_style):
	return Paragraph(txt, style)

doc = SimpleDocTemplate('/home/kopt/neuroMCTS/docs/tex/LPneuroBLS_v11_report_KR.pdf',
                         pagesize=A4, topMargin=2.2*cm, bottomMargin=2.2*cm,
                         leftMargin=2.2*cm, rightMargin=2.2*cm)
story = []

story.append(T('v11 — 학습 기반 feasibility 판별이 왜 실패하는가, 그리고 CP-SAT를 대체가 아닌 강화로 얻은 첫 긍정 결과', title_style))
story.append(T('neuroMCTS project · 2026년 7월', meta_style))

story.append(T('초록', h1_style))
story.append(T(
	'이번 사이클은 프로젝트의 핵심 novelty를 다시 학습에 둔다: solution 구성 대신, '
	'feasibility 판별(0/1 해가 존재하는가) 자체를 직접 학습 목표로 삼는다. 조건은 두 가지 — '
	'(A) 고전 솔버보다 빠른 추론, (B) 학습하지 않은 크기에서도 유지되는 zero-shot 정확도. '
	'이번 사이클과 직전 사이클에 걸쳐 총 10가지 독립적인 분류기 변형을 테스트했다: 동일 A를 공유하는 '
	'짝 feasible/infeasible 데이터(NeuroSAT식), 리터럴 극성 이중노드 GNN(NeuroSAT 아키텍처) 3가지 깊이, '
	'GNN 표현력 한계에 대한 문헌의 구체적 처방(랜덤 노드 특징, 차원 0/8/32)을 적용한 이분그래프 GNN, '
	'3가지 pooling 방식(mean/max/mean+max), 그리고 손으로 뽑은 요약 통계 대신 원본 BKZ Gram-Schmidt '
	'프로파일에 대한 시퀀스 모델. 열 가지 중 여덟 가지가 우연 수준(AUC 0.4997~0.5003) 안에 들어오며, '
	'매번 positive-control sanity task로 최적화 버그가 아님을 확인했다. 예외 두 가지(7개 집계 프로브 '
	'통계의 선형 결합, AUC 0.588 및 0.599)는 단순 보고가 아니라 기제까지 진단했다: 분류기가 사용하는 '
	'표현 자체가 정보를 담지 못한다는 것을 직접 확인했다 — near-miss 짝의 96.8%가 8개의 서로 다른 BKZ '
	'축소 중 무엇을 쓰든 완전히 동일한(bit-identical) Gram-Schmidt 프로파일을 갖고, 크고 동질적인 제약 '
	'그래프의 mean/max pooling은 참 라벨과 무관하게 모집단 상수로 수렴한다(측정 표준편차 0.0146). '
	'문헌 검토(Chen et al. 2023의 증명 — 일부 feasible/infeasible MILP 쌍은 표준 GNN이 원리적으로 '
	'구분 불가; G4SATBench의 보고 — SAT 분류기가 크기 전이 시 15~25%p 정확도 붕괴)가 같은 패턴에 대한 '
	'독립적이고 기존에 존재하던 설명을 제공한다. 이번 사이클에 처음 60×150(제안 파이프라인 자체가 '
	'6.67%로 무너지는 크기)에서 실행한 baseline 재검증은, classical solver(Gurobi/SCIP/CP-SAT)가 '
	'40×100까지 모든 크기에서 속도·정확도 둘 다 제안 파이프라인을 압도하며(최대 약 200배 빠름), '
	'CP-SAT 자체도 30초 예산 내에 60×150 feasible 인스턴스 12개 중 2개만 풀어내지만 그래도 제안 '
	'파이프라인의 ~108초·6.67%보다는 낫다는 것을 보여준다. 이를 근거로 마지막 실험은 학습의 역할을 '
	'CP-SAT 대체에서 CP-SAT 강화로 전환한다: kernel pump의 최종 반올림점(수렴 여부 무관)을 '
	'AddHint로 CP-SAT에 제공하면 해를 더 찾지는 못하지만 동일 예산 내에서 CP-SAT가 혼자서는 끝내지 '
	'못한 infeasibility 증명을 완료한다(5/6→6/6) — 이번 사이클의 유일한 긍정적, 재현 가능한 결과다. '
	'추가 학습 튜닝의 전제 조건으로 테스트한 3가지 CP-SAT 파라미터 재설정(linearization_level=2, '
	'PORTFOLIO_SEARCH, LP_SEARCH)은 전부 기본값과 같거나 더 나빠, 유의미한 파라미터를 찾기 전까지 '
	'이 방향은 닫는다.', abstract_style))

story.append(PageBreak())

story.append(T('1. 동기: 1차 목표로서의 feasibility 판별', h1_style))
story.append(T(
	'이전 사이클들은 심볼릭 코어(전파, LP hard rule, kernel pump, AHL 격자 축소) 곁에 solution 구성용 '
	'학습(MCTS 정책, DFS 순서)을 결합했다. 이번 사이클의 범위는 더 좁고 엄격하다: solution 구성은 '
	'잠시 접어두고, GNN이나 RL이 feasibility(이진 결정, ∃x∈{0,1}ⁿ: Ax=b)를 직접 판별할 수 있는지를 '
	'두 가지 조건 하에 묻는다 — (A) classical 수리최적화 baseline보다 빠른 추론, (B) 학습하지 않은 '
	'인스턴스 크기로 zero-shot 전이되는 정확도.', body_style))

story.append(T('2. 열 가지 독립적 부정 결과', h1_style))
story.append(T(
	'이번 사이클과 직전 사이클에서 테스트한 모든 feasibility 분류기 변형을 아래 표 1에 정리했다(13행 '
	'중 3행은 직전 사이클 결과로 맥락을 위해 포함, 10행이 이번 사이클 신규). 원본 (A,b) 이분그래프 '
	'또는 이중노드 그래프에 대한 GNN 계열은 깊이(4층 vs 16층), 보조 LP 특징, pooling 연산자(mean/'
	'max/mean+max), 문헌이 제시한 대칭 파괴 처방(랜덤 노드 특징, 차원 8·32)과 무관하게 전부 우연 '
	'수준 안에 들어온다(0.4997~0.5003).', body_style))

tbl1_data = [
	['변형', 'AUC', '사이클'],
	['단발 스칼라 특징 (v5-v7)', '0.43-0.55', '이전'],
	['다회 프로브 집계 스칼라, 7특징 (v10)', '0.588', '이전'],
	['사전학습 GNN 임베딩 (v10)', '0.497', '이전'],
	['짝 데이터, 동일 7특징', '0.599', '이번'],
	['이중노드 GNN (NeuroSAT), 16층, 구조+b만', '0.5001', '이번'],
	['이중노드 GNN, 16층, +LP특징', '0.5003', '이번'],
	['이중노드 GNN, 4층 (얕음)', '0.5002', '이번'],
	['이분그래프 GNN, 랜덤특징 d=0 (control)', '0.5001', '이번'],
	['이분그래프 GNN, 랜덤특징 d=8', '0.5001', '이번'],
	['이분그래프 GNN, 랜덤특징 d=32', '0.5001', '이번'],
	['이분그래프 GNN, max-pooling', '0.5001', '이번'],
	['이분그래프 GNN, mean+max pooling', '0.5002', '이번'],
	['시퀀스 모델, 원본 BKZ Gram-Schmidt 프로파일', '0.5000', '이번'],
]
t1 = Table(tbl1_data, colWidths=[9.5*cm, 3*cm, 2.5*cm])
t1.setStyle(TableStyle([
	('FONTNAME', (0,0), (-1,-1), 'NotoKR'), ('FONTSIZE', (0,0), (-1,-1), 9),
	('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
	('ALIGN', (1,0), (-1,-1), 'CENTER'), ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t1)
story.append(T('표 1: 테스트한 모든 feasibility 분류기 변형의 held-out AUC.', caption_style))

story.append(T(
	'최적화 실패 배제: 표 1의 모든 부정 결과는 신뢰하기 전에 positive-control sanity task로 '
	'검증했다 — 동일한 아키텍처·학습 루프에 참 라벨 대신 같은 입력에서 계산 가능한 쉬운 대리 라벨을 '
	'넣은 것이다. 이중노드 GNN은 연속값 임계 대리과제(mean(b)>median)에서 25 epoch만에 학습 손실 '
	'0.693→0.607, 이분그래프 GNN은 pooling에도 보존되는 상수 스칼라 신호로 6 epoch만에 0.614→0.04에 '
	'도달했다(둘 다 실행 시점에 관측, 별도 로그 미보존). 둘 다 어떤 이용 가능한 신호가 있을 때 '
	'그래디언트가 흐르고 최적화기가 수렴함을 확인시켜준다 — 참 feasibility 라벨에서의 실패는 '
	'학습 절차의 문제가 아니다.', body_style))

story.append(T('2.1 짝 내부 신호: 강력하지만 배포 불가능', h2_style))
story.append(T(
	'표 1에서 유일하게 우연을 넘는 모집단 AUC(0.599)를 보인 데이터셋은 동일 A를 공유하고 b만 '
	'1~3행 ±1 섭동한 2,529개 짝 인스턴스로 구성된다(CP-SAT 인증). 같은 7개 집계 특징의 모집단 AUC는 '
	'직전 사이클의 0.588을 사실상 재현할 뿐이지만(독립 데이터에서의 진짜 재현), 전체 모집단이 아니라 '
	'짝 내부 차이만 비교하면 훨씬 강한 효과가 드러난다: ahl_min_norm_min 특징은 완전한 2,487개 짝 '
	'중 100%에서 방향이 일치한다(Wilcoxon 부호순위 p=9.42×10⁻¹⁴). 이는 실재하는 압도적 통계 신호지만, '
	'짝의 두 구성원(즉 쌍둥이 인스턴스의 참 라벨)을 이미 알아야만 접근 가능해 배포 시점에는 쓸 수 '
	'없다. A만으로 계산되는 내재적 기준선으로 정규화해 인스턴스별로 쓸 수 있게 만들려는 시도는 '
	'모집단 AUC를 ~0.51에 그치게 해(우연 수준) 실패했다 — 흥미롭지만 배포 경로가 없는 발견으로만 '
	'기록한다.', body_style))

story.append(T('2.2 문헌의 유일한 처방, 테스트와 반증', h2_style))
story.append(T(
	'Chen 외(2023, ICLR)는 표준 GNN이 Weisfeiler-Leman 대칭 한계로 인해 특정 feasible/infeasible '
	'MILP 쌍을 구분할 수 없음을 증명하고, 모든 노드에 학습·추론 시 매번 새로 뽑는 i.i.d. 랜덤 특징을 '
	'주입하면(추론 시 16회 draw 평균) 자신들의 실험에서 구분 가능성이 회복됨을 보고한다. 이는 이번 '
	'프로젝트가 사이클 시작 시점까지 시도하지 않은, 문헌으로 검증된 유일한 구체적 개입이다. '
	'이분그래프 GNN에 구현해(bipartite_randfeat_classifier.py) 5,058개 짝 데이터셋으로 학습한 결과, '
	'랜덤 특징 차원 0(control)·8·32 전부 AUC 0.5001로 수렴 — 어떤 차원에서도 측정 가능한 효과가 '
	'없었다.', body_style))
story.append(T(
	'스스로 세운 두 번째 가설도 반증됨: control 실행의 이상하게 낮은 학습 손실을 진단하던 중, '
	'~50~70개 노드의 동질적 제약 그래프를 mean-pooling하면 pooling된 표현이 모집단 상수로 수렴한다는 '
	'것을 발견했다(1,200개 인스턴스에 걸친 pooling된 LP완화값의 표준편차 0.0146, 그룹 분리에 '
	'필요한 ~0.1 범위 대비 매우 작음) — WL-대칭 논증과는 다른 종류의 집중 부등식 희석이다. '
	'max-pooling과 mean+max pooling(이상치 신호를 평균으로 지워버리지 않는 방식)으로 재실행해 직접 '
	'검증했으나 둘 다 정확히 우연 수준으로 수렴했다(0.5001, 0.5002). 즉 희석 가설 자체는 참이지만(표준편차 '
	'측정치는 유효) 이 과제의 실제 병목은 아니다 — pooling 이전의 노드 임베딩 자체에 애초에 분리 '
	'가능한 신호가 없어 보이며, 이는 WL-비구분성 설명과 정합적이다.', body_style))

story.append(T('2.3 근본 원인 진단: near-miss 짝은 표현상 동일하다', h2_style))
story.append(T(
	'GNN 대신 원본 (A,b) 그래프가 아닌 이미 대칭이 깨진 수치 표현(BKZ 축소 격자의 원본 Gram-Schmidt '
	'로그노름 프로파일)에 시퀀스 모델을 학습시키면 신호를 회복할 수 있다는 추가 가설을 상위 20개 '
	'프로파일 항목에 대한 양방향 LSTM으로 테스트했다(gs_profile_classifier.py). positive-control '
	'대리 라벨(profile[0]이 중앙값보다 큼)은 AUC 1.0으로 학습됐고(실행 시점 관측), 참 feasibility '
	'라벨은 정확히 AUC 0.5000으로 수렴했다.', body_style))
story.append(T(
	'이를 진단한 결과(실행 시점에 직접 짝별 비교 스크립트로 계산, 별도 로그 미보존) 직접적 원인을 '
	'찾았다: 2,529개 near-miss 짝의 96.8%에서, 8회 시도 중 최선의 BKZ Gram-Schmidt 프로파일이 짝의 '
	'두 구성원 사이에 완전히 동일하다(float32에서 L2 차이 정확히 0.0). 이는 best-try 선택 방식의 '
	'아티팩트가 아님을 확인했다: 순열 없는 단일 고정 시도(tries=1)로도, 8회 각각의 개별 시도를 '
	'짝별로 비교해도 동일한 결과였다 — 테스트한 모든 시도에서 프로파일이 동일했다. 메커니즘상 b는 '
	'n+1개 기저 행 중 정확히 1개에만 영향을 주고, 나머지 n개 행은 전적으로(짝 구성원 간 공유되는) '
	'A로 결정된다 — 상위 20개 Gram-Schmidt 방향에 한해서는 축소가 명백히 A 유래 구조에 지배된다. '
	'아무리 표현력이 뛰어난 아키텍처도 비트 단위로 동일한 입력에서 두 라벨을 분리할 수 없다 — 이는 '
	'학습이나 용량의 한계가 아니다. 이는 또한 직전 사이클의 약한 0.588 집계-스칼라 신호가 이 '
	'프로파일 형태에서 나올 수 없음을 명확히 한다 — 다른 양(예: AHL이 예산 내에 풀리는지 여부)을 '
	'반영하는 것이 틀림없으며, 향후 연구는 이 둘을 혼동하지 말아야 한다.', body_style))

story.append(T('3. 문헌적 근거', h1_style))
story.append(T(
	'구조화된 문헌 검토(ref/gnn-rl-feasibility-classification.md, researcher가 수집하고 reviewer가 '
	'교정)는 위 패턴에 대한 독립적이고 기존에 존재하던 맥락을 제공한다. 세 가지 발견이 핵심이다. '
	'첫째, Chen 외의 WL-표현력 정리는 경험적 우연이 아니라 표현 수준의 불가능성 결과이며, 그 구성적 '
	'대응 결과(랜덤 특징이 자신들의 실험에서 구분 가능성을 회복시킴)는 이번 사이클이 테스트했지만 '
	'이 문제의 특정 구조(그들의 실험보다 더 조밀하고 대칭적인 0/1 market-split류 제약 행렬)에서는 '
	'확인되지 않은 유일한 개입이다. 둘째, G4SATBench(Li 외 2024, TMLR)는 가장 체계적인 GNN 기반 SAT '
	'분류기 벤치마크로, 난이도에 따라 학습분포 내 78~96% 정확도를 보고하며 4~5배 크기 전이 시 15~25%p '
	'추가로 붕괴한다고 명시한다 — GNN 분류기가 그나마 가장 잘 작동하는 유일한 문제군(CNF/SAT)에서도 '
	'조건 (B)와 정면으로 배치된다. 셋째, 출력이 곧 feasibility 판정(완전한 솔버 내부의 분기·탐색 '
	'정책이 아니라)인 강화학습 방법을 표적 검색했으나 찾지 못했다 — 조사한 모든 RL-for-SAT/MILP '
	'결과(Graph-Q-SAT, NeuroCore, NLocalSAT)는 정확한 솔버의 내부 탐색을 안내할 뿐 그 판정을 대체하지 '
	'않으며, 어떤 크기에서도 GNN feasibility 분류 추론이 classical solver를 wall-clock에서 이긴다고 '
	'보고한 1차 문헌을 찾지 못했다. 즉 조건 (A)는 문헌에서 확인도 반증도 되지 않았고, 조건 (B)는 '
	'가장 가까운 대리 과제에서 직접 반증됐다.', body_style))

story.append(T('4. Baseline 재검증', h1_style))
story.append(T(
	'이번 사이클을 촉발한 두 조건을 프로젝트 자체의 baseline 솔버 — 이번에 처음으로 60×150에서도 '
	'실행 — 에 대해 검증했다(표 2, 그림 2).', body_style))

tbl2_data = [
	['크기', 'Gurobi 정확도', 'Gurobi 시간(s)', 'SCIP 정확도', 'SCIP 시간(s)', 'CP-SAT 정확도', 'CP-SAT 시간(s)'],
	['10×25', '100.0%', '0.0023', '100.0%', '0.0053', '100.0%', '0.0011'],
	['20×50', '100.0%', '0.0276', '100.0%', '0.141', '100.0%', '0.0307'],
	['40×100', '99.4%', '1.875', '93.2%', '9.24', '90.9%', '8.03'],
	['60×150', '—', '—', '—', '—', '16.67%', '17.72'],
]
t2 = Table(tbl2_data, colWidths=[1.8*cm]+[2.1*cm]*6)
t2.setStyle(TableStyle([
	('FONTNAME', (0,0), (-1,-1), 'NotoKR'), ('FONTSIZE', (0,0), (-1,-1), 8),
	('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
	('ALIGN', (1,0), (-1,-1), 'CENTER'), ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t2)

tbl2b_data = [
	['크기 (제안 파이프라인)', '정확도', '시간(s)'],
	['10×25', '100.0%', '0.23'],
	['20×50', '78.25%', '3.78'],
	['40×100', '66.67%', '22.0'],
	['60×150', '6.67%', '107.95'],
]
t2b = Table(tbl2b_data, colWidths=[5*cm, 3*cm, 3*cm])
t2b.setStyle(TableStyle([
	('FONTNAME', (0,0), (-1,-1), 'NotoKR'), ('FONTSIZE', (0,0), (-1,-1), 9),
	('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
	('ALIGN', (1,0), (-1,-1), 'CENTER'), ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(Spacer(1, 6))
story.append(t2b)
story.append(T('표 2: 크기·방법별 solution 정확도와 평균 wall-clock 시간.', caption_style))

story.append(Image('/home/kopt/neuroMCTS/figures/v11_baseline_reality_check.png', width=16*cm, height=16*cm*5/12))
story.append(T('그림 2: (왼쪽) 크기별 solve rate. (오른쪽) 크기별 wall-clock 시간(log 스케일).', caption_style))

story.append(T(
	'직접 비교되는 모든 크기(10×25~40×100)에서 classical MILP/CP-SAT baseline이 제안 파이프라인의 '
	'정확도와 같거나 앞서고, 약 2~200배 빠르다(10×25에서 가장 크고 40×100에서 가장 작음). 이는 '
	'조건 (A)에 대한 기존 가정을 정면으로 반박하며, market-split 인스턴스가 m≈7 부근에서 LP기반 '
	'branch-and-cut을 무력화한다는 문헌적 동기(Wassermann 2025)와도 안 맞는다 — 프로젝트 자체 생성기가 '
	'만든 인스턴스는 m=40까지 Gurobi/CP-SAT가 90% 이상 정확도로 풀어낸다. 이 불일치는 해소하지 않고 '
	'있는 그대로 표기한다.', body_style))
story.append(T(
	'진짜 미해결 질문은 60×150이다 — 제안 파이프라인 자체가 무너지는 지점(6.67%, 30개 feasible 중 2개, '
	'직전 사이클 평가)이자 baseline을 한 번도 테스트한 적 없는 크기다. 이번 사이클에 별도의 더 작은 '
	'18개 인스턴스 표본(feasible 12, infeasible 6)에서 30초 예산으로 실행한 plain CP-SAT는 feasible '
	'12개 중 2개만 풀고(16.67%), infeasible 6개 중 5개는 빠르게(각 3초 이내) 증명하지만 6번째는 '
	'타임아웃으로 잘못된 기본값을 낸다. 두 수치는 서로 다른 평가셋에서 나온 것이라 정합 비교는 '
	'아니지만, 어느 쪽으로 읽어도 결론은 불리하다: CP-SAT가 여기서 절대적으로 압도적이지는 않지만, '
	'그럼에도 30초의 16.67%가 여전히 제안 파이프라인의 ~108초·6.67%보다 낫다 — 솔직한 결론은 이 '
	'크기에서도 classical solver가 여전히 우위를 유지할 가능성이 높다는 것이지, 크기 구간이 '
	'프로젝트에 유리하게 뒤집혔다는 것이 아니다.', body_style))

story.append(T('5. CP-SAT를 대체가 아닌 강화로', h1_style))
story.append(T(
	'4절의 결과를 근거로, 이번 사이클의 마지막 실험은 학습의 역할을 재조정한다: CP-SAT의 판정을 '
	'대체하는 대신, 프로젝트가 이미 가진 심볼릭 부산물로 CP-SAT 자체를 강화한다. 동일한 60×150, '
	'18개 인스턴스, 30초 예산에서 실행했다(표 3).', body_style))

tbl3_data = [
	['변형', 'Solution 정확도', 'Infeasible 증명', '평균 시간'],
	['Plain CP-SAT (기본 파라미터)', '16.67% (2/12)', '5/6', '17.72s'],
	['CP-SAT + kernel-pump hint (AddHint)', '16.67% (2/12)', '6/6', '17.70s'],
	['CP-SAT, linearization_level=2', '8.33% (1/12)', '5/6', '21.22s'],
	['CP-SAT, PORTFOLIO_SEARCH', '16.67% (2/12)', '5/6', '19.01s'],
	['CP-SAT, LP_SEARCH', '16.67% (2/12)', '5/6', '18.00s'],
]
t3 = Table(tbl3_data, colWidths=[7*cm, 3.3*cm, 2.8*cm, 2.4*cm])
t3.setStyle(TableStyle([
	('FONTNAME', (0,0), (-1,-1), 'NotoKR'), ('FONTSIZE', (0,0), (-1,-1), 9),
	('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
	('BACKGROUND', (0,2), (-1,2), colors.lightgrey),
	('ALIGN', (1,0), (-1,-1), 'CENTER'), ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t3)
story.append(T('표 3: 60×150(18개 인스턴스, 30초 cap)에서의 CP-SAT 변형들.', caption_style))

story.append(T(
	'kernel pump의 최종 반올림점(pump 자체의 수렴 여부와 무관하게 제공)으로 CP-SAT를 warm-start하면 '
	'feasible 인스턴스의 solve rate는 늘리지 못하지만(양쪽 조건 모두 2/12, 동일한 두 인스턴스, 비슷한 '
	'시간) 그렇지 않으면 잘못된 기본값으로 타임아웃됐을 infeasibility 증명 하나를 완료시킨다 '
	'(5/6→6/6; 나머지 5개 infeasible 인스턴스는 hint 유무와 무관하게 3초 이내로 빠르게 증명된다). '
	'18개 표본에서의 작은 효과라 과장하면 안 되지만, 이번 사이클에서 학습이든 아니든 어떤 개입에서 '
	'얻은 유일한 재현 가능한 긍정 결과이며, CP-SAT 자체의 정확성 보장을 전혀 건드리지 않고 얻어진다 '
	'— hint는 탐색이 시작하는 지점만 바꾼다.', body_style))
story.append(T(
	'학습된 파라미터 선택기를 만들 가치가 있는지 확인하기 위한 전제 조건으로 테스트한 3가지 CP-SAT '
	'파라미터 재설정은 전부 기본값과 같거나 더 나빴다: linearization_level=2는 오히려 1개 인스턴스를 '
	'덜 풀고, PORTFOLIO_SEARCH·LP_SEARCH는 동일한 두 인스턴스를 비슷하거나 약간 더 느린 시간에 '
	'푼다. 학습된 선택기가 실제로 기본값을 능가하려면 일부 인스턴스에서 우위를 보이는 설정이 '
	'최소한 하나는 있어야 하는데, 여기서 테스트한 셋 중 어느 것도 그렇지 않았다 — 유의미한 분산을 '
	'보이는 파라미터를 찾을 때까지 이 방향은 닫는다.', body_style))

story.append(T('6. 논의: 왜 학습이 여기서 전이되지 않는가', h1_style))
story.append(T(
	'경험적 패턴(표 1, 깊이·폭·pooling·문헌의 처방과 무관하게 정확히 우연 수준)과 문헌적 근거(3절)는 '
	'"문제가 그저 어렵고 학습이 부족했다"와는 다른 논리적 설명을 뒷받침한다. 개별 실험 수치와 무관한 '
	'세 가지 논거가 있다.', body_style))
story.append(T(
	'(i) 느린 수렴이 아니라 표현 수준의 불가능성. 2.3절은 near-miss 짝의 96.8%가 비트 단위로 동일한 '
	'모델 입력을 만든다는 것을 직접 보인다 — 동일한 입력에 대한 결정적 함수가 무엇을 출력할 수 있는지는 '
	'아무리 학습해도 바뀌지 않는다. Chen 외의 정리는 이를 더 넓은 MILP 표현 계열로 일반화한다. '
	'학습 부족이 원인이라면 깊이·폭·랜덤특징 차원·데이터량 중 어느 것을 키워도 AUC가 우연에서 서서히 '
	'멀어져야 하는데, 표 1의 모든 설정이 정확히 0.5에서 0.0003 이내에 머문다 — 느린 학습 곡선이 아니라 '
	'단단한 상한의 특징이다.', body_style))
story.append(T(
	'(ii) 정확한 정수 feasibility는 불연속적인, 정수론적 성질이다. 그래디언트 기반 학습은 입력의 '
	'매끄러운 함수를 향해 편향돼 있다. Ax=b가 정수해를 갖는지는 gcd·격자·모듈러 구조로 결정되며, '
	'한 항목을 ±1만 바꿔도 그 사이에 아무런 매끄러운 중간 신호 없이 완전히 뒤집힐 수 있다. 이는 이번 '
	'프로젝트에서 앞서 겪었던 것과 같은 장애물이다 — 연속값 신경망은 sanity 대리과제로서도 단순 '
	'비트 parity조차 학습하지 못했다.', body_style))
story.append(T(
	'(iii) 이 문제군은 매끄러운 신호를 제거하도록 구성돼 있으며, 이는 그래디언트 하강에도 같은 이유로 '
	'영향을 준다. Market-split류 인스턴스는 연속 완화를 따라가는 방법을 무력화하도록 조합최적화 '
	'문헌에서 의도적으로 구성된다(4절의 LP 반올림·branch-and-bound 어려움이 그 고전적 발현이다). '
	'그래디언트 하강도 메커니즘적으로는 연속 신호를 따라가는 또 다른 방법이다 — 어떤 연속적 방법을 '
	'무력화하도록 설계된 문제군이 다른 연속적 방법만 온전히 남겨둘 특별한 이유는 없다.', body_style))

story.append(T('7. 결론', h1_style))
story.append(T(
	'엄격한 속도·zero-shot 크기강건성 조건 하에 프로젝트의 1차 목표로 테스트된 feasibility 판별은 '
	'프로젝트 자체의 실험(단순 null이 아니라 기제까지 진단된 열 가지 독립 부정 결과)이나 문헌(불가능성 '
	'정리와 가장 가까운 실증 벤치마크 자체가 보고하는 크기 전이 붕괴) 어느 쪽의 검증도 통과하지 못했다. '
	'이번 사이클에서 학습이 낸 유일한 측정된 긍정적 기여는 완전히 다른 역할에서 나왔다 — 심볼릭 '
	'부산물(pump의 근사해)을 이미 정확한 솔버에 warm-start hint로 제공해, CP-SAT가 예산 내에 혼자 '
	'끝내지 못한 증명을 완료시킨 것이다. Baseline 재검증은 나아가 classical solver가 단순한 '
	'대조군이 아니라 40×100까지 모든 테스트 크기에서 지배적인 방법이며, 제안 파이프라인과 CP-SAT '
	'자체 둘 다 힘겨워하는 60×150에서도 여전히 앞서 있을 가능성이 높다는 것을 확립한다. 권고하는 '
	'다음 단계: CP-SAT의 판정을 대체하려는 추가 시도 이전에, hint 방향(더 큰 표본, AHL 기반 hint, '
	'부분 hint)을 먼저 확장할 것.', body_style))

doc.build(story)
print('wrote LPneuroBLS_v11_report_KR.pdf')
