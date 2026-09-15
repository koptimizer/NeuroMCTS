#!/usr/bin/env python3
"""Builds the Korean edition of the BLS branching-guidance paper.

reportlab + NotoSansKR rather than LaTeX: xelatex/lualatex Korean support is broken in
this environment. Technical terms stay in English so the two editions and the cited
literature line up term for term.

Font coverage constrains notation: NotoSansKR has no glyph for U+1D4AE (script capital S),
U+2124 (double-struck Z), or combining accents, and reportlab drops missing glyphs
silently -- a broken formula looks like a typo rather than an error. The solution set is
therefore written S, its relaxation S_LP, and estimates p^. Check coverage with fontTools
before introducing a new symbol.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 PageBreak, KeepTogether, Preformatted)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

pdfmetrics.registerFont(TTFont('NotoKR', '/home/kopt/.fonts/NotoSansKR.ttf'))
OUT = '/home/kopt/neuroMCTS/docs/tex/LPneuroBLS_paper_kr.pdf'

title_s = ParagraphStyle('T', fontName='NotoKR', fontSize=15, leading=21, alignment=TA_CENTER, spaceAfter=4)
meta_s  = ParagraphStyle('M', fontName='NotoKR', fontSize=9.5, leading=13, alignment=TA_CENTER, spaceAfter=14, textColor=colors.grey)
h1_s    = ParagraphStyle('H1', fontName='NotoKR', fontSize=12.8, leading=18, spaceBefore=15, spaceAfter=7)
h2_s    = ParagraphStyle('H2', fontName='NotoKR', fontSize=11.2, leading=16, spaceBefore=11, spaceAfter=5)
body_s  = ParagraphStyle('B', fontName='NotoKR', fontSize=9.8, leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
abs_s   = ParagraphStyle('A', fontName='NotoKR', fontSize=9.2, leading=14, alignment=TA_JUSTIFY,
                          leftIndent=16, rightIndent=16, spaceAfter=9)
cap_s   = ParagraphStyle('C', fontName='NotoKR', fontSize=8.6, leading=12, alignment=TA_CENTER,
                          textColor=colors.grey, spaceAfter=11)
eq_s    = ParagraphStyle('E', fontName='NotoKR', fontSize=10, leading=16, alignment=TA_CENTER,
                          spaceBefore=4, spaceAfter=8)
prop_s  = ParagraphStyle('P', fontName='NotoKR', fontSize=9.6, leading=14.5, alignment=TA_JUSTIFY,
                          leftIndent=18, rightIndent=18, spaceBefore=4, spaceAfter=9,
                          borderPadding=5, backColor=colors.Color(0.955, 0.955, 0.965))
code_s  = ParagraphStyle('K', fontName='Courier', fontSize=7.6, leading=10.2, spaceAfter=8,
                          leftIndent=10, backColor=colors.Color(0.96, 0.96, 0.96))
ref_s   = ParagraphStyle('R', fontName='NotoKR', fontSize=8.7, leading=12.4, alignment=TA_JUSTIFY,
                          leftIndent=14, firstLineIndent=-14, spaceAfter=4)

E = []
def P(t, s=body_s): E.append(Paragraph(t, s))
def H1(t): E.append(Paragraph(t, h1_s))
def H2(t): E.append(Paragraph(t, h2_s))
def EQ(t): E.append(Paragraph(t, eq_s))


def TBL(data, widths, caption=None, right_from=1, font=8.5):
	rows = [[Paragraph(c, ParagraphStyle('c', fontName='NotoKR', fontSize=font, leading=font + 3.2))
	          for c in r] for r in data]
	t = Table(rows, colWidths=widths, repeatRows=1)
	t.setStyle(TableStyle([
		('FONTNAME', (0, 0), (-1, -1), 'NotoKR'), ('FONTSIZE', (0, 0), (-1, -1), font),
		('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.90, 0.90, 0.92)),
		('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black),
		('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black),
		('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
		('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
		('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
		('ALIGN', (right_from, 1), (-1, -1), 'RIGHT')]))
	E.append(KeepTogether([t] + ([Paragraph(caption, cap_s)] if caption else [Spacer(1, 10)])))


# ═══════════════════════ 표지 · 초록 ═══════════════════════
P('Amortized Deduction for Binary Linear Systems:<br/>'
  'Learning Branching Guidance from Exact Conditional Marginals', title_s)
P('Binary linear system을 위한 amortized deduction:<br/>'
  '정확한 conditional marginal로부터 branching guidance를 학습하기<br/>'
  'neuroMCTS project · 2026년 9월 · 한국어판', meta_s)

H1('초록')
P('Binary linear system(BLS)은 <i>A</i>∈{0,1}<sup><i>m</i>×<i>n</i></sup>에 대해 '
  '<i>A</i><b>x</b>=<b>b</b>를 만족하는 <b>x</b>∈{0,1}<sup><i>n</i></sup>를 찾는 문제이며, '
  '레이저 기반 비파괴검사에서 정수 투영값으로부터 이진 점유 영상을 복원하는 과제로 나타난다. '
  'market-split 계열 인스턴스에서는 linear relaxation이 의도적으로 무정보하도록 설계되므로, '
  'backtracking search는 값싼 heuristic으로는 잘 내릴 수 없는 <b>branching 결정</b>에 대부분의 '
  '노력을 쓴다. 본 논문은 이 설정을 위한 학습 기반 branching heuristic과, 그것을 지도학습하기 위한 '
  '정확한 절차를 제시한다.', abs_s)
P('방법은 두 관찰에 기반한다. <b>첫째</b>, BLS를 partial assignment로 conditioning하는 것은 그 배정을 '
  '대입해 <b>더 작은 BLS를 얻는 것과 동일하다.</b> 따라서 하나의 network가 구조 변경 없이 root와 모든 '
  'interior node에서 예측하며, 학습 상태는 크기가 줄어든 평범한 instance가 된다. '
  '<b>둘째</b>, 전수열거가 가능한 크기의 instance에서는 정확한 conditional marginal '
  '<i>p<sub>j</sub></i>=Pr[<i>x<sub>j</sub></i>=1 | <i>A</i>,<b>b</b>]를 계산할 수 있고, 이것이 node의 '
  '자유 변수를 두 부류로 <b>분할</b>한다: 살아남은 모든 해가 일치해 반대로 분기하면 subtree가 비어 '
  'backtrack이 확정되는 변수(<b>DET</b>)와, 해들이 갈려 어느 쪽으로 분기해도 해에 도달하는 '
  '변수(<b>OPEN</b>)다. <b>비용을 발생시키는 것은 전자뿐이다.</b> 이 분할은 정확한 학습 target과 '
  '동시에 <b>어떤 예측이 중요한지</b>에 대한 기준을 제공한다: root에서는 변수의 6.3%만이 DET이지만 '
  'depth 8에서는 그 비중이 89.0%가 되며, 정확도가 탐색 비용으로 환산되기 시작하는 지점이 바로 여기다.', abs_s)
P('우리는 21.8K 파라미터의 bipartite graph network를 이 target에 학습시키고, 오직 branching에만 사용한다 — '
  '증명이 아니라 확률을 내놓으므로 변수 확정에는 쓰지 않는다. 같은 변수를 탐지하는 건전 절차인 LP-probing과 '
  '비교하면, probing은 변수마다 linear program을 하나씩 푸는 반면 network는 relaxation 한 번과 forward pass '
  '한 번이면 된다. 따라서 비용 비율이 instance 크기에 비례해 커진다: 양쪽 모두 warm start한 backend에서 측정해 '
  '<b>n=25에서 1.7배, n=60에서 7.2배, n=150에서 17.8배</b>다. wall-clock 예산을 맞추고 세 seed로 반복한 비교에서, '
  '21×60 기준 LP 기반 branching 대비 median <b>3.29배</b> 빠르고(seed별 sign test p=0.019), node를 6.2배 적게 '
  '전개하며, 모든 반복에서 30/30 대 25/30을 해결한다. lattice reduction이 먼저 도는 전체 pipeline에서는 '
  'guidance가 세 seed 모두에서 잔여 instance를 전부 닫아(50/50 대 45/50) end-to-end 해결율을 '
  '94.4±5.1%에서 <b>100±0.0%</b>로 올린다. solution 개수를 크기 간에 통제하면 transfer 손실이 측정되지 않는다: '
  '10×25에서만 학습한 network가 평가 크기에서 학습한 network와 통계적으로 구별되지 않는다.', abs_s)
P('<b>달성하지 못한 것도 함께 보고한다.</b> CP-SAT는 동일 instance를 모든 크기에서 <b>41~62배 빠르게</b> 풀며, '
  '우리 시스템은 그것과 경쟁하지 못한다. 그 격차를 분해하면 우리 branching이 node를 <b>1.6배 적게</b> 쓰고 '
  'node 처리는 <b>78배 느리다</b> — 결함이 branching 품질이 아니라 node당 비용에 있다는 뜻이나, 그것이 회복 가능함을 보인 것은 아니다. 그리고 probing 대비 '
  '비용 우위는 점근적으로는 실재하나, 우리 instance에서 예측이 가장 중요한 depth에서는 <b>1.3~1.7배</b>에 '
  '그치며 그 구간에서 probing은 예측이 아니라 <b>증명</b>을 제공한다. 따라서 우리가 주장하는 기여는 '
  '<b>방법론적</b>이다: 정확한 conditional posterior는 branching에 대해 얻을 수 있고 잘 정의된 지도 신호이며, '
  'in-tree 상태가 root 상태보다 나은 학습 분포이며 — root로 학습한 network는 tree 안에 배치하면 relaxation rounding보다 못하고, '
  'in-tree로 학습한 network는 3.4%p 앞선다 — 그렇게 얻은 heuristic은 instance 크기를 넘어 전이된다. '
  '코드, 생성기, 정확한 marginal label, 인스턴스 단위 결과를 공개한다.', abs_s)
E.append(PageBreak())

# ═══════════════════════ 1. 서론 ═══════════════════════
H1('1. 서론')
H2('1.1 문제')
P('<i>A</i>∈{0,1}<sup><i>m</i>×<i>n</i></sup>, <b>b</b>∈<b>Z</b><sup><i>m</i></sup>이 주어질 때 '
  '<b>binary linear system</b>(BLS)은 다음 feasibility 문제다:', body_s)
EQ('<i>A</i><b>x</b> = <b>b</b> 를 만족하는 <b>x</b> ∈ {0,1}<sup><i>n</i></sup> 를 찾아라.  &nbsp;&nbsp;(1)')
P('그 해집합을 <i>S</i> = {<b>x</b>∈{0,1}<sup><i>n</i></sup> : <i>A</i><b>x</b>=<b>b</b>}라 쓴다. '
  '각 row는 cardinality constraint로, 지정된 셀 부분집합 중 몇 개가 점유되어 있는지를 나타낸다.', body_s)
P('응용 배경은 레이저 기반 비파괴검사다. 시편을 <i>m</i>개 방향으로 스캔하면 각 스캔은 경로상 흡수 셀의 '
  '정수 개수를 반환하고, 과제는 이진 점유 영상을 복원하는 것이다. 이 설정의 두 성질이 문제 정식화를 규정한다. '
  '해는 <b>유일할 필요가 없다</b> — 동일한 측정치와 부합하는 영상이 여럿일 수 있고, <i>S</i>의 '
  '<b>아무 원소나</b> 복원하면 물리적 문제는 해결된다. 그리고 측정 노이즈가 <b>b</b>를 교란해 '
  '<i>S</i>=∅이 될 수 있으므로, 신뢰할 수 있는 infeasibility 판정도 해를 찾는 것만큼 가치가 있다.', body_s)
P('<i>S</i>≠∅ 판정은 일반적으로 NP-complete이다. 본 연구의 인스턴스는 market-split 문제를 본뜬 '
  '고난도 영역에서 의도적으로 추출되며, 그 영역에서는 linear relaxation '
  '<i>S</i><sub>LP</sub> = {<b>x</b>∈[0,1]<sup><i>n</i></sup> : <i>A</i><b>x</b>=<b>b</b>}가 '
  '대부분 fractional한 vertex를 갖는 큰 polytope이다. 따라서 relaxation 기반 bounding이 약하고, '
  'complete solver는 relaxation의 도움을 거의 받지 못한 채 어디서 분기할지 결정해야 한다 — '
  '이것이 본 논문이 학습하는 결정이다.', body_s)

H2('1.2 학습 컴포넌트가 붙는 자리, 그리고 무엇으로 학습할 것인가')
P('식 (1)을 위한 complete solver는 <b>deduction</b>(constraint propagation, relaxation 기반 가지치기)과 '
  '<b>branching 결정</b>(어느 자유 변수로 분기하고 어느 값을 먼저 시도할 것인가)을 번갈아 수행한다. '
  'deduction은 건전해야 하고 앞으로도 그래야 하지만, branching은 heuristic한 선택이라 '
  '<b>오류의 대가가 탐색 시간일 뿐 정확성이 아니다.</b> 이 비대칭이 branching을 학습 컴포넌트가 붙기에 '
  '자연스러운 자리로 만들며, mixed-integer programming에서 학습이 가장 성공한 지점이기도 하다.', body_s)
P('어려운 것은 <b>지도(supervision)</b>다. 표준적인 답이 두 가지 있는데 여기서는 둘 다 만족스럽지 않다. '
  '심어진 해 <b>x</b>*로 학습하는 것은 <i>S</i>의 한 원소를 ground truth로 취급하는데, |<i>S</i>|&gt;1이면 '
  '이것은 잘못 정의된 문제다: (<i>A</i>,<b>b</b>)의 어떤 것도 심어진 해를 다른 해와 구분하지 못하므로 '
  'label은 posterior가 아니라 <b>posterior로부터의 표본</b>이다. strong-branching imitation처럼 비싼 '
  'expert를 모방하는 것은 그 결정이 유익하면서도 수집 가능한 expert를 요구하는데, 여기서 자연스러운 후보인 '
  'LP-probing은 node당 변수당 linear program 한 번을 지불한다.', body_s)
P('우리는 세 번째 길을 택한다. 이 인스턴스들이 planted solution을 갖고 적당한 크기에서 feasibility를 '
  '열거할 수 있기 때문에 가능한 길이다: 작은 인스턴스에서 <b>전수열거로 정확한 conditional posterior</b>를 '
  '계산하고 그것에 학습한다. §3은 이것이 실행 가능하며, 정확한 의미에서 올바른 target임을 보인다 — '
  '그리고 그 posterior가 <b>어떤 예측이 탐색에 영향을 줄 수 있는지</b>까지 함께 식별함을 보인다.', body_s)

H2('1.3 기여')
P('<b>(1) in-tree 지도학습을 간단히 만드는 reduction 항등식.</b> 식 (1)을 partial assignment로 '
  'conditioning하면 같은 구조의 더 작은 BLS가 된다(명제 1). 따라서 하나의 architecture, 하나의 label 생성기, '
  '하나의 학습 루프가 root와 모든 interior node를 담당하며, depth-<i>d</i> 학습 상태는 <i>n</i>−<i>d</i>개 '
  '변수를 갖는 평범한 instance로 저장된다.', body_s)
P('<b>(2) 비용을 발생시킬 수 있는가에 따른 branching 결정의 분할.</b> 정확한 marginal이 node의 자유 변수를 '
  'DET(모든 살아남은 해가 일치; 반대로 분기하면 subtree가 빔)와 OPEN(해들이 갈림; 어느 쪽이든 해에 도달)으로 '
  '나눈다. <b>DET 오류만이 backtrack을 유발한다.</b> 그 구성을 측정하면 root에서 6.3%, depth 8에서 '
  '<b>89.0%</b>다(§4.3, §4.4). 이는 쓸모 있는 예측이 root가 아니라 <b>tree 내부</b>에 있음을 특정하며, '
  'root 수준 정확도가 왜 나쁜 목적함수인지 설명한다.', body_s)
P('<b>(3) 근사 대상인 건전 절차 대비 비용 특성 규명.</b> network는 relaxation 한 번과 forward pass 한 번, '
  'LP-probing은 변수마다 relaxation 한 번이 필요하다. 양쪽 모두 warm start한 backend에서 측정하면 비율이 '
  'n=25의 1.7배에서 n=150의 17.8배로 커진다(§4.5). 단일 수치가 아니라 <b>스케일링 법칙</b>으로 보고하며, '
  '우리 탐색이 실제로 작동하는 크기에서는 우위가 크지 않고 probing에는 <b>건전성</b>이라는 추가 장점이 '
  '있다는 점도 함께 밝힌다.', body_s)
P('<b>(4) wall-clock 예산을 맞춘 종단 평가, 3-seed 반복.</b> 21×60에서 학습된 guidance는 LP 기반 branching 대비 '
  'median 3.29배 빠르고(seed별 sign test <i>p</i>=0.019), node를 6.2배 적게 전개하며, 모든 반복에서 600초 내 '
  '30/30 대 25/30을 해결한다(§4.6). 실제 cascade에서는 end-to-end 해결율이 94.4±5.1%에서 100±0.0%로 오른다(§4.10).', body_s)
P('<b>(5) solution multiplicity를 통제한 크기 transfer.</b> |<i>S</i>|를 크기 간에 비슷하게 유지하면, '
  '10×25에서만 학습한 network가 평가 크기 21×60에서 학습한 network와 대등하다(30개 중 15개에서 더 빠르고 '
  '총 시간 차 0.4%). 학습된 양이 크기에 특수하지 않음을 보인다(§4.8).', body_s)
E.append(PageBreak())

# ═══════════════════════ 2. 선행연구 ═══════════════════════
H1('2. 선행연구')
H2('2.1 Binary linear system과 market-split의 난이도')
P('식 (1)은 0/1 integer feasibility 문제이며 일반 <i>A</i>에 대해 NP-complete이다. 그러나 실무적 난이도는 '
  'instance 분포에 달려 있다. Cornuéjols와 Dawande(1999)는 <b>market-split</b> 계열을 도입했다: '
  '<i>m</i>명의 agent가 <i>n</i>개 품목을 각자의 예산에 정확히 맞게 나눠야 하며, <i>n</i>≈10(<i>m</i>−1)일 때 '
  '놀랍도록 작은 크기에서 branch-and-bound를 무력화한다. 메커니즘은 <b>integrality gap</b>이다: '
  'relaxation polytope이 크고 그 vertex가 fractional이라 bounding이 거의 가지치기를 못 하고, '
  '탐색은 사실상 눈먼 채로 분기해야 한다. Aardal 등(2000)은 lattice basis reduction이 같은 계열을 '
  'branch-and-bound보다 훨씬 효과적으로 공략함을 보였고, 우리는 §3.2에서 이를 활용한다.', body_s)
P('우리 생성기(§4.1)는 relaxation vertex들의 퍼짐(spread)을 최대화해 이 메커니즘을 명시적으로 만든다. '
  '이는 <i>S</i><sub>LP</sub>를 키우고 relaxation signal을 설계 단계에서 약화시킨다.', body_s)

H2('2.2 Complete solver')
P('<b>Mixed-integer programming.</b> linear relaxation 위의 branch-and-bound를 presolve와 cutting plane으로 '
  '강화하는 것이 표준이다. 그 지렛대는 relaxation이 infeasible하거나 target에서 멀리 떨어진 node를 '
  '가지치는 데서 나오는데, market-split 구성이 제거하는 것이 정확히 그것이다.', body_s)
P('<b>Clause learning을 갖춘 constraint programming.</b> CP-SAT 계열 solver는 constraint propagation에 '
  'conflict-driven clause learning(CDCL)과 lazy clause generation을 결합한다. 이 계열에서의 강점은 '
  '본 논문의 논지에 핵심적인 성질에서 나온다: <b>탐색은 내려가면서 정보를 만들어낸다.</b> '
  'Pr[<i>x<sub>j</sub></i>=1 | <i>A</i>,<b>b</b>]가 root에서 무정보해도 '
  'Pr[<i>x<sub>j</sub></i>=1 | <i>A</i>,<b>b</b>,<i>x</i><sub>1</sub>=1,<i>x</i><sub>5</sub>=0]은 0이나 1일 수 있다. '
  'root에서 보이지 않던 구조가 commitment 이후 연역 가능해지며, 이것이 §3.4가 root가 아니라 '
  'tree 내부를 다루는 이유다.', body_s)
P('<b>Lattice reduction.</b> 대안적 접근은 식 (1)을 lattice에 embedding하고 짧은 vector를 찾는다. '
  'LLL은 다항시간에 basis를 축소하고, BKZ는 block size로 시간과 품질을 교환하며 이를 일반화한다. '
  'Aardal–Hurkens–Lenstra(AHL) embedding은 0/1 해가 <b>짧은 vector로 직접 나타나도록</b> 구성되어, '
  'reduction이 heuristic이 아니라 <b>certificate를 생산하는 절차</b>가 된다. 우리는 이를 전체 pipeline의 '
  '첫 단계로 사용한다.', body_s)

H2('2.3 조합최적화를 위한 학습')
P('Bengio 등(2021)은 이 분야를 개관하며 우리가 채택하는 표현을 쓴다: solver는 '
  '"<b>계산하기에 너무 비싸거나</b> 수학적으로 잘 정의되지 않은 결정"에 대해 수작업 heuristic에 의존하며, '
  '학습이 그 결정을 개선할 후보라는 것이다. <b>too expensive to compute</b>라는 표현은 그 결정이 '
  '원리적으로 계산 가능함을 전제하고 <b>비용만을 묻는다</b> — 이것이 §4.5에서 우리가 측정하는 상황이다.', body_s)
P('<b>Branching imitation.</b> 이 분야의 가장 분명한 성공은 비싼 expert의 모방이다. strong branching은 '
  '두 자식의 relaxation을 잠정적으로 풀어 후보에 점수를 매기며, 작은 tree를 만들지만 매 node에서 돌리기에는 '
  '너무 느리다. Gasse 등(2019)은 bipartite graph network에 이를 모방시켜 훨씬 적은 비용으로 이득 대부분을 '
  '회수했다. 이득은 <b>새 정보가 아니라</b> — expert는 이미 답을 알고 있었다 — 그 지식에 값싸게 접근할 수 '
  '있게 된 것이다. 우리 방법은 같은 구조적 위치를 차지하되 두 가지가 다르다: target이 expert의 점수가 아니라 '
  '<b>정확한 posterior</b>이며, 분석이 <b>expert의 결정 중 어느 것이 중요한지</b>를 식별한다.', body_s)
P('<b>직접적 satisfiability 예측.</b> network가 satisfiability를 종단으로 판정하게 하려는 시도는 기록이 더 약하다. '
  'NeuroSAT은 literal-clause graph 위의 message passing으로 분류하며 작은 random 인스턴스에서 성공하지만, '
  'G4SATBench는 크기 transfer에서 상당한 정확도 붕괴를 보고한다. Chen 등(2023)은 표준 message-passing '
  'network가 어떤 feasible/infeasible MILP 쌍을 <b>원리적으로</b> 구분할 수 없음을 증명했다. '
  '따라서 우리는 network에게 feasibility 판정을 요구하지 않는다. network는 건전성을 책임지는 '
  'complete search 내부에서 <b>branching guidance만</b> 제공한다.', body_s)

H2('2.4 Amortized optimization')
P('우리 결과를 조직하는 구분은 조합최적화에서 쓰이기 전부터 있었다. Gershman과 Goodman(2014)은 '
  '<b>amortized inference</b>를 도입했다: 관련된 query가 많을 때 매번 처음부터 푸는 것은 낭비이며, '
  '질문은 query 간에 계산을 어떻게 재사용하는가가 된다. Amos(2023)는 같은 아이디어를 '
  '<b>amortized optimization</b>으로 전개한다 — 유사한 problem instance 간의 공유 구조를 활용해 학습으로 해를 '
  '예측하는 것 — 그리고 variational inference, meta-learning, control에서의 등장을 개관한다.', body_s)
P('Millidge(2022)는 그 trade-off를 여기서 가장 관련 있는 형태로 진술한다: <b>direct</b> optimization은 '
  '당면 instance에 계산을 쓰며 호출할 때마다 그 비용을 지불하고, <b>amortized</b> optimization은 문제를 '
  'supervised learning으로 바꿔 inference를 값싸게 만들지만 풀린 인스턴스 corpus가 선결 조건이고 '
  'approximator의 generalization에 갇힌다. 생산적인 배치는 <b>hybrid</b>다 — amortized 예측이 '
  'direct optimizer를 안내한다.', body_s)
P('우리 시스템이 그 hybrid를 정확히 구현한다. direct 방법은 <b>LP-probing</b>이며 건전하되 비싸다. '
  'amortized 방법은 <b>network</b>이며 값싸되 건전하지 않다. hybrid 규칙은 network가 branching을 안내하되 '
  'probing과 propagation이 변수 확정에 대한 배타적 권한을 유지하는 것이다(§3.7).', body_s)

# ═══════════════════════ 3. 제안 방법 ═══════════════════════
H1('3. 제안 방법')
H2('3.1 설정과 표기')
P('변수 집합 <i>F</i>를 값 <b>v</b>로 고정하는 partial assignment에 대해, 그와 부합하는 해집합을 '
  '<i>S</i>(<i>F</i>,<b>v</b>) = {<b>x</b>∈<i>S</i> : <b>x</b><sub><i>F</i></sub>=<b>v</b>}라 쓰고, '
  '자유 변수 집합을 <i>K</i>라 한다. 자유 변수 <i>j</i>∈<i>K</i>의 '
  '<b>conditional marginal</b>은 다음과 같다:', body_s)
EQ('<i>p<sub>j</sub></i>(<i>F</i>,<b>v</b>) = |{<b>x</b>∈<i>S</i>(<i>F</i>,<b>v</b>) : '
   '<i>x<sub>j</sub></i>=1}| / |<i>S</i>(<i>F</i>,<b>v</b>)|  &nbsp;&nbsp;(2)')

H2('3.2 시스템 개요')
P('실제 solver는 lattice 단계와 guided complete search의 cascade다:', body_s)
EQ('propagation → LP infeasibility rule → kernel pump → AHL/BKZ → <b>guided backtracking search</b>')
P('앞의 세 단계는 값싸고 건전하다. AHL/BKZ는 크기 의존적 block 파라미터로 certificate를 생산하는 '
  'lattice reduction을 시도하며, 성공하면 검증된 해를 바로 반환한다. <b>학습 컴포넌트는 마지막 단계에만, '
  '그것도 branching 결정에만 나타난다.</b> 이 배치는 평가에서 중요하다: lattice 단계가 branching heuristic과 '
  '무관하므로 비교 대상인 모든 heuristic이 <b>정확히 같은 잔여 instance 집합</b>을 받으며, 따라서 그 잔여 위에서 '
  '조건부로 보는 것은 유리한 사례를 고르는 것이 아니라 알고리즘을 기술하는 것이다.', body_s)

H2('3.3 Conditioning은 reduction이다')
P('<b>명제 1.</b> <i>A<sub>F</sub></i>와 <i>A<sub>K</sub></i>를 각각 <i>F</i>, <i>K</i>로 인덱싱된 열 부분행렬이라 하자. 그러면<br/>'
  '&nbsp;&nbsp;&nbsp;&nbsp;<i>S</i>(<i>F</i>,<b>v</b>) ≅ {<b>y</b>∈{0,1}<sup>|<i>K</i>|</sup> : '
  '<i>A<sub>K</sub></i><b>y</b> = <b>b</b> − <i>A<sub>F</sub></i><b>v</b>} &nbsp;&nbsp;(3)<br/>'
  '이며 대응은 <b>x</b> → <b>x</b><sub><i>K</i></sub>이다. 따라서 원본 instance의 conditional marginal (2)는 '
  '<b>축소된 instance의 통상적 marginal</b>이다.', prop_s)
P('<b>증명.</b> <i>A</i><b>x</b> = <i>A<sub>F</sub></i><b>x</b><sub><i>F</i></sub> + '
  '<i>A<sub>K</sub></i><b>x</b><sub><i>K</i></sub>이다. <b>x</b><sub><i>F</i></sub>=<b>v</b>로 고정하면 '
  '<i>A</i><b>x</b>=<b>b</b> ⇔ <i>A<sub>K</sub></i><b>x</b><sub><i>K</i></sub> = <b>b</b>−<i>A<sub>F</sub></i><b>v</b>이고, '
  '<b>x</b><sub><i>F</i></sub>가 결정되어 있으므로 <b>x</b>→<b>x</b><sub><i>K</i></sub>는 두 집합 사이의 전단사다. '
  'marginal에 대한 진술은 개수를 세면 따라 나온다. □', body_s)
P('이 항등식은 초등적이지만, <b>in-tree 지도학습을 실행 가능하게 만드는 것이 바로 이것이다.</b> '
  'depth-<i>d</i> search node는 자기만의 encoding을 요구하는 특수한 대상이 아니라, 같은 문제의 '
  '<i>n</i>−<i>d</i>개 변수 instance일 뿐이다. 따라서 하나의 network, 하나의 feature 추출기, 하나의 label 생성기가 '
  'tree 전체를 담당하며, depth <i>d</i>의 학습 데이터는 solver를 계측하는 것이 아니라 <b>대입</b>으로 생산된다.', body_s)
P('두 구현 세부가 정합성을 유지한다. 대입 후 전부 0이 된 row는 제약을 부과하지 않으므로 제거한다 — '
  '남겨두면 linear programming backend가 모델을 거부한다. 그리고 자유 bound 조건 '
  '0 ≤ <b>b</b>−<i>A<sub>F</sub></i><b>v</b> ≤ <i>A<sub>K</sub></i><b>1</b>을 relaxation을 풀기 전에 검사한다. '
  '무시할 만한 비용으로 dead node의 상당 부분을 걸러내기 때문이다.', body_s)

H2('3.4 어떤 예측이 비용을 발생시킬 수 있는가')
P('살아남은 해집합이 비어 있지 않은 node에서, 자유 변수를 conditional marginal로 분할한다:', body_s)
EQ('<b>DET</b> = {<i>j</i> : <i>p<sub>j</sub></i> ∈ {0,1}},&nbsp;&nbsp;&nbsp;&nbsp; '
   '<b>OPEN</b> = {<i>j</i> : 0 &lt; <i>p<sub>j</sub></i> &lt; 1}')
P('이 구분은 통계적이 아니라 <b>운용적</b>이다. <i>j</i>∈OPEN이면 <i>x<sub>j</sub></i>로 분기한 두 자식 모두 '
  '해를 포함하므로, 어느 쪽을 골라도 해가 도달 가능한 채로 남는다 — 즉 <b>OPEN 변수에서의 어떤 branching 결정도 '
  '비용을 발생시킬 수 없다.</b> <i>j</i>∈DET이면 한쪽 자식이 비어 있다. 강제된 값을 거슬러 분기하면 '
  'subtree가 해 없이 소진되는 것이 확정되며, 이는 온전한 backtrack 비용이다.', body_s)
P('이것이 conditional marginal의 운용적 독법을 준다. 그것은 지정된 어떤 해에 대한 "맞힐 확률"이 아니라 '
  '<b>그 배정으로 살아남는 해집합의 비율</b>이며, 중요한 오류는 그 비율이 0이거나 1인 변수에서의 오류뿐이다.', body_s)
P('두 가지 귀결이 따르고, 실험이 둘 다 확인한다. <b>첫째</b>, 모든 자유 변수에 대한 총계 정확도는 나쁜 '
  '목적함수다. backtrack을 유발할 수 있는 결정과 그럴 수 없는 결정을 섞기 때문이다. 관련 있는 양은 '
  '<b>DET에서의 정확도</b>다. <b>둘째</b>, 탐색이 내려갈수록 |<i>S</i>(<i>F</i>,<b>v</b>)|가 줄어들므로 '
  'DET 비중이 depth에 따라 커진다. 다만 이것만으로는 학습된 predictor가 <b>어디서</b> 유용한지 말할 수 없다 — '
  'relaxation이 이미 올바르게 반올림하는 DET 변수에는 predictor가 필요 없기 때문이다. 유용한 영역은 DET 변수가 많으면서 '
  '<b>동시에</b> relaxation이 약한 곳이며, §4.9는 이를 지배하는 것이 depth 자체가 아니라 <b>제약 밀도 <i>m</i>/|<i>K</i>|</b>임을 '
  '보인다: 작은 instance에서는 둘이 일치하지만 큰 instance에서는 그렇지 않다.', body_s)

H2('3.5 학습 target으로서의 정확한 conditional marginal')
P('전수열거가 가능한 크기의 instance에서는 constraint solver의 all-solutions 모드로 <i>S</i>를 정확히 얻고, '
  '식 (2)를 개수로 계산한다. 학습은 그 결과 얻어지는 Bernoulli target에 대한 cross-entropy를 최소화한다:', body_s)
EQ('<i>L</i> = −(1/|<i>K</i>|) Σ<sub><i>j</i></sub> [ <i>p<sub>j</sub></i> log <i>p</i>^<sub><i>j</i></sub> '
   '+ (1−<i>p<sub>j</sub></i>) log(1−<i>p</i>^<sub><i>j</i></sub>) ]  &nbsp;&nbsp;(4)')
P('이는 정확히 <i>p</i>^<sub><i>j</i></sub> = <i>p<sub>j</sub></i>에서 최소화된다. <i>p<sub>j</sub></i> 자리에 '
  '심어진 해를 넣는 관례적 선택은 |<i>S</i>|&gt;1일 때마다 <b>잡음 섞인 대리 target</b>이 된다. '
  '심어진 해는 posterior에서 뽑은 한 표본이며, 우리 인스턴스에서 Bayes-optimal 예측조차 그것과 '
  '약 2/3만 일치한다.', body_s)
P('<b>열거 정합성.</b> 여기에 놓치기 쉬운 함정이 있고, 그것을 놓치면 파생된 모든 label이 조용히 오염된다. '
  'constraint solver는 "탐색 소진"과 "해를 이미 모은 상태에서 시간제한 도달"을 서로 다른 status로 보고한다. '
  '전자만이 <i>S</i>가 완전함을 보증한다. 후자를 인정하면 <b>잘린 해집합</b>이 대입되어 그로부터 계산된 모든 '
  'marginal이 편향된다. 우리 구현은 OPTIMAL과 INFEASIBLE만 인정하고 시간 초과한 instance는 평균에 섞지 않고 '
  '폐기한다. 유용한 진단 지표는 <b>평균 열거 시간이 시간제한과 같아지는 것</b>인데, 이는 잘린 instance가 '
  '완료로 집계되고 있다는 신호다.', body_s)
P('<b>도달 가능한 상태의 표집.</b> depth-<i>d</i> 학습 상태는 <i>d</i>를 {0,…,12}에서 균등하게 뽑고, '
  'LP 확신도 순서의 앞 <i>d</i>개를 취해, 무작위로 고른 <b>실제 해</b>의 값으로 고정해 만든다. '
  '임의의 값으로 고정하면 <i>S</i>(<i>F</i>,<b>v</b>)=∅인 상태가 생기는데, 그런 상태는 propagation이 즉시 '
  '가지치는 dead node이고 conditional marginal이 정의되지 않으며, branching heuristic이 질문받을 일이 없는 '
  '상태다. 이 depth 범위는 튜닝하지 않은 설계 선택이며, 10×25 instance에서 자유 변수 13~25개의 학습 상태를 만든다. '
  '배포된 탐색이 그 범위 밖의 상태를 질의하면 — 21×60에서는 대부분이 그렇다 — 어떻게 되는지는 §4.9가 다룬다.', body_s)

H2('3.6 Network')
P('instance는 <i>n</i>개 variable node, <i>m</i>개 constraint node, 그리고 <i>a<sub>ij</sub></i>=1인 곳마다 '
  'edge를 갖는 bipartite graph로 encoding된다. 현재(축소된) instance의 relaxation 해를 '
  '<b>x</b>~, row sum을 <i>r<sub>i</sub></i>라 하면 node feature는 다음과 같다:', body_s)
EQ('변수 <i>j</i>: ( <i>x</i>~<sub><i>j</i></sub>, deg(<i>j</i>)/<i>m</i>, '
   '|<i>x</i>~<sub><i>j</i></sub> − round(<i>x</i>~<sub><i>j</i></sub>)| )')
EQ('제약 <i>i</i>: ( <i>b<sub>i</sub></i>/<i>r<sub>i</sub></i>, <i>r<sub>i</sub></i>/<i>n</i>, '
   '(<i>b<sub>i</sub></i> − <i>A<sub>i</sub></i><b>x</b>~)/<i>r<sub>i</sub></i> )')
P('즉 변수에는 relaxation 값·열 차수·fractionality, 제약에는 tightness·scaled row size·relaxation 잔차다. '
  '모두 <b>크기 정규화</b>되어 있으며, 이것이 하나의 파라미터 집합이 임의의 (<i>m</i>,<i>n</i>)에 적용될 수 있게 한다.', body_s)
P('두 node 유형 모두 선형 사상으로 <i>h</i>=64차원에 embedding되고, graph attention을 쓰는 양방향 '
  'message passing <i>L</i>=4라운드로 정련된다. 각 라운드는 변수→제약, 제약→변수 순이며 각각 '
  'LayerNorm과 residual 연결을 갖는다. 이어 2층 head가 변수당 logit 하나를 내어 '
  '<i>p</i>^<sub><i>j</i></sub> = σ(<i>z<sub>j</sub></i>)를 준다. 모델 파라미터는 <b>21,761개</b>다.', body_s)
P('architecture는 의도적으로 작고 단일 head다. 유일한 출력은 목적함수 (4)가 정의하는 양이며, '
  '보조적인 value나 feasibility head가 없다. 탐색은 건전성을 자신의 symbolic 구성요소에서 얻고 '
  'network에게는 그 외의 어떤 것도 요구하지 않기 때문이다.', body_s)

H2('3.7 추론: commitment가 아니라 guidance')
P('network는 node당 한 번 호출된다. branching은 다음을 선택한다:', body_s)
EQ('<i>j</i>* = argmax<sub><i>j</i></sub> | <i>p</i>^<sub><i>j</i></sub> − 1/2 |,&nbsp;&nbsp;&nbsp; '
   '첫 시도 값 = 1[ <i>p</i>^<sub><i>j</i>*</sub> ≥ 1/2 ]')
P('즉 network가 가장 확신하는 변수를, 예측한 값으로 먼저 시도한다. 탐색은 complete하게 유지되고 '
  '잘못된 예측의 대가는 backtracking뿐이다.', body_s)
P('변수 <b>확정</b>은 건전한 deduction만이 수행한다. 두 메커니즘이 있다: constraint propagation, 그리고 '
  '반대값 배정이 relaxation조차 infeasible하게 만들 때 <i>x<sub>j</sub></i>를 고정하는 <b>LP-probing</b>이다. '
  '이 분업은 §4.5의 측정에서 직접 따라 나온다: 오류의 대가가 시간인 곳에서는 값싼 비건전 predictor가 낫고, '
  '오류의 대가가 정확성인 곳에서는 비싼 건전 절차가 필수이며 대체 불가능하다.', body_s)
P('<b>Algorithm 1 — Guided backtracking search</b><br/>'
  '<font face="Courier" size="8">'
  '입력: instance (A,b), guidance model f, 시간 예산 T<br/>'
  'stack ← [(A,b)]<br/>'
  'while stack ≠ ∅ and 경과시간 &lt; T:<br/>'
  '&nbsp;&nbsp;(A\',b\') ← pop(stack)<br/>'
  '&nbsp;&nbsp;if b\' &lt; 0 or b\' &gt; A\'·1: continue&nbsp;&nbsp;// 자유 bound 검사<br/>'
  '&nbsp;&nbsp;A\'의 전부-0 row 제거; 남은 row 없으면 return 임의 완성<br/>'
  '&nbsp;&nbsp;propagate; 모순이면 continue<br/>'
  '&nbsp;&nbsp;if propagation이 변수 집합 D를 결정: push reduce(A\',b\',D); continue&nbsp;&nbsp;// 명제 1<br/>'
  '&nbsp;&nbsp;if 모든 변수 배정됨: 검증되면 return 해, 아니면 continue<br/>'
  '&nbsp;&nbsp;p^ ← f(A\',b\')&nbsp;&nbsp;// 단일 forward pass, 전 변수<br/>'
  '&nbsp;&nbsp;j* ← argmax |p^_j − 1/2|;&nbsp; v ← 1[p^_j* ≥ 1/2]<br/>'
  '&nbsp;&nbsp;push reduce(A\',b\',{j*=1−v}), 이어서 reduce(A\',b\',{j*=v})<br/>'
  'return unresolved</font>', prop_s)

H2('3.8 비용 모델')
P('설계를 뒷받침하는 비교는 <b>instance 전체 판정 1회당 node 비용</b>이며, 그 구조는 상수배가 아니라 '
  '<b>점근적</b>이다. node의 자유 변수 수를 <i>n<sub>c</sub></i>라 하자. network는 feature를 만들기 위해 '
  'relaxation을 <b>한 번</b> 풀고 forward pass를 <b>한 번</b> 수행하므로 LP 호출 수가 '
  '<i>n<sub>c</sub></i>에 대해 <b>O(1)</b>이다. LP-probing은 후보 변수마다 하나씩 <b>O(<i>n<sub>c</sub></i>)</b>회 푼다. '
  '따라서 둘의 비율은 instance 크기에 비례해 커지며, <b>이 비율로 인용되는 단일 수치는 그것이 측정된 '
  '<i>n<sub>c</sub></i>와 함께여야만 의미가 있다.</b>', body_s)
P('이는 또한 이 비교가 <b>LP backend 구현에 민감하다</b>는 뜻이며, 그 민감성은 학습 방법에 불리한 방향으로 '
  '작용한다. 호출마다 모델을 재구성하는 backend는 probing의 비용을 O(<i>n<sub>c</sub></i>) 배수만큼 부풀리면서 '
  'network에는 한 번만 청구해 우위를 과장한다. 제대로 구현된 branch-and-bound는 relaxation 하나를 유지하고 '
  'bound 변경 후 부모 basis에서 재최적화하며, 이는 probe당 한 자릿수 이상 싸다. §4.5가 양쪽 모두 '
  '<b>warm start한 backend</b>로 측정하고 비율을 단일 수치가 아니라 <i>n<sub>c</sub></i>의 함수로 보고하는 이유가 이것이다.', body_s)

E.append(PageBreak())

# ═══════════════════════ 4. 실험 ═══════════════════════
H1('4. 실험')
H2('4.1 인스턴스와 solution multiplicity 통제')
P('인스턴스는 GenHard(<i>m</i>,<i>n</i>,<i>K</i>) 절차로 생성한다: 밀도 1/2의 무작위 0/1 행렬 <i>A</i>를 뽑고, '
  '<i>K</i>=20개의 후보 해를 심어 각각 <b>b</b>를 계산한 뒤, <b>vertex spread</b>(무작위 objective 방향에서 얻은 '
  'relaxation vertex들의 평균 쌍거리)가 최대인 것을 채택한다. vertex spread 최대화는 '
  '<i>S</i><sub>LP</sub>를 직접 키우며, market-split 인스턴스를 relaxation 기반 방법에 어렵게 만드는 '
  '메커니즘이다(§2.1). 심어진 해의 유일성은 강제하지 않는다.', body_s)
P('<b>왜 크기 간 |<i>S</i>|를 통제해야 하는가.</b> solution multiplicity는 예측 과제의 난이도만이 아니라 '
  '<b>성격</b>을 바꾼다. |<i>S</i>|=1이면 모든 자유 변수가 DET이고 잘 예측하는 것이 곧 푸는 것이지만, '
  '|<i>S</i>|가 크면 대부분이 OPEN이고 과제는 퍼진 posterior를 추정하는 것이 된다. 따라서 |<i>S</i>|를 '
  '통제하지 않고 크기를 비교하면 크기와 과제 정체성이 교락된다. 고정된 <i>n</i>에서 <i>m</i>이 오르면 '
  '|<i>S</i>|가 줄어들므로, 크기마다 <i>m</i>을 골라 이를 맞춘다.', body_s)
TBL([['크기', '<i>m</i>', '<i>m/n</i>', '|<i>S</i>| 중앙값', 'root ceiling', 'conditional ceiling'],
     ['10×25', '10', '0.40', '23', '70.4%', '87.0%'],
     ['18×50', '18', '0.36', '19', '71.3%', '87.2%'],
     ['21×60', '21', '0.35', '10', '—', '—']],
    [2.6*cm, 1.5*cm, 1.6*cm, 2.6*cm, 2.8*cm, 3.4*cm],
    '표 1. 사용한 크기와 맞춰진 solution multiplicity. ceiling(§4.3의 Bayes-optimal per-variable 정확도)이 '
    '두 학습 크기 간에 1%p 이내로 일치하므로, 변하는 변수는 크기다.')
P('<b>통제의 한계.</b> |<i>S</i>|를 맞추면 <i>m/n</i>이 변한다(0.40→0.35). 고정된 <i>n</i>에서 둘을 동시에 '
  '유지할 수 없고, 과제 정체성을 결정하는 |<i>S</i>|를 우선했다. 통제는 또한 <b>입자적</b>이다: <i>n</i>=60에서 '
  '|<i>S</i>| 중앙값은 <i>m</i>=20,21,22에 대해 42→12→2로 계단을 이루므로 <i>m</i>=21이 사용 가능한 최선의 '
  '정수이며, 실제 평가셋은 중앙값 10에 27/30이 검증됐다.', body_s)
P('<b>열거가 멈추는 지점.</b> <i>n</i>=70에서는 목표 multiplicity를 주는 <i>m</i>에서 120초 내에 열거가 '
  '하나도 완료되지 않았다(0/10). <i>n</i>=100에서는 장애가 더 근본적이다: 심어진 해가 있음에도 '
  '<i>m</i>∈{28,32,36,40}에서 constraint solver가 20초 내에 <b>해를 하나도</b> 찾지 못한다. 그 크기에서 '
  '병목은 분기를 잘 고르는 것이 아니라 해를 하나라도 찾는 것이므로 branching 비교가 유익하지 않다. '
  '따라서 <i>n</i>≤60에서 평가하며 이 경계를 한계로 보고한다.', body_s)

H2('4.2 구현과 프로토콜')
P('<b>소프트웨어.</b> 학습 feature와 원래 탐색 루프의 relaxation은 PySCIPOpt를 통한 SCIP로 풀고, §4.5·§4.6의 warm start '
  'relaxation은 Gurobi 13(dual simplex, bound 제자리 변경)을 쓴다 — 라이선스가 필요하며, HiGHS가 동등한 incremental 인터페이스를 '
  '제공하므로 무료 대체가 가능하겠으나 그 타이밍은 검증하지 않았다. 열거와 CP-SAT baseline은 OR-Tools, lattice reduction은 '
  'fpylll(LLL, BKZ)이다. network는 PyTorch와 PyTorch Geometric(GATConv)으로 구현했다. relaxation의 한 성질을 적어둔다: '
  '<b>objective가 없으므로</b> 그 해는 relaxation polytope의 임의의 vertex이고 LP 코드마다 다른 vertex를 돌려준다. '
  '한 실험 안에서 두 branching arm은 같은 vertex를 소비하므로 비교는 공정하지만, SCIP 기반과 Gurobi 기반 실행 사이의 절대 node 수는 다르다.', body_s)
P('<b>학습.</b> Adam, learning rate 10<sup>−3</sup>, 20 epoch, in-tree 상태 10,000개, batch size 1'
  '(instance 크기가 달라서), gradient norm clipping 1.0, 정확한 marginal target에 대한 손실 (4). '
  '학습 상태는 depth 0~12에서 균등하게, 원본 instance당 6개씩, 1,760개 instance 풀에서 뽑는다. '
  '평가는 분리된 풀의 440개 instance를 쓴다. 학습은 CPU 코어 1개로 약 <b>20분</b> 걸린다. '
  '<b>하이퍼파라미터는 튜닝하지 않았다</b>: 폭·깊이·learning rate·epoch 수·학습 depth 범위는 처음 시도한 값이며 validation split이 없다. '
  '따라서 보고 수치는 test set 선택으로 오염되지 않았으나, 이 설정이 최적이라는 주장도 하지 않는다.', body_s)
P('<b>스레딩.</b> 타이밍에 민감한 모든 구성요소는 단일 스레드로 실행한다(torch.set_num_threads(1)). '
  '이 크기의 graph에서 다중 스레드 BLAS는 역효과다: forward pass가 8스레드에서 <b>11.3ms</b>, '
  '1스레드에서 <b>0.9ms</b>로 측정된다.', body_s)
P('<b>탐색 평가.</b> 각 arm은 동일한 instance에 대해 subprocess 종료로 강제되는 인스턴스당 hard wall-clock '
  '예산으로 실행된다. lattice reduction이 in-process 타이머로 중단할 수 없는 blocking native 호출 안에서 '
  '실행되기 때문이다. instance는 20코어 머신에서 동시성 <b>6</b>으로 실행한다. 이 낮은 동시성은 의도적이다: '
  'wall-clock 예산 비교는 CPU 경합에 민감하며, 더 높은 동시성에서 단독 실행 시 14~15초에 풀리는 instance가 '
  '20초 상한을 넘는 것을 관측했다.', body_s)
P('<b>통계.</b> 동일 instance에 대한 arm 간 해결율 차이는 McNemar 정확검정으로, 인스턴스 단위 속도 차이는 '
  '양측 부호검정으로 검정한다. 검정력이 다를 수 있으므로 단일 총계 대신 둘 다 보고한다.', body_s)

H2('4.3 Bayes ceiling 대비 예측 품질')
P('<i>S</i>가 정확히 열거되므로 Bayes-optimal per-variable 정확도를 계산할 수 있다:', body_s)
EQ('ceil = (1/|<i>K</i>|) Σ<sub><i>j</i></sub> max(<i>p<sub>j</sub></i>, 1−<i>p<sub>j</sub></i>)  &nbsp;&nbsp;(5)')
P('이는 살아남은 해들의 다수값을 답함으로써 달성되며, (<i>A</i>,<b>b</b>)로부터 <b>x</b>*를 예측하는 어떤 '
  '결정론적 predictor도 이를 넘을 수 없다. 표 2는 <i>S</i>가 완전히 열거된 held-out 10×25 instance 400개에서 '
  '여러 predictor를 이에 대조한다.', body_s)
TBL([['Predictor', '전체 변수', 'top-3', 'marginal과의 <i>L</i><sub>1</sub>'],
     ['Bayes ceiling', '<b>70.8%</b>', '<b>93.2%</b>', '0.0000'],
     ['제안 network', '66.7%', '86.6%', '0.1216'],
     ['LP relaxation rounding', '60.9%', '65.8%', '—']],
    [5.6*cm, 2.8*cm, 2.4*cm, 4.0*cm],
    '표 2. held-out 10×25 평가 풀의 root 상태 182개에서 정확한 posterior 대비 root 수준 예측. '
    'network는 이하 모든 탐색 실험에 쓰인 것과 <b>동일한 checkpoint</b>다. top-3은 predictor가 가장 확신하는 변수 3개로 제한한 것이다.')
P('network는 전체에서 ceiling의 4.1%p, top-3 부분집합에서 6.6%p 이내에 도달하며, relaxation rounding과 ceiling 사이 격차의 '
  '대부분을 회수한다(ceiling 70.8에 대해 60.9→66.7). 설계 선택과 관련된 예비 실험 둘을 완전성을 위해 보고하되, 여기서 평가한 '
  'checkpoint에서 나온 것이 아님을 밝힌다: 이전 10×25 풀에서 selection/assignment/value/feasibility 4개 head를 갖는 '
  '2.11M 파라미터 network는 root 상태에서 60.7%로 relaxation rounding(60.6%)과 통계적으로 구별되지 않았다 — 즉 <b>용량이 '
  '구속 조건이 아니며</b> 단일 목적 head가 낫다. 그리고 단일 head architecture를 심어진 해 대신 정확한 marginal로 학습하면 '
  'posterior와의 <i>L</i><sub>1</sub>이 0.1261에서 0.0977로 개선되고 argmax 정확도는 변하지 않았는데, §3.4의 분석이 '
  'argmax가 아니라 <b>marginal</b>을 소비하므로 이것이 중요하다.', body_s)
P('<b>총계가 가리는 것.</b> 표 3은 같은 root 변수들을 §3.4의 분할로 분해한다.', body_s)
TBL([['root 변수', '개수', '비중', '부분집합 ceiling'],
     ['<b>DET</b> (거슬러 분기하면 subtree가 빔)', '38', '6.3%', '100.0%'],
     ['<b>OPEN</b> (어느 쪽이든 해에 도달)', '562', '93.7%', '67.2%'],
     ['전체', '600', '100%', '69.3%']],
    [8.2*cm, 2.0*cm, 2.0*cm, 3.8*cm],
    '표 3. root 수준 분해, <i>S</i>가 열거된 10×25 instance 24개. '
    '총계는 그 94%가 어떤 branching 결정도 backtrack을 유발할 수 없는 변수로 이뤄져 있다.')
P('root에서는 변수의 93.7%가 OPEN이다. 따라서 총계 ceiling 70.8%는 <b>탐색에 영향을 줄 수 없는 예측이 '
  '지배하며</b>, 영향을 줄 수 있는 6.3%에서는 정보가 이미 완전하다(100%). 결과적으로 root 수준 정확도는 '
  'branching heuristic에게 약한 목적함수이고, 핵심 질문은 정보가 존재하는가가 아니라 <b>무엇이 그것을 '
  '탐지하는가</b>이다 — 다음 절의 주제다.', body_s)

H2('4.4 Depth, 그리고 잔차가 있는 곳')
P('명제 1을 써서, 실제 해에서 뽑은 prefix를 따라 각 depth에서 conditional ceiling과 propagation이 강제하는 '
  '남은 변수의 비율을 측정한다(표 4).', body_s)
TBL([['depth', '살아남은 |<i>S</i>|', 'conditional ceiling', '제안 network', 'LP rounding'],
     ['0', '25.3', '70.8%', '66.7%', '60.9%'],
     ['4', '4.2', '82.3%', '72.8%', '68.5%'],
     ['7', '1.8', '91.5%', '81.2%', '77.8%'],
     ['8', '1.4', '94.4%', '84.3%', '80.9%'],
     ['12', '1.1', '98.8%', '96.8%', '95.5%'],
     ['전체 depth', '—', '87.2%', '80.0%', '76.5%']],
    [2.4*cm, 3.0*cm, 3.8*cm, 3.2*cm, 3.0*cm],
    '표 4. 10×25 평가 풀의 held-out in-tree 상태에서 depth에 따른 conditional 정보량(<i>n</i>=25, LP 확신도 변수 순서). '
    '표 2와 동일한 checkpoint.')
P('중요한 변화는 수준이 아니라 <b>구성</b>에 있다: DET 비중이 root의 6.3%에서 depth 8의 <b>87.9%</b>로 오른다(표 5). '
  'root에서는 거의 모든 branching 결정이 무해하다 — 양쪽 자식 모두 해를 포함하기 때문이다. depth 8에서는 '
  '거의 모든 결정이 틀리면 subtree를 비운다. network의 relaxation rounding 대비 우위는 전 depth에서 3~6%p이며 depth에서 '
  '사라지지 않는다 — network가 포착하는 것이 단순히 depth 의존적이지 않다는 첫 징후이며, §4.9가 이를 정밀하게 다룬다.', body_s)
P('<b>다른 모든 것을 통제했을 때 in-tree 지도가 중요한가?</b> root 학습 network와 in-tree 학습 network의 비교는 '
  '다른 것이 하나도 다르지 않을 때만 유익하다. 표 4b는 instance 풀, label 종류(정확한 marginal), architecture, optimizer, '
  '학습 상태 수(각 1,760개)를 고정하고, 상태가 root 상태인지 depth 0~12에서 뽑은 in-tree 상태인지만 바꾼다. '
  '둘 다 같은 held-out in-tree 상태에서 평가한다.', body_s)
TBL([['depth', 'conditional ceiling', 'root 학습', 'in-tree 학습', 'LP rounding'],
     ['0', '70.7%', '67.8%', '66.5%', '60.9%'],
     ['4', '82.1%', '71.4%', '72.4%', '68.3%'],
     ['6', '89.7%', '74.6%', '78.2%', '74.0%'],
     ['8', '94.4%', '77.4%', '84.2%', '81.3%'],
     ['10', '97.1%', '81.7%', '91.4%', '89.4%'],
     ['12', '98.8%', '85.8%', '96.5%', '95.3%'],
     ['전체 depth', '87.2%', '75.4%', '<b>79.8%</b>', '76.4%']],
    [2.4*cm, 3.6*cm, 2.8*cm, 3.0*cm, 3.0*cm],
    '표 4b. 학습 상태 분포의 통제 비교. 풀·label·architecture·optimizer·상태 수가 같고 상태를 뽑는 depth만 다르다. '
    '10×25 풀의 held-out in-tree 상태에서 평가.')
P('in-tree 지도의 가치는 전체 4.4%p다. 더 시사적인 것은 root 학습 network 자체다: 75.4%로, in-tree 상태에서는 '
  'relaxation rounding<b>보다 낮다</b>. 즉 root 상태로 학습한 network는 tree 안에서 단지 뒤처지는 것이 아니라 '
  '대체하려던 classical heuristic보다 못하다. root에서 interior로의 이동은 root 학습 network가 일반화하는 분포가 아니며, '
  'in-tree 학습은 학습된 guidance를 baseline보다 낫게 만드는 요인 그 자체다.', body_s)
P('표 5는 이 depth들에서 잔차를 분해하고 각 방법이 무엇을 탐지하는지 묻는다.', body_s)
TBL([['depth', '%DET', 'network<br/>(DET)', 'propagation', 'LP integrality', 'LP-probing', 'network<br/>(OPEN)'],
     ['5', '61.9%', '89.7%', '1.2%', '67.9%', '51.6%', '54.4%'],
     ['6', '74.2%', '88.2%', '1.6%', '70.4%', '68.0%', '52.5%'],
     ['7', '79.9%', '88.8%', '3.7%', '73.3%', '72.8%', '51.2%'],
     ['8', '87.9%', '89.4%', '5.3%', '76.9%', '79.8%', '53.8%']],
    [1.6*cm, 1.8*cm, 2.4*cm, 2.6*cm, 2.8*cm, 2.4*cm, 2.4*cm],
    '표 5. 표 4와 같은 상태에서의 잔차 분해. 4~6열은 ground-truth DET 집합에 대한 recall(LP-probing은 §4.5의 warm start '
    'backend로 측정), 3열과 7열은 각 부분집합에서 network의 argmax 정확도다.', font=8.0)
P('잔차는 전적으로 DET 변수 위에 있다: depth 7에서 0.80×11.2≈9%p로, conditional ceiling 대비 측정된 '
  '10.3%p 미달분과 부합한다. OPEN 변수에서 50% 근처 성능은 결함이 아니라 올바른 거동이다 — 그 변수들은 '
  '진정으로 미결정이기 때문이다. 두드러지는 수치는 <b>unit propagation이 논리적으로 결정된 변수의 '
  '1.2~5.3%만 탐지한다</b>는 것이다: 정보가 존재하고 연역 가능한데도 가장 값싼 건전 규칙으로는 닿지 않는다. '
  'LP-probing은 52~80%를, LP integrality는 68~77%를 회수한다.', body_s)

H2('4.5 Node당 비용')
P('두 방법 모두 instance 전체 판정에 도달하기까지 반드시 치러야 하는 비용 전부를 청구한다: network는 '
  'feature용 relaxation 한 번과 forward pass 한 번, probing은 자유 변수마다 relaxation 한 번. 양쪽 모두 '
  '동일한 <b>warm start backend</b>를 쓴다 — 모델 하나를 유지하며 변수 bound를 제자리에서 바꾸고 부모 basis에서 '
  '재최적화하는 방식(Gurobi dual simplex)으로, branch-and-bound 구현이 relaxation을 재사용하는 방식 그대로다. '
  '모든 측정은 단일 스레드다.', body_s)
TBL([['인스턴스', '<i>n</i>', 'network', 'LP-probing', '배수', 'probing 탐지율'],
     ['10×25', '25', '3.22 ms', '5.60 ms', '1.7×', '—'],
     ['18×50', '50', '2.33 ms', '11.84 ms', '5.1×', '—'],
     ['21×60', '60', '2.57 ms', '18.39 ms', '7.2×', '—'],
     ['40×100', '100', '4.02 ms', '62.46 ms', '15.5×', '—'],
     ['60×150', '150', '4.45 ms', '79.06 ms', '<b>17.8×</b>', '—'],
     ['<i>10×25의 tree 내부 상태, depth별:</i>', '', '', '', '', ''],
     ['depth 5', '20', '2.44 ms', '4.02 ms', '1.6×', '51.6%'],
     ['depth 6', '19', '2.39 ms', '3.98 ms', '1.7×', '68.0%'],
     ['depth 7', '18', '2.35 ms', '3.43 ms', '1.5×', '72.8%'],
     ['depth 8', '17', '2.34 ms', '3.13 ms', '<b>1.3×</b>', '79.8%']],
    [4.4*cm, 1.4*cm, 2.4*cm, 2.6*cm, 1.8*cm, 3.0*cm],
    '표 6. instance 전체 판정 1회당 node 비용. 양쪽 모두 warm start backend, 단일 스레드. '
    'depth 5~8에서 network의 DET 변수 정확도는 85.9~92.7%이며 마지막 열의 probing 탐지율과 대조된다.', font=8.0)
P('비율은 비용 모델이 예측한 대로 거동한다: <i>n</i>=25에서 무시할 만하고 <i>n</i>=150에서 17.8배에 이르며, '
  'relaxation 호출 수의 O(<i>n<sub>c</sub></i>) 대 O(1) 차이를 따라간다. 두 귀결이 따르는데, '
  '<b>서로 반대 방향을 가리킨다.</b>', body_s)
P('<b>우위는 실재하지만 점근적이며, 우리가 탐색할 수 있는 구간에서는 작다.</b> 우리 탐색이 실제로 작동하는 '
  '크기(<i>n</i>=25~60)에서 비율은 1.3~7.2배다. 예측이 가장 중요한 depth(10×25의 5~8, '
  '<i>n<sub>c</sub></i>=17~20)에서는 <b>1.3~1.7배</b>다. 그 규모에서 둘 중 하나를 고르는 실무자는 '
  '<b>probing을 택해야 한다</b> — probing은 건전하기 때문이다: 검사가 발화하면 결론이 증명인 반면 network는 '
  '어떤 보장도 제공하지 않는다. 학습 대안이 설득력을 갖는 것은 <i>n</i>≳100부터이고, §4.1이 보이듯 그 구간에서는 '
  '<b>우리가 시도한 모든 전략에서 탐색 자체가 실패한다.</b>', body_s)
P('<b>측정은 baseline 구현 품질에 민감하다.</b> 이 비교의 이전 버전은 호출마다 LP 모델을 재구성하는 backend를 '
  '썼고, 거기에 더해 network 자신의 feature relaxation을 타이머에서 제외했다. 그 조합이 depth 5~8에서 '
  '<b>19~23배</b>를 보고했는데 올바르게 측정한 값은 <b>1.3~1.7배</b>다 — 한 자릿수 이상의 과장이며, '
  '두 방법이 계산하는 내용이 아니라 <b>전적으로 양쪽을 어떻게 계측했는가</b>에서 비롯됐다. 정정된 수치를 '
  '보고하며 일반적 위험을 함께 적어둔다: classical 절차와의 어떤 비용 비교에서든, classical 쪽은 실무자가 '
  '구현하듯 구현되어야 하고 학습 쪽은 자신의 전처리 비용을 청구받아야 한다.', body_s)

H2('4.6 탐색 성능')
P('lattice 단계를 끄고 branching 품질만 격리한 채, wall-clock 예산을 맞춰 네 가지 branching 전략을 비교한다. '
  'random은 임의 변수·값을, lp는 가장 정수에 가까운 relaxation 값을 갖는 변수를 그 값으로, '
  'lp-probe는 전 자유 변수를 probing해 강제된 것을 건전하게 확정한 뒤 lp처럼 분기하며, '
  'model은 제안 network를 쓴다.', body_s)
TBL([['크기 (예산)', '전략', '해결율', 'node 중앙값', '시간 중앙값', 'node/s'],
     ['10×25 (60초)', 'random', '100%', '1,514', '0.07초', '—'],
     ['', 'lp', '100%', '66', '0.12초', '—'],
     ['', 'lp-probe', '100%', '<b>12</b>', '0.79초', '—'],
     ['', 'model', '100%', '22', '<b>0.08초</b>', '—'],
     ['20×50 (300초)', 'random', '<b>0%</b>', '—', '—', '—'],
     ['', 'lp', '100%', '8,876', '24.93초', '376'],
     ['', 'lp-probe', '100%', '<b>173</b>', '28.14초', '6'],
     ['', 'model', '100%', '1,676', '<b>6.37초</b>', '273'],
     ['21×60 (600초)', 'lp', '90.0%', '69,015', '190.76초', '172'],
     ['', 'model', '<b>100%</b>', '<b>12,010</b>', '<b>47.28초</b>', '135']],
    [3.0*cm, 2.6*cm, 2.2*cm, 2.8*cm, 2.6*cm, 1.8*cm],
    '표 7. branching 전략 비교, lattice 단계 비활성화, 각 칸 30개 instance.')
P('세 가지를 짚는다. <b>첫째</b>, node 수와 wall-clock이 이 방법들을 <b>정반대 순서로</b> 세운다: '
  'lp-probe는 압도적으로 작은 tree를 만들지만(20×50에서 lp 대비 51배 적음) 가장 느리며, 초당 6 node를 '
  '처리하는 반면 lp는 376 node를 처리한다. branching heuristic을 tree 크기만으로 평가하는 통상적 관행은 '
  '여기서 <b>최악의 방법을 선택하게 된다.</b> <b>둘째</b>, model이 생산적인 지점에 위치한다: lp 대비 node를 '
  '5.3배 줄이면서 node당 비용은 lp보다 낮아 시간 중앙값이 3.9배 낮다. <b>셋째</b>, 21×60에서 이점이 '
  '질적으로 변하며 <b>재현된다</b>. 세 seed에 걸쳐 학습된 guidance는 매번 30/30을 풀고 relaxation 기반 '
  'branching은 매번 25/30을 푼다. 양쪽 다 닫은 instance에서 전자가 18/25에서 더 빠르며 median 비율 '
  '<b>3.29배</b>(seed별 3.35, 3.30, 3.24; 각 seed 내 부호검정 <i>p</i>=0.019)다. seed 간 분산은 무시할 만하다 — '
  '해결율 표준편차가 정확히 0이고 속도배수는 ±0.06 — 이는 결과가 <b>tie-breaking 무작위성에 기대고 있지 '
  '않음</b>을 말해준다.', body_s)
P('두 arm을 가르는 5개 instance는 <b>모든 seed에서 동일한 5개</b>이며, lp가 닫고 model이 놓친 instance는 '
  '한 번도 없다. 이 일관성이 이 증거가 취할 수 있는 가장 강한 형태이지만, 통상적 검정으로는 여전히 유의하지 '
  '않다: 30개 instance에서 5–0 분할은 McNemar <b><i>p</i>=0.0625</b>이고, <b>반복을 아무리 늘려도 개선되지 '
  '않는다.</b> 제약이 되는 양이 실행 횟수가 아니라 instance 개수이기 때문이다. 해결율 우위를 통상적 수준에서 '
  '입증하려면 더 많은 seed가 아니라 <b>더 큰 평가셋</b>이 필요하다. 인스턴스 단위로 검정한 속도 우위는 '
  '각 seed 내에서 유의하다.', body_s)

H2('4.7 Complete solver와의 비교')
P('표 7은 <b>우리 탐색 루프 안에서</b> branching 전략을 비교한 것이라 branching 품질은 측정하지만 이 시스템이 '
  '기성 solver에 대해 어디쯤 있는지는 말해주지 않는다. 그 비교를 동일 instance·단일 스레드·동일 예산으로 '
  '여기 보고한다.', body_s)
TBL([['크기', 'CP-SAT', 'SCIP', '제안 (model)', 'CP-SAT 우위'],
     ['10×25', '<b>0.002초</b>', '0.009초', '0.08초', '40배'],
     ['18×50', '<b>0.046초</b>', '0.385초', '2.84초', '62배'],
     ['21×60', '<b>1.159초</b>', '3.687초', '47.28초', '41배']],
    [2.6*cm, 3.0*cm, 2.8*cm, 3.2*cm, 3.0*cm],
    '표 8. 중앙값 해결 시간. 모든 방법이 모든 크기에서 30/30을 해결한다. 제안 시스템은 '
    '<b>어느 크기에서도 complete solver를 이기지 못하며</b>, 자신이 relaxation용으로 이미 쓰고 있는 SCIP에게도 뒤진다.')
P('격차가 크고, 우리는 이를 명확히 진술한다: <b>본 논문의 기여는 CP-SAT와 경쟁하는 solver가 아니다.</b> '
  '다만 격차가 어디서 오는지는 물어볼 가치가 있다. branching heuristic이 실제로 무엇을 하고 있는지에 대한 '
  '답이 거기 있기 때문이다. 표 9는 21×60에서 CP-SAT 자체의 branch 카운터를 써서 격차를 '
  '<b>트리 크기</b>와 <b>node 처리량</b>으로 분해한다.', body_s)
TBL([['', 'CP-SAT', '제안', '비율'],
     ['node (중앙값)', '18,827 branches', '<b>12,010</b>', '우리가 1.6배 적음'],
     ['처리량', '<b>19,782 node/초</b>', '254 node/초', 'CP-SAT 78배 빠름'],
     ['총 시간 (중앙값)', '1.065초', '47.28초', '41배']],
    [4.0*cm, 4.2*cm, 3.4*cm, 4.4*cm],
    '표 9. 21×60 격차의 분해. CP-SAT는 추가로 instance당 중앙값 5,514개의 conflict clause를 학습한다.')
P('학습된 guidance는 CP-SAT보다 <b>작은 탐색 트리</b>를 만든다 — CP-SAT의 clause learning에도 불구하고 node가 '
  '1.6배 적다 — 그리고 전적으로 <b>node당 비용</b>에서 진다. 그 비용이 어디로 가는지는 측정할 수 있다. 21×60 root에서 '
  '우리 루프는 node relaxation을 재구성해 푸는 데 <b>4.34ms</b>, network forward pass에 <b>1.88ms</b>, propagation과 '
  'bookkeeping에 0.1ms 미만을 쓴다. relaxation 비용은 Python을 벗어나지 않고도 제거된다 — 같은 LP의 warm start '
  '재최적화는 <b>0.08ms</b>이며 §4.6은 그 변경을 적용한 탐색을 보고한다 — 그러나 그러면 남는 비용의 90% 이상이 '
  'forward pass이고, 우리는 그것을 줄이지 않았다. 따라서 <b>격차가 구현 산물이라고 주장하지 않는다.</b> node 수가 '
  '확립하는 것은 더 좁다: 부족분이 branching 품질에 있지는 않다는 것이다. 격차를 닫으려면 최소한 compiled inference '
  '경로와, 다음 단락이 적듯 우리 탐색에 없는 알고리즘 구성요소가 필요하다.', body_s)
P('이 해석에는 정직한 단서 두 가지가 붙는다. 트리 크기가 완전히 통약 가능하지는 않다 — CP-SAT의 "branch"에는 '
  'restart와 conflict 기반 점프가 포함되고 각 node가 우리보다 많은 propagation을 수행한다 — 따라서 1.6배는 '
  '자릿수 수준의 진술로 읽어야 한다. 그리고 경쟁력 있는 구현에는 언어 이식 이상이 필요하다: '
  '<b>incremental relaxation, clause learning, restart</b>가 모두 우리 탐색에 없고, 뒤의 둘은 CP-SAT가 우리가 '
  '멈추는 크기를 넘어 확장하게 해주는 요소다.', body_s)

H2('4.8 크기 transfer')
P('feature가 크기 정규화되어 있고 파라미터가 (<i>m</i>,<i>n</i>)에 의존하지 않으므로 하나의 network가 임의 '
  '크기에 적용된다. 표 1처럼 |<i>S</i>|를 맞춘 상태에서 그것이 실제로 전이되는지 검증한다.', body_s)
TBL([['학습 크기', '평가 크기', '해결율', '시간 중앙값', 'node 중앙값'],
     ['10×25', '10×25', '100%', '0.08초', '23'],
     ['10×25', '18×50', '100%', '2.84초', '833'],
     ['18×50', '18×50', '100%', '3.42초', '1,001'],
     ['10×25', '21×60', '100%', '47.28초', '12,010'],
     ['18×50', '21×60', '100%', '54.75초', '14,108']],
    [3.0*cm, 3.0*cm, 2.6*cm, 3.2*cm, 3.0*cm],
    '표 10. Transfer, lattice 단계 비활성화, 행마다 30개 instance. '
    '2~3행과 4~5행은 각각 평가 크기를 고정하고 학습 크기만 바꾼 것이다.')
P('21×60 평가 크기에서, 2.4배 작은 문제인 10×25에서만 학습한 network와 18×50에서 학습한 network는 '
  '<b>통계적으로 구별되지 않는다</b>: 전자가 정확히 30개 중 15개에서 더 빠르고, 속도비 중앙값 0.92배, '
  '총 시간이 0.4% 차이(2,924.9초 대 2,914.5초)이며 양쪽 모두 30/30을 푼다. root 수준 예측도 같은 결론을 준다: '
  '10×25 network가 10×25, 20×50, 40×100에서 각각 67.8%, 69.5%, 69.4%를 기록하며 relaxation rounding 대비 '
  '우위(+11.0, +7.2, +4.7%p)를 모든 크기에서 유지한다.', body_s)
P('이것이 실무적으로 중요한 이유는 <b>label 생성이 구속 조건</b>이기 때문이다. 정확한 marginal은 열거를 '
  '요구하는데 이는 <i>n</i>≈60을 넘으면 실패한다. transfer는 label을 얻을 수 있는 곳에서 학습한 network를 '
  '그럴 수 없는 곳에 <b>배치</b>할 수 있음을 뜻한다.', body_s)

H2('4.9 전이된 network는 어디서 도움이 되는가')
P('§4.4는 유용한 예측 영역을 10×25 instance의 depth 6~8에 위치시켰다. 그 위치가 transfer 후에도 유지되는지는 별개의 '
  '질문이며, 유지되지 않는다: 21×60에서 탐색은 자유 변수 5~60개인 상태에서 network를 질의하는데, 그중 학습 범위 13~25 '
  '안에 드는 것은 <b>19.7%</b>뿐이고 <b>80.2%</b>는 어떤 학습 상태보다 크다. 표 11은 21×60 in-tree 상태 중 15초 안에 '
  '열거 가능한 부분집합에서, 자유 변수 수별로 network와 relaxation rounding을 정확한 conditional marginal에 대조한다.', body_s)
TBL([['자유 변수', '학습 범위 내', '<i>m</i>/|<i>K</i>|', 'network (DET)', 'LP (DET)', '이득', 'corr(network, LP)'],
     ['20', '예', '1.05', '100.0%', '100.0%', '+0.0', '1.000'],
     ['25', '예', '0.84', '100.0%', '100.0%', '+0.0', '0.998'],
     ['30', '아니오', '0.70', '95.7%', '95.0%', '+0.7', '0.954'],
     ['40', '아니오', '0.53', '75.5%', '75.2%', '+0.4', '0.747'],
     ['50', '아니오', '0.42', '70.2%', '65.0%', '<b>+5.3</b>', '0.664'],
     ['60', '아니오', '0.35', '93.3%', '88.9%', '<b>+4.4</b>', '0.658']],
    [2.0*cm, 2.2*cm, 2.0*cm, 2.6*cm, 2.2*cm, 1.6*cm, 3.2*cm],
    '표 11. 자유 변수 수별 21×60 in-tree 상태, 행당 열거 가능 상태 9~14개. 이득은 DET 변수에서 network의 relaxation rounding '
    '대비 마진, 마지막 열은 두 predictor 출력의 Pearson 상관이다.', font=8.0)
P('패턴은 depth 기반 설명이 예측하는 것과 <b>정반대</b>다. 학습 범위 안에서 network는 relaxation rounding과 구별되지 않는다 — '
  '둘 다 완벽하고 출력 상관이 1.000이다 — <i>m</i>/|<i>K</i>|≥0.8에서는 축소 instance가 과제약이어서 relaxation이 이미 '
  '모든 DET 변수에서 정수이기 때문이다. network의 마진 전부는 학습 중 본 적 없는 상태, 즉 자유 변수 50~60개에서 생기며, '
  '거기서 relaxation이 가장 약하고(<i>m</i>/|<i>K</i>|≈0.35~0.42) network 출력은 relaxation과 탈상관되어 있다. '
  '§3.4의 용어로는, DET 변수가 존재하지만 relaxation이 그것을 찾지 못하는 영역이다.', body_s)
P('두 결론이 따른다. 첫째는 §4.4를 교정한다: depth는 10×25에서 우연히 성립한 제약 밀도의 대리변수였다 — 거기서는 root에서 '
  '내려가도 <i>m</i>/|<i>K</i>|가 0.40에서 0.59로만 움직이지만, 21×60에서는 같은 하강이 0.35에서 1.0을 넘어선다. 유용한 '
  '영역은 <i>m</i>/|<i>K</i>|로 정의되며, 배포 크기에서는 tree의 <b>위쪽</b>에 놓인다. 둘째, §4.8의 transfer 결과는 '
  'multiplicity를 맞춘 비교만이 보이는 것보다 강하다: network는 크기와 제약 밀도 양쪽에서 학습 범위 밖의 상태로 외삽하며, '
  '그 외삽이 기여가 있는 곳이다. 동시에 한계이기도 하다 — 학습 범위는 이를 염두에 두고 고른 것이 아니기 때문이다(§4.12).', body_s)

H2('4.10 전체 pipeline에서의 배치')
P('표 7은 lattice reduction을 꺼서 branching을 격리한 것이다. 실제 배치에서는 lattice 단계가 먼저 돌고 '
  'search는 그 잔여만 본다. 그 단계가 branching heuristic과 무관하므로 모든 arm이 동일한 잔여 집합을 받는다.', body_s)
TBL([['크기', '전략', 'lattice가 해결', '잔여를 search가 해결', 'end-to-end'],
     ['10×25', '양쪽', '30/30', '—', '100%'],
     ['18×50', 'lp / model', '24/30', '6/6 / 6/6', '100% / 100%'],
     ['21×60', 'lp', '10~16/30', '45/50', '94.4±5.1%'],
     ['21×60', 'model', '10~16/30', '<b>50/50</b>', '<b>100±0.0%</b>']],
    [2.4*cm, 2.8*cm, 3.2*cm, 4.2*cm, 3.0*cm],
    '표 12. Cascade 결과. 21×60은 3개 seed 기준이고 작은 크기는 단일 실행이다. '
    'lattice coverage가 seed마다 달라지는데(30개 중 16, 14, 10) lattice 단계가 무작위 열 순열을 '
    '뽑기 때문이며, 탐색에 넘어오는 잔여도 그에 따라 변한다.')
P('lattice coverage는 크기에 따라 떨어진다 — 30/30, 24/30, 16/30 — 그리고 그 감소가 <b>branching 품질이 '
  '중요해질 여지를 만든다.</b> 10×25와 18×50에서는 잔여가 비어 있거나 어느 전략으로도 쉽게 닫힌다. '
  '21×60에서는 seed에 따라 10~20개 instance가 search로 넘어오고, 학습된 guidance는 세 반복 모두에서 그 전부를 '
  '닫는다(seed 합산 50/50). relaxation 기반 branching은 45/50을 닫는다. end-to-end로 100±0.0% 대 94.4±5.1%다. '
  'lp arm의 분산은 branching이 아니라 <b>lattice 단계</b>에서 온다: coverage가 낮게 나온 seed에서 더 많은 '
  'instance가 탐색으로 떨어지고, 그 탐색이 때때로 닫지 못한다.', body_s)

H2('4.11 graph 구조가 기여하는가')
P('MarginalNet은 node당 6개 스칼라를 소비해 bipartite graph 위로 전파한다. 특징 집합이 이만큼 작으면 '
  'graph를 무시하는 변수별 모델이 진지한 baseline이 된다 — 그것이 대등하다면 message passing은 장식이다. '
  '표 13은 동일한 target·손실·분할로 학습하고 <b>구조 사용 여부만</b> 다른 모델들을 비교한다. 집계 변형은 '
  '각 변수에 인접한 row들의 제약측 특징 평균과 최대를 덧붙인 것으로, message passing 한 라운드가 전달할 '
  '정보에 해당한다.', body_s)
TBL([['모델', '정확도', 'marginal과의 <i>L</i><sub>1</sub>', '파라미터'],
     ['Linear, 변수 특징만', '78.1%', '0.2530', '<b>4</b>'],
     ['MLP, 변수 특징만', '77.8%', '0.2522', '8,641'],
     ['MLP + 제약 집계', '79.4%', '0.2361', '9,025'],
     ['MarginalNet (bipartite graph)', '<b>84.1%</b>', '<b>0.1514</b>', '21,761'],
     ['Bayes ceiling', '87.3%', '0.0000', '—']],
    [6.4*cm, 2.6*cm, 3.6*cm, 2.6*cm],
    '표 13. 10×25 instance의 tree 내부 상태에서의 ablation, 학습 2,000개·테스트 800개 상태.')
P('message passing은 <b>기여한다</b>: 구조를 쓰지 않는 최선 모델 대비 +4.7%p, <i>L</i><sub>1</sub> 36% 감소이며, '
  'MLP가 남기는 ceiling까지의 잔차 중 약 60%를 회수한다. 둘 중 <i>L</i><sub>1</sub> 쪽이 더 관련이 깊은데, '
  '§3.4의 분석이 소비하는 것이 argmax가 아니라 marginal이기 때문이다.', body_s)
P('다만 같은 표가 주장의 <b>한계</b>도 정한다. relaxation 값만 쓰는 <b>파라미터 4개 선형 모델</b>이 78.1%에 '
  '도달하며 이는 ceiling에서 9.2%p 이내다. 가용 신호의 대부분은 그 단일 특징에 있고, graph는 그 위에 '
  '유용하지만 지배적이지는 않은 증분을 더한다.', body_s)

H2('4.12 한계')
P('<b>Label 생성이 평가만이 아니라 방법 자체를 제한한다.</b> 정확한 marginal은 <i>S</i>의 열거를 요구하는데 '
  '이 계열에서는 <i>n</i>≈60을 넘으면 완료되지 않는다. transfer는 더 큰 크기에서의 배치를 허용하지만 '
  '거기서의 학습은 허용하지 않는다. solution sampling으로 <i>p<sub>j</sub></i>를 근사하는 것이 자연스러운 '
  '확장이며 미검증이다.', body_s)
P('<b>유용한 크기 창이 양쪽에서 막혀 있다.</b> 그 아래에서는 lattice reduction이 전부를 닫아 branching 품질이 '
  '무의미하고, 그 위(<i>n</i>=100)에서는 constraint solver가 20초 내에 해를 하나도 찾지 못해 모든 전략에서 '
  'search가 실패한다. 우리 시연은 <i>n</i>=60에 있으며 창이 더 확장된다는 것은 보이지 못했다.', body_s)
P('<b>검정력을 제약하는 것은 실행 횟수가 아니라 instance 개수다.</b> 21×60에서의 해결율 이점은 5–0이고 '
  '세 seed에서 동일하지만, 30개 instance의 5–0 분할은 McNemar <i>p</i>=0.0625가 최선이다. '
  '<b>반복으로는 개선되지 않으며 더 큰 평가셋만이 개선한다.</b> 인스턴스 단위 속도 이점은 유의하다'
  '(각 seed 내 <i>p</i>=0.019).', body_s)
P('<b>Multiplicity 통제는 근사적이다.</b> 세 크기에서 |<i>S</i>| 중앙값이 23, 19, 10이며, <i>n</i>=60에서는 '
  '연속한 정수 <i>m</i>에 대해 42→12→2로 계단을 이루므로 더 정밀한 matching이 불가능하다. 평가 instance 30개 중 '
  '3개는 |<i>S</i>|가 미검증이며 그렇게 표시되어 있다. |<i>S</i>|를 맞추는 것은 <i>m/n</i>을 0.40에서 0.35로 '
  '이동시키기도 한다.', body_s)
P('<b>비용 우위는 규모와 baseline 구현 품질에 의존한다.</b> amortization 논변은 점근적으로는 성립하지만 '
  '우리 탐색이 작동하는 크기에서는 가치가 크지 않다: 예측이 가장 중요한 depth에서 warm start한 probing 대비 '
  '<b>1.3~1.7배</b>이며, 그 probing은 건전하기까지 하다. 15~18배에 이르는 것은 <i>n</i>≥100부터인데 '
  '그 구간에서는 우리가 시도한 모든 전략에서 탐색이 실패한다. 즉 <b>논변이 가장 강한 영역을 우리는 시연하지 못한다.</b>', body_s)
P('<b>탐색 구현이 경쟁력이 없다.</b> 21×60에서 CP-SAT가 종단으로 41배 빠르다. 우리 branching이 node를 더 적게 '
  '만들므로 부족분은 node 처리량(78배)이지만, 경쟁력 있는 구현이 달성 가능함을 보인 것은 아니다: '
  'incremental relaxation, clause learning, restart가 모두 없고, 뒤의 둘은 더 큰 크기에서 CP-SAT에게 결정적이다.', body_s)
P('<b>학습 분포와 배포 분포가 거의 겹치지 않는다.</b> 학습 범위 {0,…,12}는 배포를 염두에 두고 고른 것이 아니다. 그것은 '
  '자유 변수 13~25개인 상태를 만드는 반면 21×60 탐색은 5~60개인 상태를 질의하며 질의의 80%가 25 위에 있다. 방법은 이 때문이 '
  '아니라 이에도 <b>불구하고</b> 작동하며(§4.9), 학습 범위를 배포 영역에 맞추는 것은 명백하지만 미검증인 개선이다.', body_s)
P('<b>하이퍼파라미터가 튜닝되지 않았고 validation split이 없다.</b> §4.2 참조. 이 설정은 첫 시도로서 방어 가능하고 test set에 '
  '오염되지 않았으나, 튜닝된 baseline MLP나 튜닝된 depth 범위가 마진을 바꾸는지 묻는 것은 정당한 지적이다.', body_s)
P('<b>단일 instance 계열.</b> 모든 결과가 하나의 응용에서 나온 하나의 생성기에 관한 것이다. amortization 논변이 '
  '다른 constraint 계열로 전이되는지는 미검증이다.', body_s)

# ═══════════════════════ 5. 결론 ═══════════════════════
H1('5. 결론')
P('본 논문은 binary linear system을 위한 학습 기반 branching heuristic과, 그것을 지도학습하는 정확한 절차를 '
  '제시했다. 두 요소가 방법을 떠받친다. BLS를 partial assignment로 conditioning하는 것은 그것을 축소하는 것과 '
  '동일하므로, 평범한 instance로 학습된 하나의 network가 구조 변경 없이 search tree의 모든 node에서 예측한다. '
  '그리고 전수열거가 가능한 크기에서 정확한 conditional posterior는 학습 target을 공급하는 동시에 '
  '<b>어떤 예측이 탐색에 영향을 줄 수 있는지를 식별한다</b>: 살아남은 모든 해가 일치하는 변수만이 backtrack을 '
  '유발할 수 있으며, 그 비중은 root의 6.3%에서 depth 8의 89.0%로 오른다.', body_s)
P('우리 탐색 루프 안에서 측정하면 heuristic은 설계대로 동작한다. 21×60에서 relaxation 기반 branching 대비 '
  'median 3.29배 빠르고 node를 6.2배 적게 전개하며, 세 seed 모두에서 30/30 대 25/30을 해결한다. '
  'solution multiplicity를 통제하면 2.4배 작은 규모에서 학습한 network가 평가 크기에서 학습한 것과 구별되지 '
  '않게 동작하는데, 정확한 지도가 작은 크기에서만 가능하므로 이는 중요한 성질이다.', body_s)
P('<b>결과가 입증하지 못하는 것도 동등하게 명시한다.</b> 이 시스템은 complete solver와 경쟁하지 못한다: '
  'CP-SAT가 시험한 모든 크기에서 41~62배 빠르다. 그 격차의 분해는 변명이 아니라 정보다 — 우리 branching이 '
  'node를 1.6배 적게 만들고 처리량에서 78배 지므로, 부족분은 incremental relaxation·clause learning·restart가 '
  '없는 Python 연구 구현에 있다 — 그러나 그것을 메울 수 있음을 보인 것은 아니다. 비용 논변도 우리가 처음 '
  '부여했던 무게를 지지 못한다. 제대로 warm start한 probing baseline에 대해 우리 instance에서 예측이 중요한 '
  'depth에서의 우위는 1.3~1.7배이고, 그 구간에서 probing은 network가 갖지 못한 <b>건전성</b>을 갖는다. '
  '15~18배에 이르는 것은 <i>n</i>≥100부터이며 그 구간에서는 모든 전략에서 탐색이 실패한다. '
  '이 비교의 이전 버전은 probe마다 LP를 재구성하는 backend를 쓰고 network 자신의 feature 풀이를 타이머에서 '
  '제외해 <b>실제 값이 1에 가까운 규모에서 19~23배를 보고했다.</b> 오류가 두 방법의 계산 내용이 아니라 전적으로 '
  '계측에서 비롯됐으므로 이 정정을 기록해 둔다.', body_s)
P('<b>남는 것, 그리고 우리가 주장하는 것은 방법론적이다.</b> planted-solution instance 계열에서 정확한 '
  'conditional posterior는 계산 가능하며 branching에 대해 잘 정의된 지도 신호다 — 관례적 target인 '
  '심어진 해 하나는 그렇지 않다. search tree 내부에서 뽑은 상태가 root 상태보다 나은 학습 분포이고, '
  'reduction 항등식이 그 생성을 solver 계측이 아니라 <b>대입</b>의 문제로 만든다. 그렇게 얻은 heuristic은 '
  'solution multiplicity를 비슷하게 유지하면 instance 크기를 넘어 전이된다. 이것이 실용적으로 유용해지는지는 '
  '우리가 하지 않은 두 가지에 달려 있다: 전수열거 없이 marginal을 근사해 정확한 지도를 <i>n</i>≈60 너머로 '
  '확장하는 것, 그리고 branching 개선이 시스템 수준에서 보일 만큼 탐색을 잘 구현하는 것이다.', body_s)

# ═══════════════════════ 재현성 ═══════════════════════
H1('재현성')
P('코드는 proposed_src/ 아래에 구성요소별로 정리되어 있고, 공용 모듈은 util/에, 각 디렉토리의 ROLES.txt에 '
  '모든 파일 설명이 있다. 스크립트는 각자의 디렉토리에서 실행한다.', body_s)
E.append(Preformatted(
"""# |S|를 맞춘 instance와 정확한 marginal 생성
cd proposed_src/v15_bayes
python3 v15_gen_by_m.py --m 10 --n 25 --count 2200 --K 20 \\
        --out ../../runs/data/sols_10x25.json

# in-tree 학습 상태 구성 (depth 0-12, instance당 6개)
cd ../v17_conditional
python3 v17_make_conditional_data.py \\
        --sols ../../runs/data/full_10x25_train.json \\
        --max_depth 12 --per_instance 6 \\
        --out ../../runs/data/cond_10x25_train.json

# 학습 (CPU 1코어로 약 20분)
python3 v17_train_conditional.py \\
        --train ../../runs/data/cond_10x25_train.json \\
        --test  ../../runs/data/cond_10x25_test.json \\
        --n_train 10000 --n_test 2000 --epochs 20 --target soft \\
        --out ../../runs/model_n25

# 평가: branching arm x {lattice off, lattice on}
cd ../v22_transfer
python3 v22_cascade_search.py \\
        --data_dir ../../instances/v22test_21x60 \\
        --tag EVAL --time_limit 600 --block 12 --tries 10 \\
        --arms lp model --ckpt ../../runs/model_n25/conditional.pt \\
        --ahl off on --jobs 6 --out_root ../../runs/eval""", code_s))
P('모든 생성기는 seed가 고정되어 있다. 공개 checkpoint는 각 96KB다. instance 데이터와 가중치는 배포 대신 '
  '위 명령으로 재생성하며, 전체 평가 사이클은 v22_transfer/run_v22_cd.sh에 스크립트화되어 있다.', body_s)

# ═══════════════════════ 참고문헌 ═══════════════════════
H1('참고문헌')
for r in [
 'Aardal, K., Hurkens, C. A. J., and Lenstra, A. K. (2000). Solving a system of linear Diophantine equations with lower and upper bounds on the variables. <i>Mathematics of Operations Research</i>, 25(3):427–442.',
 'Aardal, K., Bixby, R. E., Hurkens, C. A. J., Lenstra, A. K., and Smeltink, J. W. (2000). Market split and basis reduction: Towards a solution of the Cornuéjols–Dawande instances. <i>INFORMS Journal on Computing</i>, 12(3):192–202.',
 'Amos, B. (2023). Tutorial on amortized optimization. <i>Foundations and Trends in Machine Learning</i>, 16(5):592–732.',
 'Bengio, Y., Lodi, A., and Prouvost, A. (2021). Machine learning for combinatorial optimization: A methodological tour d’horizon. <i>European Journal of Operational Research</i>, 290(2):405–421.',
 'Chen, Z., Liu, J., Wang, X., Lu, J., and Yin, W. (2023). On representing mixed-integer linear programs by graph neural networks. In <i>International Conference on Learning Representations</i>.',
 'Cornuéjols, G. and Dawande, M. (1999). A class of hard small 0-1 programs. In <i>Integer Programming and Combinatorial Optimization</i>, pages 284–293.',
 'Gasse, M., Chételat, D., Ferroni, N., Charlin, L., and Lodi, A. (2019). Exact combinatorial optimization with graph convolutional neural networks. In <i>Advances in Neural Information Processing Systems</i>.',
 'Gershman, S. J. and Goodman, N. D. (2014). Amortized inference in probabilistic reasoning. In <i>Proceedings of the 36th Annual Meeting of the Cognitive Science Society</i>, pages 517–522.',
 'Khalil, E. B., Le Bodic, P., Song, L., Nemhauser, G., and Dilkina, B. (2016). Learning to branch in mixed integer programming. In <i>AAAI Conference on Artificial Intelligence</i>, pages 724–731.',
 'Lenstra, A. K., Lenstra, H. W., and Lovász, L. (1982). Factoring polynomials with rational coefficients. <i>Mathematische Annalen</i>, 261(4):515–534.',
 'Li, Z., Guo, J., and Si, X. (2023). G4SATBench: Benchmarking and advancing SAT solving with graph neural networks. <i>Transactions on Machine Learning Research</i>.',
 'Marques-Silva, J. P. and Sakallah, K. A. (1999). GRASP: A search algorithm for propositional satisfiability. <i>IEEE Transactions on Computers</i>, 48(5):506–521.',
 'Millidge, B. (2022). Deconfusing direct vs amortized optimization. 비심사 기술 노트, https://www.beren.io/2022-09-25-Deconfusing-direct-vs-amortized-optimization/',
 'Moskewicz, M. W., Madigan, C. F., Zhao, Y., Zhang, L., and Malik, S. (2001). Chaff: Engineering an efficient SAT solver. In <i>Design Automation Conference</i>, pages 530–535.',
 'Ohrimenko, O., Stuckey, P. J., and Codish, M. (2009). Propagation via lazy clause generation. <i>Constraints</i>, 14(3):357–391.',
 'Schnorr, C. P. and Euchner, M. (1994). Lattice basis reduction: Improved practical algorithms and solving subset sum problems. <i>Mathematical Programming</i>, 66(1–3):181–199.',
 'Selsam, D., Lamm, M., Bünz, B., Liang, P., de Moura, L., and Dill, D. L. (2019). Learning a SAT solver from single-bit supervision. In <i>International Conference on Learning Representations</i>.',
 'Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. (2018). Graph attention networks. In <i>International Conference on Learning Representations</i>.',
 'Xu, K., Hu, W., Leskovec, J., and Jegelka, S. (2019). How powerful are graph neural networks? In <i>International Conference on Learning Representations</i>.']:
	P(r, ref_s)

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.0*cm, rightMargin=2.0*cm,
                         topMargin=2.0*cm, bottomMargin=2.0*cm,
                         title='Amortized Deduction for Binary Linear Systems (KR)',
                         author='neuroMCTS project')


def footer(canvas, doc_):
	canvas.saveState()
	canvas.setFont('NotoKR', 8)
	canvas.setFillColor(colors.grey)
	canvas.drawCentredString(A4[0] / 2.0, 1.1 * cm, str(doc_.page))
	canvas.restoreState()


doc.build(E, onFirstPage=footer, onLaterPages=footer)
print(f'saved: {OUT}')
