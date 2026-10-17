#!/usr/bin/env python3
"""Korean edition of docs/tex/261012_paper.tex (single density p=0.3, RQ structure), rendered with reportlab + Noto Sans KR (this machine
has no Korean TeX). Figures are the PNGs produced by v33_paper/make_261008_figures.py, shared
with the English edition. Technical terms stay in English; all text is black; the body uses the Medium
(wght 500) static instance and bold runs the Bold (700) instance, both instantiated from the variable
NotoSansKR.ttf with fontTools (the variable font's default instance is Thin, which made earlier
editions hard to read). Symbols are restricted to glyphs the font covers."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.lib import colors
import os

pdfmetrics.registerFont(TTFont('NotoKR', '/home/kopt/.fonts/NotoSansKR-Medium.ttf'))
pdfmetrics.registerFont(TTFont('NotoKR-B', '/home/kopt/.fonts/NotoSansKR-Bold.ttf'))
addMapping('NotoKR', 0, 0, 'NotoKR'); addMapping('NotoKR', 1, 0, 'NotoKR-B'); addMapping('NotoKR', 0, 1, 'NotoKR'); addMapping('NotoKR', 1, 1, 'NotoKR-B')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '../../docs/tex/261012_paper_kr.pdf')
FIG = os.path.join(HERE, '../../docs/tex/fig/261012')
TAB = os.path.join(HERE, '../../docs/tex/tab/261012')
BLACK = colors.black

title_s = ParagraphStyle('T', fontName='NotoKR-B', fontSize=15, leading=21, alignment=TA_CENTER, spaceAfter=4, textColor=BLACK)
meta_s = ParagraphStyle('M', fontName='NotoKR', fontSize=9.5, leading=13, alignment=TA_CENTER, spaceAfter=12, textColor=BLACK)
h1_s = ParagraphStyle('H1', fontName='NotoKR-B', fontSize=12.8, leading=18, spaceBefore=14, spaceAfter=6, textColor=BLACK)
h2_s = ParagraphStyle('H2', fontName='NotoKR-B', fontSize=11, leading=16, spaceBefore=10, spaceAfter=4, textColor=BLACK)
h3_s = ParagraphStyle('H3', fontName='NotoKR-B', fontSize=10, leading=15, spaceBefore=8, spaceAfter=3, textColor=BLACK)
body_s = ParagraphStyle('B', fontName='NotoKR', fontSize=9.6, leading=14.8, alignment=TA_JUSTIFY, spaceAfter=6, textColor=BLACK)
abs_s = ParagraphStyle('A', fontName='NotoKR', fontSize=9.0, leading=13.8, alignment=TA_JUSTIFY, leftIndent=16, rightIndent=16, spaceAfter=8, textColor=BLACK)
cap_s = ParagraphStyle('C', fontName='NotoKR', fontSize=8.4, leading=12.0, alignment=TA_JUSTIFY, textColor=BLACK, spaceAfter=10)
eq_s = ParagraphStyle('E', fontName='NotoKR', fontSize=10, leading=16, alignment=TA_CENTER, spaceBefore=3, spaceAfter=7, textColor=BLACK)
ref_s = ParagraphStyle('R', fontName='NotoKR', fontSize=8.5, leading=12.2, alignment=TA_JUSTIFY, leftIndent=14, firstLineIndent=-14, spaceAfter=3, textColor=BLACK)
E = []
def P(t, s=body_s): E.append(Paragraph(t, s))
def H1(t): E.append(Paragraph(t, h1_s))
def H2(t): E.append(Paragraph(t, h2_s))
def H3(t): E.append(Paragraph(t, h3_s))
def EQ(t): E.append(Paragraph(t, eq_s))
def IMG(name, width, caption):
	im = Image(os.path.join(FIG, name + '.png')); r = im.imageHeight / im.imageWidth
	im.drawWidth = width; im.drawHeight = width * r
	E.append(KeepTogether([im, Spacer(1, 3), Paragraph(caption, cap_s)]))
def TBL(data, widths, caption, right_from=1, font=8.3):
	cs = ParagraphStyle('c', fontName='NotoKR', fontSize=font, leading=font + 3.2, textColor=BLACK)
	rows = [[Paragraph(c, cs) for c in r] for r in data]
	t = Table(rows, colWidths=widths, repeatRows=1)
	t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.92)), ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black),
	                       ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black), ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
	                       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
	                       ('ALIGN', (right_from, 1), (-1, -1), 'RIGHT')]))
	E.append(KeepTogether([Paragraph(caption, cap_s), t, Spacer(1, 6)]))
W = A4[0] - 4.0 * cm


import re
def tex_rows(name):
	"""Rows of a generated LaTeX table (tab/261008/<name>.tex) as lists of plain-text cells; the LaTeX header block
	(between \\toprule and the first \\midrule) is skipped and group-label rows come back as single-cell rows."""
	out = []; hdr = False
	for l in open(os.path.join(TAB, name + '.tex')):
		l = l.strip()
		if l.startswith('\\toprule'): hdr = True; continue
		if hdr:
			if l.startswith('\\midrule'): hdr = False
			continue
		if l.startswith('\\addlinespace'): out.append(['__gap__']); continue
		if l.startswith('\\multicolumn'):
			m = re.search(r'\\emph\{(.*)\}\}', l); out.append([clean(m.group(1))]); continue
		if ' & ' not in l or (l.startswith('\\') and not l.startswith('\\textbf')): continue
		out.append([clean(c) for c in l.rstrip('\\').split('&')])
	return out
def clean(c):
	c = re.sub(r'\\textbf\{\\underline\{([^}]*)\}\}', r'<b><u>\1</u></b>', c.strip())
	c = re.sub(r'\\hl\{([^}]*)\}', r'<b><u>\1</u></b>', c)
	c = c.replace('\\textbf{', '<b>').replace('\\emph{', '<i>')
	c = c.replace('$\\times$', '×').replace('\\times', '×').replace('\\,', ' ').replace('\\%', '%').replace('\\cdot', '·').replace('---', '—')
	c = re.sub(r'\$([^$]*)\$', r'\1', c); c = re.sub(r'10\^\{(-?\d+)\}', r'10<sup>\1</sup>', c); c = re.sub(r'\{,\}', ',', c)
	c = c.replace('{', '').replace('}', '</b>' if '<b>' in c and '</b>' not in c else '')
	c = c.replace('\\', '')
	return c


# ═══════════════ 제목 · 초록 ═══════════════
P('Binary Linear System을 위한 일반화 가능한 학습된 branching', title_s)
P('Generalisable Learned Branching for Binary Linear Systems<br/>'
  'Gwang-Jong Ko, Taesu Cheong*, In-Chan Choi* · School of Industrial and Management Engineering, Korea University · 2026-10-12 · 한국어판', meta_s)
H1('초록')
P('Binary linear system(BLS)은 0/1 행렬 <i>A</i>에 대해 <i>A</i><b>x</b>=<b>b</b>를 만족하는 vector <b>x</b>∈{0,1}<sup><i>n</i></sup>를 찾는 문제다; 우리는 해가 존재하고 그것을 내놓아야 하는 해 찾기 버전을 다룬다. 목적함수가 없고 linear relaxation은 넓고 무정보하며 '
  'unit propagation은 root 근처에서 작동하지 않으므로, complete depth-first search의 성패는 branching 선택이 결정한다. 학습된 branching은 mixed-integer program과 SAT를 위해 개발됐고, '
  '우리는 그 세 주요 계열이 BLS로 옮겨오면 각각 무언가를 잃음을 보인다: strong branching 모방은 개선할 bound가 없어 LP probing으로 퇴화하고, message-passing marginal은 밀집 factor graph에서 '
  'node당 network forward pass의 두 자릿수 배 비용이 들며, from-scratch 강화학습은 문제가 정확히 제공하는 값 순서를 스스로 발견해야 한다. 본 논문은 두 개의 정확한 신호로 두 단계에 걸쳐 학습한 branching 규칙을 '
  '제시한다. <b>Stage 1</b>은 <b>어느 값</b>을 먼저 시도할지 배운다. 열거 가능한 instance에서 각 자유 변수의 conditional marginal은 정확하고, BLS를 partial assignment로 conditioning하는 것은 '
  '축소하는 것과 같으므로 축소 instance로 학습한 graph network 하나가 모든 node에서 예측한다. <b>Stage 2</b>는 <b>어느 변수</b>로 분기할지 배운다. 결정의 subtree가 전개하는 node 수는 탐색이 '
  '정확히 산출하며, Stage 1 규칙에서 출발해 network를 동결한 채 그 비용으로 actor-critic fine-tuning하면 branching 순서만 바뀐다. 고전 규칙 7종(most-integral·most-fractional LP branching, maxSD '
  'counting, dom/wdeg, impact, 전수 LP probing, uniform)과 참조 신호 2종(정확한 marginal, belief propagation)을 같은 탐색 루프에서, 세 크기 100개 세트로 평가한다. 학습 규칙은 모든 '
  '크기에서 node당 relaxation을 하나만 푸는 규칙(단일 relaxation 규칙) 전부보다 node를 적게 쓰고(가장 강한 규칙 대비 1.5~3.2배, sign test <i>p</i>≤0.035), 28×80에서는 시간도 전부보다 빠르다(2.0~3.1배, '
  '<i>p</i>≤0.002). 모든 node에서 자유 변수당 relaxation을 하나씩 푸는 LP probing만이 node를 더 적게 쓰되 5~8배의 시간이 든다. <i>n</i>≤60에서 학습한 규칙은 <i>n</i>=70, 80에서 우위를 유지하고 그 우위는 크기와 함께 커진다. 값×변수 교차 ablation은 이득이 변수 선택에 '
  '있음을 보인다: 비용 조정 변수에 LP 반올림 값을 붙이면 전체 규칙과 같고, LP 변수에 정확한 marginal 값을 붙이면 28배 나쁘다. SCIP에 branching 규칙으로 꽂으면 policy는 conflict analysis '
  'on/off 모두에서 reliability pseudocost branching보다 node를 1.9~3.8배 적게 쓰고 큰 두 크기에서 시간은 동률이며, presolve와 clause learning을 갖춘 complete solver는 end-to-end에서 여전히 '
  '더 빠르다; 격차는 tree 크기가 아니라 node당 비용에 있다.', abs_s)
P('<b>키워드</b>: Binary linear systems · Learning to branch · Exact conditional marginals · Reinforcement learning · Depth-first search · Graph neural networks', abs_s)

# ═══════════════ 1 ═══════════════
H1('1. 서론')
P('Binary linear system(BLS)은 해를 찾는 문제', body_s)
EQ('find <b>x</b>∈{0,1}<sup><i>n</i></sup> such that <i>A</i><b>x</b>=<b>b</b>, &nbsp;<i>A</i>∈{0,1}<sup><i>m</i>×<i>n</i></sup>, <b>b</b>∈Z<sup><i>m</i></sup><sub>≥0</sub> &nbsp;&nbsp;(1)')
P('이며 해집합을 <i>S</i>로 쓴다. 레이저 기반 비파괴검사에서 이진 점유 영상이 정수 투영 집합을 정확히 재현해야 하는 복원 과제의 일반형이다: 스캔 대상은 존재하므로 해가 존재하고 '
  '유일하지 않은 경우가 흔하며, 복원은 생산 라인의 속도를 따라가야 한다. 본 논문 전체에서 instance에는 항상 해가 있고 과제는 그 해를 내놓는 것이다; 시스템이 infeasible한지 판정하는 것은 범위 밖이다(3.1절). 중요한 instance는 우변이 각 행 범위의 한가운데 근처에 있으며, 크기에 비해 어렵다.', body_s)
P('이런 instance에서는 세 가지 성질이 표준 도구를 무력하게 만든다. 목적함수가 없으므로 branch-and-bound의 bound 기반 가지치기가 기댈 것이 없다. linear relaxation은 항상 '
  'feasible하지만 정보가 적다. relaxation polytope {<b>x</b>∈[0,1]<sup><i>n</i></sup>: <i>A</i><b>x</b>=<b>b</b>}는 차원이 <i>n</i>−rank <i>A</i>≈0.65<i>n</i>이고 ½·<b>1</b> 근처의 점을 포함하는데, 모든 우변이 행 범위의 한가운데에 있기 '
  '때문이다. simplex가 돌려주는 vertex는 분수 좌표가 많아야 <i>m</i>개이지만 나머지 <i>n</i>−<i>m</i>개 좌표가 0과 1 중 어느 쪽에 놓이는지는 <i>S</i>가 아니라 pivot 경로가 정한다: root vertex를 '
  '반올림하면 변수의 59%가 해와 일치하며, 우연은 50%, 3.3절의 Bayes 상한은 69%다. '
  'constraint solver의 주력인 unit propagation은 행이 포화될 때만 발동하는데 root에서는 결코, 그 근처에서도 거의 일어나지 않는다. 이 계열의 고전적 처방인 enumeration을 갖춘 lattice basis '
  'reduction은 건전하고 우리가 다루는 모든 instance를 풀지만 비용이 complete search보다 크기에 따라 빠르게 커진다; lattice가 남기는 instance와 그것이 멈추는 크기에서 complete search를 '
  '결정하는 것은 branching 결정의 품질이고, 그것이 본 논문의 대상이다.', body_s)
H2('1.1 학습된 branching이 여기에 남겨 둔 것')
P('세 계열의 연구가 branching 결정을 학습하며 2절이 검토한다: strong branching 모방, message passing에 의한 근사 marginal, subtree 크기에 '
  '대한 강화학습. 각각은 목적함수가 있는 문제, 희소한 제약 graph, 또는 희소한 보상을 위해 개발됐고, BLS에 적용하면 각각 무언가를 잃는다. 목적함수가 없으면 strong branching은 점수 매길 bound '
  '개선이 없어 node마다 자유 변수당 relaxation 하나를 지불하는 규칙인 LP probing으로 퇴화한다. BLS의 밀집 factor graph에서 belief propagation은 강한 damping과 추정 한 번에 수백 번의 반복이 '
  '필요하고, root 수준의 사용법은 complete search가 모든 node에서 필요로 하는 conditional marginal을 제공하지 않는다. from-scratch 강화학습은 어느 변수로 분기할지를 가르치는 같은 희박한 '
  '신호로부터 어느 값을 먼저 시도할지를 스스로 발견해야 하지만, BLS는 열거할 수 있을 만큼 작은 instance에서 최적의 첫 값을 정확히 제공한다. 빠진 것은 BLS가 제공하는 정확한 신호 — 자유 변수의 '
  'conditional marginal과 branching 결정의 정확한 비용 — 를 취해 값과 변수를 따로 학습하고, 그 신호를 감당할 수 있는 크기에서 한 번 학습해 감당할 수 없는 크기에 배포하는 branching 규칙이다. 본 '
  '논문은 그런 프레임워크를 제시하고(4절) 고전 규칙과 세 전이형에 대해 같은 탐색 루프에서 평가한다.', body_s)
H2('1.2 연구 질문')
P('실험은 네 질문으로 구성되며, 각각 세 크기 100개 세트에서 모든 branching 규칙을 같은 탐색 루프로 돌려 답한다. '
  '<b>RQ1</b> 학습 규칙이 고전 branching 규칙과 전이된 learned-branching baseline 3종을 node와 시간 <b>모두</b>에서 이기는가? 작은 tree가 시간을 사는 것은 node당 비용이 허락할 때뿐이므로 그 '
  '비용이 어디로 가는지도 포함한다. <b>RQ2</b> <i>n</i>≤60에서 학습한 규칙이 더 큰 instance로 일반화되는가, 그 우위는 크기에 따라 줄어드는가 커지는가? <b>RQ3</b> 두 신호는 서로 대체 가능한가, '
  '각 단계는 필요한가? 비용 조정 policy가 탐색에서 실제로 무엇을 바꿨는지를 포함한다. <b>RQ4</b> complete solver 안의 branching 규칙으로서, 그리고 end-to-end로 complete solver와 겨루면 '
  '어떠한가?', body_s)
H2('1.3 기여')
P('(1) branching 결정의 두 부분을 문제의 두 정확한 신호로 학습하는 BLS용 학습 branching 프레임워크 — 값은 열거 가능한 곳에서 학습해 reduction 항등식으로 complete search의 '
  '모든 node로 옮기고, branching 순서는 값 모델을 동결한 채 정확한 subtree 비용으로 학습하므로 이득이 귀속되고 모델이 본 적 없는 크기로 전이된다. (2) RQ1~RQ4에 대한 통제된 답: 고전 규칙 7종·'
  '참조 신호 2종·전이된 학습 baseline 3종을 한 루프에서, 세 크기 100개 세트, 값×변수 교차 ablation, 학습 변형 ablation, 비용 신호가 가르친 것의 반증 구조 분석, SCIP branching plug-in으로서 '
  'reliability pseudocost branching과의 비교, SCIP·CP-SAT·Gurobi·lattice enumeration과의 end-to-end 비교. (3) 그렇게 보고하는 두 부정적 발견: 변수 선택을 학습하고 나면 정확한 marginal 값은 LP '
  '반올림보다 나을 것이 없고, 학습 규칙은 complete solver와의 wall-clock 격차를 닫지 못한다.', body_s)
P('2절은 관련 연구를, 3절은 문제와 reduction 항등식을, 4절은 방법을, 5절은 연구 질문별 실험을, 6절은 한계와 함께 결론을 다룬다.', body_s)
IMG('framework', W, '그림 1. guided depth-first search. node 안에서 1~4단계는 건전하며 변수를 확정하거나 가지를 치는 유일한 단계다; 5단계의 network는 선택만 하며, 값은 Stage 1에서, '
    '변수는 Stage 2에서 온다. lattice reduction은 쓸 때 탐색 앞의 전처리로 돌며 모든 branching 비교에서는 끈다.')

# ═══════════════ 2 ═══════════════
H1('2. 선행연구')
H2('2.1 해집합의 marginal을 branching 신호로')
P('결정에 동의하는 해의 비율로 분기하는 것은 학습 이전부터 있었다: counting-based search(Pesant et al., 2012)는 개별 제약 안에서 계산한 '
  'solution density가 최대인 변수-값으로 분기하고, belief-propagation-guided decimation(Montanari et al., 2007)은 근사 marginal로 가장 편향된 변수를 고정한다. NSNet(Li and Si, '
  '2022)은 SAT 공식의 만족 배정을 열거해 얻은 정확한 marginal로 graph network를 학습하고 추정치를 반올림해 local search의 초기 배정으로 쓴다; MIP-GNN(Khalil et al., 2022), Neural '
  'Diving(Nair et al., 2020), predict-and-search(Han et al., 2023)는 수집한 준최적 해로부터 변수별 bias를 예측해 변수 고정·warm start·탐색 제한에 쓴다. 이들 모두에서 marginal이나 bias는 root에서 한 번 추정되어 변수를 고정하거나 incomplete search를 '
  '시작하는 데 쓰인다; complete search의 모든 node에서 partial assignment에 conditional인 marginal — branching 규칙이 소비하는 것 — 을 제공하는 것은 없다.', body_s)
H2('2.2 Learning to branch')
P('Khalil et al.(2016)과 Gasse et al.(2019)은 strong branching을 모방하며, 후자는 우리도 쓰는 bipartite 변수–제약 graph network를 쓴다; 모방은 순위가 '
  '줄이려는 tree 크기가 아니라 expert의 순위를 target으로 삼는다. tree 크기를 직접 최적화하는 것은 강화학습 연구의 한 줄기다: FMSTS(Etheve et al., 2020)는 node의 subtree 크기를 관측 '
  '가능한 Q-value로 쓰고 depth-first search에서는 모든 subtree를 최소화하면 전체 tree가 최소화됨을 보이며 from scratch로 학습한다; tree MDP(Scavuzzo et al., 2022)는 subtree 단위 '
  'credit assignment를 형식화하고, retro branching(Parsonson et al., 2023)은 회고적으로 추출한 subtree 경로에서 학습하며, SORREL(Feng and Yang, 2025)은 시연으로 초기화한다. '
  'Neuro#(Vaezipoor et al., 2021)은 #SAT solver의 residual formula 위의 graph network를 evolution strategies로 학습해 더 큰 instance로 일반화한다. 강화학습을 정확한 solver 안에서 쓰는 더 넓은 흐름은 Mazyavkina et al.(2021)이 정리했고, operations research 문헌에서도 학습된 policy·value network를 heuristic tree search 안에 '
  '둔 사례(Hottung et al., 2020)와 제약 구조 위의 graph network가 그런 학습 heuristic의 표준 표현이 된 흐름(Smit et al., 2025)이 있다. 이 방법들이 최적화하는 목적의 고전적 '
  '진술이 fail-first 원칙(Haralick and Elliott, 1980)과 impact-based search(Refalo, 2004)다; bound가 아니라 feasibility가 어려운 mixed-integer program에서는 relaxation의 변화를 강제하도록 '
  '설계한 branching 규칙이 bound 중심 규칙보다 integer feasibility에 빨리 이른다는 것을 Pryor and Chinneck(2011)이 보였는데, BLS는 내내 그 영역에 있다. constraint programming에서는 value '
  'ordering을 해로부터(Chu and Stuckey, 2015) 또는 deep Q-learning으로(Cappart et al., 2021) 변수 선택과 함께 또는 별도로 학습했다.', body_s)
H2('2.3 Market split, lattice 방법, complete solver')
P('Cornuéjols and Dawande(1999)의 market split 문제는 (1)과 같은 시스템에서 branch-and-bound의 약점을 드러내는 벤치마크다: '
  '<i>n</i>=10(<i>m</i>−1)개의 이진 변수 위에 등식 행 <i>m</i>개, 계수는 {0,…,99}에서 균등하게 뽑고 각 우변은 행 합의 절반으로 두므로, relaxation polytope가 넓고 vertex가 정수성에서 '
  '멀며 bounding은 거의 가지치지 못한다; branch-and-cut은 <i>m</i>≈7에서 멈춘다. Aardal et al.(2000)은 이 instance를 lattice 위에 재정식화해 풀었다: Aardal–Hurkens–Lenstra '
  'embedding은 해를 lattice의 가장 짧은 vector 중 하나로 만들고, LLL/BKZ reduction(Lenstra et al., 1982; Schnorr and Euchner, 1994)이 그것을 찾으며, 복원한 vector를 검증하므로 '
  '방법은 건전하다. lattice enumeration은 이 벤치마크에서 여전히 최고 수준이다: Wassermann(2025)은 QOBLIB market-split instance를 CPU 하나로 <i>m</i>=14까지 푼다. 우리 '
  'instance는 market split을 어렵게 만드는 재료 — 밀집한 행과 범위 한가운데의 우변 — 를 공유하고, 0/1 계수·심어진 해·최대 80개 변수의 탐색 영역이라는 점에서 다르다; 따라서 lattice '
  'reduction은 BLS solver의 자연스러운 전처리 단계다. clause learning을 갖춘 constraint solver(Marques-Silva and Sakallah, 1999; '
  'Ohrimenko et al., 2009)는 내려가면서 정보를 제조하며 우리가 비교하는 wall-clock 기준이다. 직접 satisfiability 예측은 기록이 약하고(Selsam et al., 2019; Li et al., 2023) '
  'message-passing network는 어떤 feasible/infeasible 쌍을 원리적으로 구분할 수 없다(Chen et al., 2023). 비싼 node별 계산을 대체하는 node별 추정치를 학습하는 것은 amortized inference(Amos, 2023; Gershman '
  'and Goodman, 2014)의 사례다.', body_s)
H2('2.4 이 접근들이 BLS에 남겨 둔 것')
P('우리 설정에 가장 가까운 세 learned-branching 계열은 feasibility 시스템으로 옮겨오면 각각 본질적인 무언가를 잃는다. <i>strong branching 모방</i>(Khalil '
  'et al., 2016; Gasse et al., 2019)은 목적함수를 전제한다: expert는 두 자식의 bound 개선으로 후보를 점수 매기는데 BLS에는 개선할 bound가 없다. feasibility 시스템에서 strong branching이 보유하는 '
  '유일한 정보는 자식의 relaxation이 infeasible한지 여부이므로 expert는 LP probing — 반증할 수 있는 것을 고정하고 LP 값으로 분기하는 규칙 — 으로 퇴화하며 node마다 자유 변수당 relaxation 하나를 '
  '지불한다; 그것을 모방하면 node당 비용 없이 순위를 물려받겠지만 학습된 모방이 expert의 tree를 유지하는지는 열어 두는 질문이며(6절), 우리가 시험하는 것은 expert 그 자체다. <i>message passing에 의한 근사 marginal</i>(Li and Si, 2022; '
  'Montanari et al., 2007)은 BLS에서 잘 정의된다: 각 행은 cardinality factor이고 정확한 factor message를 쓰는 sum-product는 구현 가능하다. 두 가지가 잘못된다. factor graph가 밀집해 모든 행이 '
  '변수의 약 3분의 1에 닿으므로 loopy belief propagation은 강한 damping이 필요하고 수렴하는 곳에서도 수백 번 반복한다; 그리고 NSNet식 사용법은 root에서 한 번 marginal을 추정해 incomplete '
  'local search용 배정을 decode하는 반면 complete search는 모든 node에서 partial assignment에 <b>조건부</b>인 marginal이 필요하다. <i>subtree 크기에 대한 from-scratch 강화학습</i>(Etheve et '
  'al., 2020; Scavuzzo et al., 2022)은 depth-first search에 올바른 credit assignment를 제공한다. 제공하지 않는 것은 값 선택이다: 변수–값 쌍 위의 단일 policy는 어느 변수로 분기할지를 가르치는 같은 '
  '희박한 보상으로 어느 값을 먼저 시도할지도 배워야 하는데, BLS는 열거 가능한 instance에서 최적 값을 conditional marginal로 정확히 제공한다; 또한 학습된 순서가 어디서 출발해야 하는지, tree '
  '크기의 이득을 어떻게 귀속시켜야 하는지도 말하지 않는다. 학습과 무관한 공백이 둘 더 있다. bounded enumeration을 갖춘 lattice reduction(Aardal et al., 2000; Wassermann, 2025)은 건전하며 '
  '우리가 다루는 모든 instance를 풀고, <i>n</i>=60에서는 1초 미만, <i>n</i>=80에서는 분 단위가 든다(5.5절); BLS solver의 자연스러운 전처리 단계이며 branching 규칙의 비교는 그런 단계를 끈 탐색만으로 '
  '이루어져야 한다. 그리고 network에 feasibility 판정을 맡길 수 없으므로(Selsam et al., 2019; Chen et al., 2023) 학습 규칙은 탐색의 순서만 정하고 탐색을 완전하게 남겨야 한다. 4절의 '
  '프레임워크는 이 두 제약 아래 세 학습 공백을 닫도록 만들어졌다: 문제가 제공하는 정확한 conditional marginal로 값을, 탐색이 산출하는 정확한 subtree 비용으로 변수를, 따로 그 순서로 학습하고, '
  '결코 변수를 확정하거나 가지를 치지 않는다.', body_s)

# ═══════════════ 3 ═══════════════
H1('3. 문제 정의와 준비')
H2('3.1 Instance와 범위')
P('우리는 (1)의 feasible instance를 다룬다: 모든 instance는 해를 심어 생성하므로 <i>S</i>≠∅이며 infeasibility 판별은 본 논문의 범위 밖이다. 구성은 2절의 market-split 방식에 0/1 계수를 '
  '적용한 것이다. 생성기는 각 원소가 확률 0.3으로 1인 무작위 <i>A</i>를 뽑고 정확히 <i>n</i>/2개가 1인 무작위 <b>x</b>*를 심어 <b>b</b>=<i>A</i><b>x</b>*로 두므로, 모든 우변은 행 범위의 '
  '한가운데, 즉 0/1 완성의 수가 가장 많고 propagation이 발동할 수 없는 곳에 놓인다. <i>A</i>마다 후보 planting 20개 중 relaxation polytope가 가장 넓은 것(무작위 선형 목적 4개가 돌려주는 '
  'vertex 간 평균 거리로 측정)을 남기므로 채택된 instance의 relaxation은 생성기가 만들 수 있는 만큼 무정보하다. 유일성은 강제하지 않는다; 열거가 완료되는 크기에서 해의 수 중앙값은 '
  '<i>n</i>=25, 50, 60에서 64, 112, 113이다.', body_s)
H2('3.2 Conditioning은 reduction이다')
IMG('reduction', W, '그림 2. 명제 1과 그것이 유도하는 분할. <b>x</b><sub><i>F</i></sub>=<b>v</b>인 node는 더 작은 BLS <i>A<sub>K</sub></i><b>y</b>=<b>b</b>−<i>A<sub>F</sub></i><b>v</b>이고 그 '
    'conditional marginal은 그 instance의 통상적 marginal이다. 살아남은 모든 해가 일치하는 변수가 FORCED이며, 그 반대로 분기할 때만 backtrack 비용이 든다.')
P('(1) 위의 depth-first search의 node는 고정 변수 집합 <i>F</i> 위의 partial assignment <b>x</b><sub><i>F</i></sub>=<b>v</b>이고 <i>K</i>는 자유 변수다. '
  '<i>S</i>(<i>F</i>,<b>v</b>)={<b>x</b>∈<i>S</i>: <b>x</b><sub><i>F</i></sub>=<b>v</b>}를 node와 일관된 해집합이라 하고', body_s)
EQ('<i>p<sub>j</sub></i> = Pr[<i>x<sub>j</sub></i>=1 | <i>A</i>, <b>b</b>, <b>x</b><sub><i>F</i></sub>=<b>v</b>] = |{<b>x</b>∈<i>S</i>(<i>F</i>,<b>v</b>): <i>x<sub>j</sub></i>=1}| / |<i>S</i>(<i>F</i>,<b>v</b>)|, &nbsp;<i>j</i>∈<i>K</i> &nbsp;&nbsp;(2)')
P('를 conditional marginal, 즉 <i>S</i> 위의 균등 측도로 정의한 marginal이라 하자.', body_s)
P('<b>명제 1.</b> <i>A<sub>F</sub></i>, <i>A<sub>K</sub></i>를 <i>F</i>, <i>K</i>로 인덱싱된 열 부분행렬이라 하면 <b>x</b>→<b>x</b><sub><i>K</i></sub>는 <i>S</i>(<i>F</i>,<b>v</b>)에서 '
  '{<b>y</b>∈{0,1}<sup>|<i>K</i>|</sup>: <i>A<sub>K</sub></i><b>y</b>=<b>b</b>−<i>A<sub>F</sub></i><b>v</b>}로의 전단사다. 따라서 원본 instance의 conditional marginal은 축소 instance의 '
  '통상적 marginal이다. <i>증명.</i> <i>A</i><b>x</b>=<i>A<sub>F</sub></i><b>x</b><sub><i>F</i></sub>+<i>A<sub>K</sub></i><b>x</b><sub><i>K</i></sub>이고 <b>x</b><sub><i>F</i></sub>=<b>v</b>로 고정하면 '
  '<i>A</i><b>x</b>=<b>b</b> ⇔ <i>A<sub>K</sub></i><b>x</b><sub><i>K</i></sub>=<b>b</b>−<i>A<sub>F</sub></i><b>v</b>이며 <b>x</b><sub><i>F</i></sub>가 결정되어 있으므로 전단사다. marginal 진술은 '
  '개수를 세면 따른다. □', body_s)
P('두 귀결이 있다(그림 2). 첫째, 하나의 architecture·라벨 생성기·학습 루프가 모든 node를 담당한다: depth-<i>d</i> 학습 상태는 <i>n</i>−<i>d</i>개 변수의 instance로 저장되고, 작은 '
  'instance로 학습한 network는 더 큰 instance의 탐색이 만들어내는 축소 instance에서 평가된다. 둘째, <i>p<sub>j</sub></i>∈{0,1}은 살아남은 모든 해가 <i>x<sub>j</sub></i>에 일치한다는 뜻이고 '
  '반대로 분기하면 subtree가 빈다. 이런 변수를 FORCED, 나머지를 FREE라 부른다: FREE 변수는 어느 branch로도 해에 도달하므로 첫 값이 틀려도 해를 포함한 subtree를 탐색하는 것 이상의 '
  '비용이 없다. FORCED 오류만이 backtrack 비용을 낳는다.', body_s)
H2('3.3 Node의 모습과 정확도 상한')
P('작은 instance에서는 <i>S</i>(<i>F</i>,<b>v</b>)를 열거할 수 있으므로 해의 어떤 predictor도 넘을 수 없는 Bayes 최적 변수별 정확도를 계산할 수 있다:', body_s)
EQ('ceil(<i>F</i>,<b>v</b>) = (1/|<i>K</i>|) Σ<sub><i>j</i>∈<i>K</i></sub> max(<i>p<sub>j</sub></i>, 1−<i>p<sub>j</sub></i>) &nbsp;&nbsp;(3)')
P('이는 살아남은 해의 다수 값으로 답할 때 달성된다. 10×25 instance의 held-out in-tree 상태에서 살아남은 해집합은 root의 중앙값 74에서 depth 10이면 하나로 줄고, 자유 변수 중 '
  'FORCED 비중은 root의 3%에서 depth 5의 41%, depth 8의 73%, depth 12의 94%로 오른다. root에서는 거의 어떤 결정도 비용을 낳을 수 없고 상한 (3)은 FREE 변수에 지배된다; depth 6부터는 '
  '대부분의 결정이 비용을 낳을 수 있다. 그림 3은 그 상한에 대한 Stage 1 network를 depth별로 보여준다.', body_s)
IMG('prediction', 0.62 * W, '그림 3. 10×25 instance held-out in-tree 상태에서 정확한 marginal에 대한 예측, depth별: Stage 1 network와 relaxation rounding의 변수별 정확도를 '
    'Bayes 상한 (3)과 비교.')

# ═══════════════ 4 ═══════════════
H1('4. 제안 방법')
H2('4.1 개요')
P('solver는 학습된 network가 branching 결정을 안내하는 depth-first search다(그림 1); 학습된 것은 결코 변수를 확정하거나 가지를 치지 않으므로 탐색은 완전하게 유지된다. network(그림 4)는 두 '
  '단계로 학습한다(그림 5): Stage 1은 열거한 instance의 정확한 conditional marginal에 marginal head를 맞추고, Stage 2는 동결된 trunk 위에 policy head를 더해 모든 branching 결정이 치르는 '
  '정확한 node 수로 fine-tuning한다. BKZ 뒤 bounded enumeration을 붙인 Aardal–Hurkens–Lenstra embedding의 lattice reduction(Aardal et al., 2000)은 우리가 다루는 모든 instance를 푸는 건전한 '
  '전처리 단계이며 작은 크기에서는 싸다(5.5절); 방법의 일부가 아니며 측정되는 차이가 branching 결정뿐이도록 모든 branching 규칙 비교에서 끈다.', body_s)
H2('4.2 Guided depth-first search')
P('모든 instance는 알고리즘 1의 depth-first 절차로 처음부터 탐색한다. node는 partial assignment이며 명제 1에 따라 축소 instance '
  '(<i>A</i>′, <b>b</b>′)=(<i>A<sub>K</sub></i>, <b>b</b>−<i>A<sub>F</sub></i><b>v</b>)로 다루고, <i>r<sub>i</sub></i>는 <i>A</i>′의 행 <i>i</i>에 남은 자유 변수 수다.', body_s)
TBL([['단계', '알고리즘 1: guided depth-first search의 node 하나'],
     ['1', '<b>Reduce.</b> (<i>F</i>,<b>v</b>)를 pop; <i>A</i>′=<i>A<sub>K</sub></i>, <b>b</b>′=<b>b</b>−<i>A<sub>F</sub></i><b>v</b>를 만든다. <i>r<sub>i</sub></i>=0인 행은 버린다; 행이 남지 않으면 <i>A</i><b>x</b>=<b>b</b>를 검증하고 <b>x</b>를 반환한다.'],
     ['2', '<b>Bound check.</b> 어떤 행에서 <i>b</i>′<sub><i>i</i></sub>&lt;0 또는 <i>b</i>′<sub><i>i</i></sub>&gt;<i>r<sub>i</sub></i>이면 node를 버린다.'],
     ['3', '<b>Propagate.</b> <i>b</i>′<sub><i>i</i></sub>=0(그 행의 자유 변수를 0으로 고정) 또는 <i>b</i>′<sub><i>i</i></sub>=<i>r<sub>i</sub></i>(1로 고정)인 행이 있는 동안 고정을 적용한다; 모순이면 node를 버리고, 변수가 고정됐으면 축소 node를 push하고 돌아간다.'],
     ['4', '<b>Relax.</b> <i>F</i>의 bound를 <b>v</b>로 조인 단일 warm-start relaxation {<i>A</i><b>x</b>=<b>b</b>, 0≤<b>x</b>≤1}을 재최적화한다(이전 basis에서 dual simplex, 0.08 ms). infeasible이면 node를 버리고, 아니면 vertex <b>x</b><sup>LP</sup><sub><i>K</i></sub>를 읽는다.'],
     ['5', '<b>Predict.</b> (<i>A</i>′, <b>b</b>′, <b>x</b><sup>LP</sup><sub><i>K</i></sub>)의 bipartite graph 위에서 network를 한 번 forward하여 <i>j</i>∈<i>K</i>에 대한 p^<sub><i>j</i></sub>와 <i>g<sub>j</sub></i>를 얻는다(4.5절의 dense 경로에서 0.3~0.5 ms). <i>j</i>*=argmax<sub><i>j</i></sub> <i>g<sub>j</sub></i>, <i>v</i>=1[p^<sub><i>j</i>*</sub>≥1/2]를 고른다.'],
     ['6', '<b>Branch.</b> 자식 <i>x</i><sub><i>j</i>*</sub>=1−<i>v</i>를 먼저, 자식 <i>x</i><sub><i>j</i>*</sub>=<i>v</i>를 나중에 push하여 <i>v</i>를 먼저, backtrack 시 1−<i>v</i>를 탐색한다.']],
    [1.2*cm, W - 1.2*cm],
    '알고리즘 1. guided depth-first search의 node 하나. 1~4단계는 건전하며 변수를 확정하거나 가지를 치는 유일한 단계다; 학습 컴포넌트는 5단계에만 들어간다.', right_from=9, font=8.0)
P('세 가지 설계 선택이 이후에 중요하다. 0/1 행에서 propagation은 "이 행의 자유 변수 <i>r<sub>i</sub></i>개 중 정확히 <i>b</i>′<sub><i>i</i></sub>개가 1"이라 읽히며 '
  '<i>b</i>′<sub><i>i</i></sub>∈{0, <i>r<sub>i</sub></i>}일 때만 변수를 고정할 수 있으므로 우리 instance의 root 근처에서는 작동하지 않는다. relaxation은 node마다 다시 만들지 않는다: '
  '원래 <i>n</i>개 변수 위의 linear program 하나를 탐색 내내 유지하고 node를 bound 변경 집합으로 표현하므로, 자식이든 backtrack 후의 형제든 연속한 node는 bound 몇 개만 다르고 '
  'dual simplex는 부모 basis에서 재개한다; 이는 재구성보다 50배 싸며 Python 루프가 초당 약 1,000 node를 처리하게 하는 요인이다. 그 infeasibility 판정은 행들의 선형결합을 보므로 '
  'propagation보다 엄격히 강하고 network 앞의 마지막 건전 단계다. 그것이 돌려주는 vertex <b>x</b><sup>LP</sup>가 network의 주 입력 특징이다.', body_s)
H2('4.3 Stage 1: 정확한 conditional marginal을 target으로')
IMG('network', W, '그림 4. network. 축소 instance가 변수당·행당 특징 3개를 지닌 bipartite graph로 들어가고, graph-attention 4 라운드의 trunk가 변수와 행마다 embedding을 만들며, '
    'head 셋이 그것을 읽는다. marginal head는 Stage 1에서, policy head는 동결된 trunk 위에서 Stage 2에서 학습하고, value head는 Stage 2의 critic으로만 쓰인다.')
IMG('training', W, '그림 5. 두 학습 단계. Stage 1은 열거 가능한 instance의 정확한 conditional marginal에서 <b>어느 값</b>을 배우고, Stage 2는 탐색이 산출하는 모든 결정의 정확한 '
    'node 수에서 <b>어느 변수</b>를 배운다. Stage 1 network는 동결된다.')
H3('Network')
P('network <i>f</i><sub>θ</sub>(그림 4)는 graph attention(Velickovic et al., 2018)을 쓰는 bipartite 변수–제약 graph network로 Gasse et al.(2019)의 architecture 계열이다. '
  '변수 node는 특징 3개(relaxation 값 <i>x</i><sup>LP</sup><sub><i>j</i></sub>, 열 밀도, fractionality |<i>x</i><sup>LP</sup><sub><i>j</i></sub>−round(<i>x</i><sup>LP</sup><sub><i>j</i></sub>)|), '
  '제약 node는 3개(우변 <i>b</i>′<sub><i>i</i></sub>/<i>r<sub>i</sub></i>, 밀도 <i>r<sub>i</sub></i>/|<i>K</i>|, relaxation 잔차 (<i>b</i>′<sub><i>i</i></sub>−(<i>A</i>′<b>x</b><sup>LP</sup>)<sub><i>i</i></sub>)/<i>r<sub>i</sub></i>)를 '
  '지니고, <i>a</i>′<sub><i>ij</i></sub>=1인 곳에 간선이 있다. linear layer가 둘을 폭 64로 올리고, attention 4 라운드가 변수→제약, 제약→변수 message를 번갈아 주고받으며(residual 연결, '
  'layer normalization), 2층 marginal head가 각 변수 embedding <b>h</b><sub><i>j</i></sub>를 logit 하나로 사상한다: p^<sub><i>j</i></sub>=σ(<i>m<sub>j</sub></i>); 파라미터는 총 21,761개. '
  'Stage 2에서 더하는 head 둘은 4.4절에 있다. architecture는 표준이고 기여는 target이다.', body_s)
H3('Target과 학습 상태')
P('열거 가능한 instance(10×25)에서 <i>S</i>는 constraint solver의 all-solutions mode로 얻고 <i>p<sub>j</sub></i>는 개수로 계산한다. 소진까지 완료된 열거만 '
  '인정한다; 해를 이미 모은 채 시간제한에 걸린 열거는 잘린 <i>S</i>를 대입해 모든 라벨을 편향시킨다. 학습 상태는 도달 가능한 in-tree 상태다: depth <i>d</i>∈{0,…,12}를 균등하게 뽑고 '
  'LP 확신도 순서의 prefix를 무작위로 고른 실제 해의 값으로 고정한 뒤 축소 instance를 다시 열거해 자유 변수의 <i>p<sub>j</sub></i>를 얻는다. network는 soft target에 대한 cross-entropy', body_s)
EQ('<i>L</i><sub>1</sub> = −(1/|<i>K</i>|) Σ<sub><i>j</i>∈<i>K</i></sub> [ <i>p<sub>j</sub></i> log p^<sub><i>j</i></sub> + (1−<i>p<sub>j</sub></i>) log(1−p^<sub><i>j</i></sub>) ] &nbsp;&nbsp;(4)')
P('로 학습하며 이는 p^=<i>p</i>에서 정확히 최소화된다; <i>p<sub>j</sub></i> 자리에 심어진 해를 넣는 통상적 선택은 |<i>S</i>|&gt;1이면 잡음 섞인 대리 target이다.', body_s)
H3('Branching 규칙')
P('node에서 network를 한 번 평가하고 Stage 1 규칙은 가장 확신하는 marginal의 다수 값으로 분기한다:', body_s)
EQ('<i>j</i>* = argmax<sub><i>j</i>∈<i>K</i></sub> |p^<sub><i>j</i></sub> − 1/2|, &nbsp;&nbsp; <i>v</i> = 1[p^<sub><i>j</i>*</sub> ≥ 1/2] &nbsp;&nbsp;(5)')
H2('4.4 Stage 2: 정확한 subtree node 수를 목적함수로')
P('규칙 (5)는 살아남은 해의 더 큰 몫이 취하는 값을 먼저 시도하여 첫 자식이 marginal이 약속하는 만큼의 해를 보존하게 한다. 첫 자식이 해를 포함하지 않을 때 탐색이 얼마를 지불하는지, 자식이 얼마나 작아지는지는 말하지 않는다; 둘 다 subtree의 성질이고 '
  'depth-first search는 그것을 정확히 산출한다. node <i>u</i>에 대해 <i>c</i>(<i>u</i>)를 <i>u</i> 아래에서 해를 찾거나 subtree를 소진할 때까지 전개한 node 수라 하자; 탐색은 <i>u</i>에서 '
  'backtrack해 나올 때 그것을 돌려준다. <i>c</i>(<i>u</i>)를 <i>u</i>에서 내린 branching 결정의 비용으로 쓴다; <i>c</i>(root)는 전체 탐색이 해에 도달하기까지 전개하는 node 수이며 실험 '
  '전체에서 보고하는 tree 크기 지표다. Stage 2는 −log <i>c</i>(<i>u</i>)에 대해 branching 순서를 fine-tuning한다.', body_s)
H3('결정 과정')
P('상태는 propagation 후의 축소 instance, 행동은 자유 변수 <i>j</i>, 첫 값은 동결된 Stage 1 network의 1[p^<sub><i>j</i></sub>≥1/2]로 유지한다. 각 결정이 자기 subtree를 '
  '가지므로 credit은 episode가 아니라 결정 단위로 배정된다 — sparse한 해결/미해결 보상에는 없는 성질이며 문제를 well-posed하게 만드는 성질이다(Etheve et al., 2020).', body_s)
H3('Policy와 value head')
P('<i>f</i><sub>θ</sub>의 동결된 trunk 위에 head 둘을 더한다(그림 4). policy logit은', body_s)
EQ('<i>g<sub>j</sub></i>(<i>s</i>) = log |p^<sub><i>j</i></sub> − 1/2| + <i>h</i><sub>φ</sub>(<b>h</b><sub><i>j</i></sub>) &nbsp;&nbsp;(6)')
P('이며 <i>h</i><sub>φ</sub>는 마지막 층을 0으로 초기화한 2층 network라서 π(<i>j</i>|<i>s</i>) ∝ exp(<i>g<sub>j</sub></i>/τ)는 규칙 (5)를 부드럽게 한 것에서 출발하고 그 argmax는 정확히 '
  '그 규칙이다. value head <i>V</i><sub>ψ</sub>(<i>s</i>)는 mean·max pooling한 변수·제약 embedding과 log|<i>K</i>|, log <i>m</i>으로 log <i>c</i>를 예측한다.', body_s)
H3('갱신')
P('매 epoch 자유 변수 45~60개인 학습 상태 400개를 뽑아 표집 policy(τ=1)를 해까지 전개하고 모든 결정을 정확한 비용과 함께 기록한다(epoch당 약 21,000 결정). epoch 안에서 '
  '표준화한 advantage α(<i>u</i>)=<i>V</i><sub>ψ</sub>(<i>s<sub>u</sub></i>)−log <i>c</i>(<i>u</i>)로', body_s)
EQ('<i>L</i><sub>2</sub> = Σ<sub><i>u</i></sub> [ −α(<i>u</i>) log π(<i>j<sub>u</sub></i>|<i>s<sub>u</sub></i>) − λ H(π(·|<i>s<sub>u</sub></i>)) + (<i>V</i><sub>ψ</sub>(<i>s<sub>u</sub></i>) − log <i>c</i>(<i>u</i>))<sup>2</sup> ], &nbsp;λ=0.01 &nbsp;&nbsp;(7)')
P('이는 Monte-Carlo return을 쓰는 on-policy actor-critic 갱신이며(Williams, 1992) return이 정확하므로 bootstrapping이 필요 없다. φ, ψ만 학습한다; θ는 고정이므로 값을 고르고 '
  'FORCED/FREE 분할을 정의하는 marginal은 변하지 않고 어떤 개선도 branching 순서만으로 귀속된다. critic은 규칙 (5) 하에서 기록한 결정으로 사전학습하며 그 held-out 순위 상관이 actor '
  '갱신의 관문이다 — 비용 순위를 못 매기는 baseline은 advantage를 잡음으로 만든다. epoch는 학습 풀의 held-out 상태 아래 총 node 수로 고르며 평가 instance는 결코 쓰지 않는다. '
  '배포 시 policy는 greedy다: <i>j</i>*=argmax<sub><i>j</i></sub> <i>g<sub>j</sub></i>(<i>s</i>).', body_s)
H2('4.5 Node당 비용')
P('자유 변수 |<i>K</i>|개인 node에서 guided search는 warm-start relaxation 하나와 forward pass 하나를 지불한다. forward pass는 dense adjacency tensor 위에서 TorchScript로 돈다(graph가 최대 '
  '변수 80개·행 28개라 sparse message passing은 얻는 것이 없다); 단일 thread로 자유 변수 수에 따라 0.3~0.5 ms이며 Stage 2의 head는 측정 가능한 비용을 더하지 않는다. 단일 relaxation 고전 규칙은 '
  'relaxation만 지불하므로 학습 규칙의 node는 1.3~1.5배 비싸며, 그 비율은 relaxation이 크기와 함께 커짐에 따라 줄어든다(5.2절, 표 3). 같은 FORCED 변수를 탐지하는 건전 절차인 LP '
  'probing(<i>x<sub>j</sub></i>를 각 값으로 고정해 relaxation이 infeasible해지는 쪽을 가지치기)은 node마다 relaxation |<i>K</i>|개를 지불한다.', body_s)

def TT(name, header, widths, caption, font=7.2, right_from=1):
	"""Korean table from a generated LaTeX table: Korean header row, group-label rows spanning the first cell."""
	n = len(header); rows = [header]
	for r in tex_rows(name):
		if r == ['__gap__']: rows.append([''] * n); continue
		rows.append(([r[0]] + [''] * (n - 1)) if len(r) == 1 else (r + [''] * (n - len(r)))[:n])
	TBL(rows, widths, caption, right_from=right_from, font=font)

# ═══════════════ 5 ═══════════════
H1('5. 실험')
P('실험은 1절의 네 질문에 설정 다음에 차례로 답한다(5.2~5.5절). ', body_s)
H2('5.1 설정')
H3('Instance')
P('instance는 3.1절대로 생성한다. 크기는 10×25, 18×50, 21×60, 24×70, 28×80이고 제약 밀도 <i>m</i>/<i>n</i>≈0.35를 유지한다; 평가 세트는 작은 두 크기에서 30개, 큰 세 크기에서 '
  '각 100개이며 생성기 seed로 학습 풀과 분리된다. Stage 1은 별도의 10×25 instance 1,600개의 in-tree 상태로, Stage 2는 별도의 21×60 instance 500개의 상태로 학습한다. <i>n</i>=100에서는 '
  'CP-SAT로 20 s 안에 어떤 방법도 instance를 풀지 못하므로 <i>n</i>=80이 마지막 크기다.', body_s)
H3('학습')
P('Stage 1: 소진까지 열거한 10×25 instance 1,600개, instance당 depth 0~12의 in-tree 상태 8개, 학습 상태 10,000개·테스트 상태 2,000개; Adam, 학습률 10<sup>−3</sup>, 20 epoch, batch 1, '
  'gradient clipping 1.0, CPU 코어 하나로 약 20분, 튜닝한 hyperparameter 없음, validation split 없음. 테스트 상태의 FORCED 변수 정확도는 79.1%로 Bayes 상한 83.3%, relaxation rounding '
  '76.0~77.5%와 비교된다. Stage 2: 자유 변수 45~60개인 21×60 상태에서 Stage 1 규칙으로 기록한 결정으로 critic을 사전학습하고(log <i>c</i>와의 held-out Spearman 상관 0.81, 상태 크기만 아는 '
  'baseline 0.75), 갱신 (7)을 6 epoch; epoch는 held-out 학습 상태로 고른다(epoch 6, 그 상태 아래 node 수가 Stage 1 규칙의 0.36).', body_s)
TBL([['그룹', '규칙', '분기 변수', '첫 값'],
     ['고전', 'LP rule (most-integral)', 'argmax<sub><i>j</i></sub> |<i>x</i><sup>LP</sup><sub><i>j</i></sub>−1/2|', 'round(<i>x</i><sup>LP</sup><sub><i>j</i></sub>)'],
     ['', 'most-fractional', 'argmin<sub><i>j</i></sub> |<i>x</i><sup>LP</sup><sub><i>j</i></sub>−1/2|', 'round(<i>x</i><sup>LP</sup><sub><i>j</i></sub>)'],
     ['', 'maxSD (Pesant et al., 2012)', '행 안 solution density 최대', '밀한 값'],
     ['', 'dom/wdeg (Boussemart et al., 2004)', '가중 실패 차수 최대', 'round(<i>x</i><sup>LP</sup><sub><i>j</i></sub>)'],
     ['', 'impact (Refalo, 2004)', '평균 탐색공간 축소 최대', 'impact가 작은 값'],
     ['', 'LP probing (= BLS의 strong branching)', '반증 가능한 값을 고정한 뒤 LP rule', 'round(<i>x</i><sup>LP</sup><sub><i>j</i></sub>)'],
     ['', 'uniform', '균등 무작위 자유 변수', '균등'],
     ['참조', 'oracle marginals', 'argmax<sub><i>j</i></sub> |<i>p<sub>j</sub></i>−1/2|, <i>p<sub>j</sub></i>는 열거로 정확', '1[<i>p<sub>j</sub></i>≥1/2]'],
     ['', 'BP marginals (NSNet-based)', '위와 같되 damped sum-product의 p~<sub><i>j</i></sub>', '1[p~<sub><i>j</i></sub>≥1/2]'],
     ['학습', 'Stage 1', '가장 확신하는 marginal, argmax<sub><i>j</i></sub> |p^<sub><i>j</i></sub>−1/2|', '1[p^<sub><i>j</i></sub>≥1/2]'],
     ['', 'Stage 1+2', '비용 조정 점수, argmax<sub><i>j</i></sub> <i>g<sub>j</sub></i>', '1[p^<sub><i>j</i></sub>≥1/2]']],
    [1.2*cm, 5.2*cm, 6.4*cm, W - 12.8*cm],
    '표 1. 비교하는 branching 규칙. 모두 같은 depth-first 루프(알고리즘 1: node당 warm-start relaxation 하나, 같은 propagation, 같은 예산)에서 돌고 5단계만 다르다. 단일 relaxation 고전 규칙은 '
    'node당 relaxation 하나, LP probing은 자유 변수당 하나를 지불한다; 참조 신호는 배포 불가능하며 값 신호가 할 수 있는 것의 상한이다.', right_from=9, font=7.4)
H3('규칙')
P('표 1이 규칙을 나열한다. node당 relaxation을 하나만 푸는 규칙을 <i>단일 relaxation</i> 규칙이라 부른다; 표의 고전 규칙은 모든 node에서 자유 변수당 한 번씩 relaxation을 다시 푸는 LP probing을 빼고 전부 여기에 든다. LP 규칙 둘은 most-integral과 most-fractional vertex 규칙이다; maxSD는 counting-based search로 0/1 행의 solution density가 정확히 '
  '<i>b</i>′<sub><i>i</i></sub>/<i>r<sub>i</sub></i>라 계산 비용이 없다; dom/wdeg와 impact는 constraint programming의 표준 적응형 규칙 둘로 우리 루프의 실패·축소 통계로 구현했다; LP probing은 '
  '목적함수 없는 strong branching이 환원되는 것이다; uniform 규칙은 instance가 어렵다는 점검이다. oracle은 모든 node에서 <i>S</i>(<i>F</i>,<b>v</b>)를 열거하며 감당 가능한 곳에서만 돌린다; '
  'BP 규칙은 정확한 cardinality-factor message, damping 0.9, 최대 300회 반복의 sum-product를 돌린다. 2절의 학습 baseline은 다음과 같이 나타난다: LP probing은 strong branching 모방을(expert '
  '그 자체다), BP는 message-passing marginal을, 5.4절에서 무작위 head와 LP 규칙에서 출발한 비용 조정 policy는 from scratch와 고전 규칙에서 출발하는 강화학습을 대표한다.', body_s)
H3('프로토콜과 통계')
P('instance마다 두 지표를 기록한다: 해에 도달하기까지 전개한 node 수(4.4절의 <i>c</i>(root); branching 품질을 재며 탐색이 relaxation vertex가 주어지면 결정적이므로 '
  'seed 불변)와 단일 thread wall-clock 시간(node당 비용도 반영). 예산은 다섯 크기에 대해 instance당 60, 300, 600, 1,200, 1,200 s이며 프로세스 종료로 강제한다; 20코어 Intel Core Ultra 7 265KF 머신(메모리 32 GB)에서 instance 6개를 '
  '동시에 돌리고 GPU는 쓰지 않으며 학습 arm은 4.5절의 dense TorchScript 경로를 쓴다. 루프는 Python 3.10과 NumPy 2.2, relaxation은 LP 엔진으로 쓴 Gurobi 13.0, network는 PyTorch 2.10이고, solver 비교는 '
  'PySCIPOpt 6.1을 통한 SCIP 10.0, OR-Tools CP-SAT 9.15, Gurobi 13.0, fpylll 0.6을 쓴다. 비교는 instance별로 짝짓는다: 두 규칙이 모두 푼 instance에서 instance별 비율의 중앙값, 그 중앙값의 '
  '95% bootstrap 구간(instance 1,000회 재표집), 각 규칙이 이긴 instance 수, 동률을 제외한 양측 exact sign test를 보고하며, PAR-2(미해결 instance를 예산의 두 배로 센 평균 시간)가 해결률과 시간을 '
  '함께 요약한다. node·시간 원값의 중앙값은 방향을 보여주기 위한 것이고 검정은 그 위에서 하지 않는다. 다중 비교 보정은 하지 않는다; 논문 전체에 짝 검정이 약 60개 있으므로 0.05 근처의 개별 <i>p</i>값은 시사적인 것으로 읽어야 하고 결론은 <i>p</i>&lt;10<sup>−3</sup>인 비교에 기댄다. 학습 arm의 타이밍은 부록 D에서 평가 seed 두 개로 더 반복한다.', body_s)

H2('5.2 RQ1: 고전 branching과 전이된 learned branching 대비')
TT('rq1_main', ['규칙', '해결', 'node 중앙값', '초 중앙값', 'PAR-2 s', 'node, 규칙 / Stage 1+2', '시간, 규칙 / Stage 1+2'], [3.2*cm, 1.4*cm, 1.8*cm, 1.4*cm, 1.4*cm, 3.9*cm, 3.9*cm],
   '표 2. 크기당 100개 instance에서 같은 탐색 루프의 branching 규칙. 중앙값은 해결 instance 기준; 마지막 두 열은 공통 해결 instance에서 규칙 대 Stage 1+2의 instance별 비율 중앙값, '
   '95% bootstrap 구간, 양측 sign test <i>p</i>값(동률 제외). 비율이 1보다 크면 규칙이 학습 규칙보다 node를 더 쓰거나 더 오래 걸린다. PAR-2는 미해결 instance를 예산의 두 배로 센다.', font=6.8)
IMG('rq1_nodes', W, '그림 6. 세 100개 세트에서 해에 도달하기까지의 node 수, 규칙마다 instance를 따로 정렬; 다른 곡선 아래의 곡선은 모든 분위에서 더 작은 tree다. node마다 자유 변수당 '
    'relaxation 하나를 쓰는 LP probing만이 학습 규칙 아래에 있다.')
H3('Tree 크기')
P('표 2와 그림 6이 주 결과다. 세 크기 각각에서 학습 규칙은 단일 relaxation 고전 규칙 전부보다 node를 적게 쓰고, 짝 검정은 그 전부에 대해 유의하다. 크기별 최강의 싼 규칙(21×60과 '
  '24×70에서 dom/wdeg, 28×80에서 maxSD) 대비 instance별 비율 중앙값은 1.52배(<i>p</i>=0.035), 1.79배(<i>p</i>=0.035), 3.20배(<i>p</i>=10<sup>−9</sup>)다. pseudocost가 없을 때 MIP solver가 쓰는 '
  'most-fractional branching 대비로는 2.27배, 3.67배, 3.98배(<i>p</i>≤2×10<sup>−4</sup>)다. most-integral LP rule 대비로는 작은 두 크기에서 24배와 38배이고, 28×80에서 그 규칙은 100개 중 40개만 예산 안에 '
  '풀며 그 위에서 19배 많은 node를 쓴다(PAR-2 1,688 대 75 s).', body_s)
P('이득은 분포적이지 보편적이지 않다: 학습 규칙은 싼 규칙들에 대해 100개 중 61~80개에서 이기고 나머지에서 진다. 더 작은 tree를 만드는 유일한 규칙은 LP probing으로, 푸는 instance에서 2.7~2.9배다. '
  '이것은 strong branching의 전이형이다: 모든 node에서 relaxation이 반증할 수 있는 모든 변수를 고정하므로 어떤 점수보다 node당 정보가 많고, 그 대가로 node당 relaxation |<i>K</i>|개를 지불한다. Stage 1 '
  '단독(비용 조정 순서 없는 marginal 규칙)은 세 크기에서 전체 규칙보다 node를 2.3배, 3.2배, 5.0배 더 쓴다(짝 중앙값). 그 자체로는 21×60과 24×70 모두에서 LP rule보다 약 12배 아래이고, 21×60에서 싼 '
  '고전 규칙과 동급, 24×70부터는 그 뒤이며, 28×80 instance 10개를 풀지 못한다. 따라서 marginal만으로는 경쟁력 있는 값 신호이자 평범한 변수 점수이고, 규칙을 앞서게 하는 것은 비용 조정 순서다.', body_s)
TT('profile', ['크기', '규칙', 'ms / node', 'LP 비중', 'forward 비중', '비용 비율'], [1.6*cm, 4.4*cm, 1.7*cm, 1.5*cm, 1.8*cm, 1.7*cm],
   '표 3. dense 추론 경로의 node당 비용(단일 relaxation 행은 LP rule, 즉 most-integral branching이며 표 1의 모든 단일 relaxation 고전 규칙이 같은 node당 비용을 가진다): 크기당 instance 6개, 각 규칙의 전체 tree를 '
   '계측(node당 ms와 warm-start relaxation·network forward pass가 차지하는 node 시간 비중의 instance 간 중앙값; 나머지는 축소·propagation·장부 처리). 마지막 열은 그 행 대비 규칙의 node당 비용이다. forward pass 한 번은 0.33~0.53 ms이며 자유 변수 수에 따라 커진다.', right_from=2)
H3('Wall-clock')
P('작은 tree는 node당 비용이 허락할 때만 시간을 사며, 표 3이 전체 tree를 계측해 그 비용을 잰다. 단일 relaxation 규칙의 node는 세 크기에서 0.47, 0.55, 0.61 ms이고 그 90%가 warm-start '
  'relaxation이다. 학습 규칙의 node는 0.71, 0.73, 0.82 ms이며 forward pass가 28~32%, relaxation이 57~62%다. 따라서 overhead 비율은 <i>n</i>=60에서 1.5배, <i>n</i>=70, 80에서 1.3배인데 forward pass가 '
  'relaxation보다 크기에 따라 느리게 커지기 때문이고, 그동안 node 우위는 1.5배에서 3.2배로 오르며 wall-clock이 그 뒤를 따른다(표 2). 21×60에서 학습 규칙은 most-fractional·maxSD branching과 같은 '
  '속도이고 dom/wdeg보다 느리다(0.83배, <i>p</i>=0.021). 24×70에서는 most-fractional(2.4배, <i>p</i>=4×10<sup>−4</sup>)과 maxSD(1.5배, <i>p</i>=0.004)보다 빠르고 dom/wdeg와 동률이다. 28×80에서는 전부보다 '
  '3.1배, 2.4배, 2.0배 빠르고(<i>p</i>≤0.002), most-fractional 99개·uniform 91개 대비 100/100을 풀며, PAR-2가 가장 낮다. LP probing은 node당 relaxation을 자유 변수당 하나씩 13~18 ms 지불하여 더 작은 '
  'tree에도 세 크기에서 학습 규칙보다 5~8배 느리고, 28×80에서 instance 22개를 풀지 못한다(PAR-2 820 대 75 s). 역시 probing하는 impact-based search는 21×60에서 7배, 24×70에서 19배 느리면서 node를 '
  '6~9배 쓰고(짝 중앙값), 28×80에서는 49/100을 푼다(PAR-2 1,428 s).', body_s)
P('작은 두 크기(10×25, 18×50; 30개, tree 14~120 node)에서는 모든 규칙이 수백분의 1초 안에 있다. 학습 규칙은 10×25에서 가장 느린데, 14-node tree는 forward pass 14번을 상각할 수 없기 때문이다. 이 '
  '크기들은 부록 A에 보고하며 질문이 결정되는 곳이 아니다.', body_s)
H3('참조 신호')
P('표 1의 두 규칙은 배포 불가능하며 값 신호가 할 수 있는 것의 상한이다. oracle은 모든 node에서 <i>S</i>(<i>F</i>,<b>v</b>)를 열거하고 가장 확신하는 정확한 marginal로 분기한다. 10×25에서 node '
  '22개를 쓰며, Stage 1 network 21개, 비용 조정 규칙 14개, dom/wdeg 15.5개와 비교된다. 따라서 확신도 규칙 아래의 정확한 값 신호는 학습된 근사보다 나을 것이 없고 고전적 변수 점수보다 나쁘므로, 남은 '
  '여지는 Stage 2가 다루는 변수 선택에 있다. NSNet 전이형인 belief-propagation marginal은 18×50에서 Stage 1과 같은 크기의 tree(118 대 116 node)를 만들지만, 추정 한 번이 35 ms로 dense forward pass '
  '0.3~0.5 ms에 비해 비싸서 wall-clock으로는 학습 규칙의 50배다. marginal network는 이 신호를 훨씬 싼 값에 node당 한 번씩 돌리는 것이고, 이 비용의 message passing으로는 그렇게 할 수 없다. NSNet '
  '자체의 marginal 사용법, 즉 root 추정 하나를 incomplete local search의 초기 배정으로 decode하는 방식도 돌렸다. 잔차 ||<i>A</i><b>x</b>−<b>b</b>||<sub>1</sub>에 대한 WalkSAT식 탐색은 21×60의 100개를 '
  '전부 1.2~1.5 s에 풀어 학습 탐색의 3~4배 시간이 든다. 초기 배정이 marginal network·LP 반올림·균등 추출 어느 것에서 오든 같은 시간이 걸린다. 이 크기에서 marginal은 local search에 아무것도 사 주지 '
  '않고, 비교는 신호의 두 사용법이 아니라 두 탐색 사이의 것이다.', body_s)
H2('5.3 RQ2: 더 큰 instance로의 일반화')
IMG('rq2_scaling', 0.72 * W, '그림 7. 크기에 따른 스케일링. (a) 규칙별 node 중앙값과 (b) wall-clock 중앙값(<i>n</i>=25, 50은 30개; <i>n</i>=60, 70, 80은 100개). Stage 1은 <i>n</i>=25, Stage 2는 '
    '<i>n</i>=60에서 학습했으므로 <i>n</i>=70, 80은 외삽이다.')
P('두 단계 모두 중요한 크기 아래에서 학습했다: marginal network는 10×25 상태, branching 순서는 21×60 상태다. architecture는 크기와 무관하고(축소 instance 위의 graph network) 학습 신호는 '
  '어느 크기에서도 정의된다. 작은 instance에서 탐색 heuristic을 학습해 더 큰 instance에 배포하는 것은 operations research의 학습 탐색에서 통상적인 체제이며(Hottung et al., 2020; Yan et al., 2025), '
  '질문은 그 우위가 외삽을 견디는가이다. 그림 7과 표 2가 두 부분으로 답한다. 첫째, 규칙은 전이된다: 어느 단계도 본 적 없는 24×70과 28×80에서 모든 instance를 예산 안에 풀고 단일 relaxation 규칙 '
  '전부에 대해 유의한 node 우위를 유지한다. 둘째, 우위는 줄지 않고 크기와 함께 커진다. 최강의 싼 규칙 대비 node 비율은 <i>n</i>=60의 1.5배에서 70의 1.8배, 80의 3.2배로, most-fractional branching 대비 '
  '2.3배에서 3.7배, 4.0배로 오른다. dom/wdeg 대비 wall-clock 비율은 <i>n</i>=60(0.83배)과 70(1.09배) 사이에서 1을 넘어 80에서 2.0배에 이른다. 기제는 두 표의 곱이다. 어떤 규칙 R에 대해', body_s)
EQ('time(R) / time(ours) = [nodes(R) / nodes(ours)] × [ms/node(R) / ms/node(ours)] &nbsp;&nbsp;(8)')
P('이며 첫째 인자는 표 2의 node 비율, 둘째 인자는 표 3의 비용 비율의 역수다. 둘째 인자는 학습 규칙이 node마다 약 0.2 ms(세 크기에서 0.24, 0.18, 0.21 ms)의 forward pass를 지니므로 '
  '항상 치르는 벌점이지만, relaxation은 0.47에서 0.61 ms로 커지는데 forward pass는 그렇지 않아 벌점은 0.66, 0.75, 0.74에 머물고 크기에 따라 커지지 않는다. 최강의 싼 규칙 대비 node 비율 1.52, '
  '1.79, 3.20에 이 벌점을 곱하면 1.0, 1.3, 2.4가 되고, 이것이 비율의 중앙값과 중앙값의 비율 차이를 감안하면 표 2의 시간 비율(실측 0.83, 1.09, 1.96)이다. 따라서 시간 우위의 성장은 node 우위의 '
  '성장이 유계인 벌점을 통과한 것이다; forward pass가 relaxation보다 빨리 커졌다면 커지는 node 우위가 커지는 벌점에 잠식될 수 있었는데, 표 3은 그렇지 않음을 보인다. node 우위 자체도 탐색이 긴 곳에서 '
  '더 크며, 5.4절은 이를 비용 조정 순서가 틀린 값을 반증하는 방식으로 추적한다. 전이는 단계 사이에 고르지 않다. Stage 1 단독은 모든 크기에서 LP rule 대비 node 우위를 유지하지만 24×70부터 시간에서, 28×80에서 해결률에서(90/100) 싼 고전 규칙에 뒤진다. '
  '유용하게 전이되는 것은 조합이며, Stage 2의 Stage 1 대비 이득은 <i>n</i>=60의 2.3배에서 <i>n</i>=80의 5.0배로 커진다. 전이가 하지 못하는 것은 유용한 창을 넓히는 일이다: <i>n</i>=100에서는 어떤 '
  '규칙도 예산 안에 instance를 풀지 못하며 다음 크기에는 더 싼 node나 clause learning이 필요하다.', body_s)
H2('5.4 RQ3: 두 신호는 대체 가능한가, 각 단계는 필요한가?')
TT('rq3_cross', ['변수, 값', 'node (21×60)', '비율', 'node (24×70)', '비율', 'node (28×80)', '비율'], [3.6*cm, 1.5*cm, 3.3*cm, 1.5*cm, 3.3*cm, 1.5*cm, 2.3*cm],
   '표 4. 값×변수 교차 ablation: 각 arm은 변수 선택 점수 하나와 첫 값 규칙 하나를 결합하고 나머지는 동일하다. 크기별 열: node 중앙값과 Stage 1+2 대비 instance별 node 비율 '
   '중앙값(bootstrap 구간, sign test <i>p</i>값).', font=6.6)
IMG('rq3', 0.6 * W, '그림 8. 표 4의 교차 ablation을 Stage 1+2 대비 instance별 node 비율 중앙값으로(log 눈금, 95% bootstrap 구간), 세 크기.')
H3('값과 변수는 분리 가능하며 이득은 변수가 지닌다')
P('표 4와 그림 8은 변수 선택 점수 셋(LP most-integral, Stage 1 확신도, Stage 2 비용 조정)과 첫 값 규칙 둘(LP 반올림, Stage 1 marginal)을 각각 결합한다. 열로 읽으면 변수 점수가 결정적이다. '
  'Stage 1 값을 고정한 채 변수 점수를 LP에서 Stage 1, Stage 2로 옮기면 21×60에서 tree가 12배, 다시 2.3배 줄어든다. 정확히 학습한 marginal 값을 붙인 LP 변수는 21×60에서 전체 규칙보다 28배, 24×70에서 '
  '37배 나쁘고, 전체 규칙이 전부 푸는 28×80에서 100개 중 26개를 푼다. 행으로 읽으면 값 규칙은 결정적이지 않다. Stage 2 변수를 고정한 채 marginal 값을 LP 반올림으로 바꾸면 tree는 21×60에서 '
  '0.79배(<i>p</i>=0.09), 24×70에서 1.19배(<i>p</i>=0.27), 28×80에서 1.41배(<i>p</i>=0.09)로 어느 것도 5% 수준에서 유의하지 않되, tree가 커질수록 방향은 marginal 값 쪽으로 기운다. Stage 1 변수를 고정하면 LP '
  '값이 작은 두 크기에서 1.5배와 1.3배 더 많은 node를 들이고, 그 arm은 크기와 함께 전체 규칙에서 더 멀어지는데(2.8배, 3.3배, 5.9배) 이것이 tree와 함께 커지는 Stage 2의 이득이다.', body_s)
P('두 결론이 따르며 하나는 4절의 설계에 반한다. 신호는 중요한 의미에서 대체 불가능하다: marginal로 학습한 network는 trunk와 Stage 2가 출발하는 확신도 점수를 통해 좋은 변수 점수를 가능하게 하는 '
  '것이고, 어떤 값 정확도도 LP 변수 선택을 구제하지 못한다. 그러나 정확한 marginal <b>값</b>은 대체 가능하다. 비용 조정 점수로 변수를 고르고 나면 relaxation을 반올림해도 어느 크기에서도 유의하게 다르지 '
  '않은, 28×80에서 기껏 1.4배 큰 tree가 나온다. 이는 확신도 규칙 아래의 정확한 marginal조차 고전적 변수 점수를 이기지 못한다는 5.2절의 oracle 결과와 일치한다. 이 방법에서 값 신호의 역할은 값을 '
  '고르는 것이 아니라 표현을 학습시키는 것이다.', body_s)
H3('Stage 2가 배운 것')
P('fine-tuning은 instance별 기록이 드러내는 방식으로 탐색을 바꾼다(부록 B의 표 A2). 모든 규칙에 대해 해 경로에서 첫 번째로 시도한 값이 반증된(그 아래 subtree가 소진된) 빈도와 그 반증 한 번의 '
  '비용을 기록한다. Stage 1 확신도 규칙은 가장 드물게 틀리고(21×60에서 결정의 6%) 실수당 가장 많이 지불한다(246 node). 비용 조정 규칙은 세 배 자주 틀리고(19%) 실수당 4분의 1을 지불하며(65 node), '
  '고르는 변수가 더 많은 propagation과 relaxation 고정을 촉발하므로 해 경로가 더 짧다(결정 46개 대 28개). 24×70의 고전 규칙에 대해서도 같다: dom/wdeg, maxSD, most-fractional branching은 결정의 '
  '14~16%에서 각 835~911 node로 반증되고 학습 규칙은 21%에서 387 node다. 이는 설계가 아니라 비용 신호에서 학습된 fail-first(Haralick and Elliott, 1980)다: policy는 가장 확신하는 변수에서 벗어나 틀린 '
  '값이 싸게 소진되는 변수로 옮겨 간다. Stage 2 변수에 LP 값을 붙인 arm도 같은 양상(16%, 63 node)이므로 구조는 변수 선택에서 온다.', body_s)
TT('rq3_variants', ['변형', 'node (21×60)', '비율', 'node (24×70)', '비율', 'node (28×80)', '비율'], [3.9*cm, 1.5*cm, 3.2*cm, 1.5*cm, 3.2*cm, 1.7*cm, 2.0*cm],
   '표 5. 학습 변형: 기준 Stage 1+2 규칙 대비 node 중앙값과 instance별 node 비율 중앙값, 크기별. Stage 1 변형은 확신도 규칙으로 돌린다(Stage 2 없음); Stage 2 변형은 따로 '
   '명시하지 않는 한 동결된 기준 trunk를 공유한다.', font=6.4)
H3('각 단계는 필요한가, 레시피의 어느 부분이 중요한가?')
P('표 5는 나머지를 고정한 채 각 단계의 학습을 바꾼다. Stage 1 레시피는 tree를 바꾸기는 한다 — 21×60에서 최대 1.5배, 28×80에서 약 ±25% — 그러나 값 정확도가 예측하는 방향으로는 아니고, 전체 규칙에 '
  '근접할 만큼도 아니다. Stage 1에서는 레시피의 어느 것도 값 정확도를 움직이지 않는다. 라벨 형태(심어진 0/1 해 대 soft marginal), 학습 seed, 학습 상태 수(10,000 대 '
  '2,771)는 FORCED 변수의 테스트 정확도를 기껏 0.2 point 바꾼다(상한 83.3% 대비 79.0~79.2%). 같은 상태로 학습한 NSNet architecture(Li and Si, 2022)의 network는 80.4%에 이르고, 10×25 대신 21×60 '
  '상태로 학습하면(M1) 21×60 상태의 정확도가 2.4 point 오른다. tree는 다른 이야기를 한다. 기준 레시피의 학습 seed 둘은 21×60에서 기준의 628 대비 553과 562 node를 쓰므로 Stage 1 규칙에는 약 ±10%의 '
  'seed 분산이 있고, 어떤 라벨·architecture 선택도 올바른 방향으로 그것을 넘지 못한다. 값 정확도가 <b>더 높은</b> 두 network인 M1과 NSNet architecture가 <b>가장 큰</b> tree를 쓴다: 21×60에서 912와 '
  '892, 24×70에서 기준 레시피의 6,199~8,175 대비 10,754와 12,500이다. 28×80에서는 예산 안에 가장 적게 푸는데(100개 중 85와 70; 다른 Stage 1 변형 91~95, 전체 규칙 100), 모든 Stage 1 변형이 거기서 '
  '전체 규칙의 4.0~5.7배 node를 쓴다.', body_s)
P('따라서 값 정확도와 tree 크기는 같은 양이 아니며, 이는 표 4를 반대편에서 본 요점이다. Stage 1 레시피가 필요한 것은 이 network들 중 어느 것이든 Stage 2가 2~3배 개선하는 trunk와 확신도 점수를 '
  '공급한다는 의미에서만이다.', body_s)
P('Stage 2에서는 레시피가 우리가 설계한 것보다 덜 중요하다. policy head를 무작위로 시작하면 — 이 trunk로 from-scratch 강화학습이 할 일 — 초기 policy는 marginal 규칙의 7배 root 비용에서 출발하지만, '
  '6 epoch 뒤에는 기준과 동급이다(세 크기에서 1.04배, 0.86배, 1.03배, 모두 비유의). LP 규칙에서 시작해도 같은 곳에 이른다(1.06배, 0.86배, 1.12배). trunk를 풀면 held-out node 수가 셋째 epoch부터 '
  'Stage 1 규칙의 2.7~5.4배로 발산하지만, epoch 선택이 남기는 첫 epoch의 checkpoint는 동결된 것만큼 좋다(0.76배, 0.83배, 1.06배, 비유의). 따라서 동결된 trunk는 fine-tuning을 계속할 수 있을 만큼 '
  '안정하게 유지하고 이득을 순서에 귀속시키는 것이다. tree 크기의 원천은 아니며, 우리는 그렇게 주장하지 않는다.', body_s)
H2('5.5 RQ4: Complete solver와의 경합')
TT('rq4_scip', ['conflict', '규칙', '해결', 'node 중앙값', '초 중앙값', 'node, 규칙 / plug-in', '시간, 규칙 / plug-in'], [1.3*cm, 3.4*cm, 1.4*cm, 1.8*cm, 1.4*cm, 3.9*cm, 3.9*cm],
   '표 6. SCIP 안(presolve·cut·primal heuristic 끔, depth-first node 선택, 첫 해에서 정지): SCIP 자체 규칙 대 branching plug-in으로 꽂은 학습 규칙, conflict analysis off/on. 비율은 '
   '표 2와 같이 규칙 / plug-in.', font=6.6, right_from=2)
H3('SCIP 안의 branching 규칙으로')
P('학습 규칙은 축소 instance와 그 relaxation vertex의 함수이므로 MIP solver의 branching 규칙으로 설치할 수 있다. 우리는 PySCIPOpt branching plug-in으로 구현하고 SCIP(Bestuzheva et al., 2021)를 '
  '같은 instance에서 presolve·cutting plane·primal heuristic을 끄고 depth-first node 선택과 첫 해에서의 정지로 돌렸다. 그러면 SCIP의 탐색은 우리 루프가 하는 것과 같되 propagation과 LP만 SCIP의 것이 '
  '된다. 비교 대상은 reliability pseudocost branching(RPB; SCIP 기본값이자 learning-to-branch 문헌이 모방하는 규칙, Achterberg et al., 2005), SCIP의 most-fractional 규칙, inference 규칙이다. '
  'conflict analysis는 SCIP의 clause learning이다. node가 relaxation이나 propagation으로 infeasible로 판정되면 solver는 dual ray나 propagation 이유에서 함께 성립할 수 없는 bound 변경의 최소 집합을 '
  '추출해 전역 conflict constraint로 더하고, 이후 같은 조합에 이르는 모든 subtree는 탐색 없이 잘린다. 우리 루프에는 그런 장치가 없으므로 끄고 켜서 모두 돌린다.', body_s)
P('표 6은 node 우위가 엔진 교체와 clause learning 추가를 견딤을 보인다. 100개 세트에서 plug-in은 conflict analysis off/on 모두 모든 크기에서 RPB보다 node를 1.9~3.8배 적게 쓰고(sign test 전부 '
  '<i>p</i>≤3×10<sup>−5</sup>) SCIP의 most-fractional·inference 규칙보다 1.7~4.0배 적게 쓴다. clause learning은 두 규칙을 불균등하게 돕는다. 켜면 RPB는 24×70에서 22%, 28×80에서 34% 적은 node를 쓰고 '
  'plug-in은 6%와 3%만 줄어, RPB 대비 node 비율은 3.5~3.8배에서 1.9~3.1배로 줄되 유의하게 남는다. 틀린 값을 이미 얕고 싸게 반증하는 규칙은 다른 곳에서 재사용할 긴 conflict를 적게 만들고, bound 중심 '
  '규칙은 conflict constraint가 잘라낼 것을 더 남긴다는 읽기다. 시간은 RPB와는 동률, 가장 싼 규칙에는 패배다. forward pass가 branching callback의 60~80%라 tree가 수백 node인 21×60에서 plug-in은 모든 '
  'SCIP 규칙보다 3배 느리다. 24×70과 28×80에서는 conflict analysis off/on 모두 중앙 시간이 RPB의 0.8~1.1배 안에 있고(짝 차이 어느 것도 비유의) SCIP most-fractional 규칙에는 0.66~0.86배로 '
  '뒤진다(<i>p</i>=0.02~0.06). 따라서 solver 안에서 RQ1의 답은 갈린다: 학습 규칙은 모든 크기에서 tree 크기로는 더 나은 branching 규칙이고, tree가 forward pass를 상각할 만큼 크면 solver 기본 규칙과 '
  '시간이 같으며, 가장 싼 규칙을 시간에서 이기지는 못한다.', body_s)
TT('rq4_e2e', ['방법', '해결', '초 중앙값', 'PAR-2 s'], [5.2*cm, 2.0*cm, 2.0*cm, 2.0*cm],
   '표 7. end-to-end, 같은 instance·예산, 단일 thread: 기본 설정의 complete solver, bounded enumeration을 갖춘 lattice reduction, 학습 탐색. 해결 instance의 시간 중앙값과 PAR-2; node 수는 '
   '엔진 간 비교 불가라 생략한다.', font=7.2)
H3('End-to-end')
P('표 7은 학습 탐색을 기본 설정의 complete solver(thread 하나의 Gurobi, Gurobi Optimization, 2024; worker 하나의 CP-SAT, Perron and Didier, 2023; presolve·cut·heuristic·conflict '
  'analysis·RPB를 갖춘 SCIP)와 bounded enumeration을 갖춘 lattice reduction(BKZ block 40, 열 순열 30개, 300 s 캡의 fpylll enumeration, FPLLL team, 2023) 옆에 같은 instance와 예산으로 놓는다. '
  '모든 방법이 모든 instance를 푼다. Gurobi가 모든 크기에서 가장 빠르고(<i>n</i>=60, 70, 80에서 중앙 0.05, 0.31, 6.6 s) CP-SAT가 그 다음이다(0.04, 0.36, 12.2 s; 28×80에서 꼬리가 무겁다). 기본 설정 '
  'SCIP는 24×70에서 학습 탐색과 동률(2.33 대 2.39 s)이고 28×80에서 2.8배 빠르다(16.5 대 46.6 s). 두 MIP solver 사이의 2.5~8배 격차는 우리 설정 탓이 아니다. 둘 다 thread 하나에서 기본 설정으로 돌며, '
  '차이는 탐색이 빨리 닫히는 instance에서 효과를 보는 LP 엔진과 탐색 구현에서 온다(SCIP 자체 통계에서 presolve와 primal heuristic은 이 instance에서 아무것도 찾지 못한다). 표 6의 plug-in 비교는 '
  'presolve와 cut을 끈 SCIP라 다른 측정이다. lattice enumeration은 21×60 세트를 instance당 0.2 s에 풀지만 비용이 탐색보다 빠르게 커진다: 24×70에서 13.2 s, 28×80에서 125 s로 학습 탐색보다 5.5배와 '
  '2.7배 느리다.', body_s)
P('두 가지가 동시에 참이다. 학습 탐색은 이 instance를 푸는 가장 빠른 방법이 아니다: 상용 MIP solver와 presolve·clause learning을 갖춘 CP-SAT solver는 <i>n</i>≤70에서 한두 자릿수, <i>n</i>=80에서 '
  '4~7배 앞서고 SCIP는 가장 큰 크기에서 앞선다. 그리고 그것은 branching 질문과 무관하다. branching 결정을 분리하는 비교는 표 2이며, SCIP 안에서 결정 품질은 그대로 옮겨지고 node당 비용은 그렇지 않다. '
  '정직한 요약은 본 논문이 자기 루프와 SCIP 안에서 branching 규칙을 개선하지만 그 자체로 더 빠른 solver를 만들지는 않는다는 것이다. solver와의 격차는 clause learning과 compiled node이고 학습 규칙은 그 '
  '어느 것도 다루지 않는다.', body_s)
# ═══════════════ 6 ═══════════════
H1('6. 결론')
P('branching 결정은 두 질문을 던지며 본 논문은 각각의 정확한 답으로 학습했다: 해를 열거할 수 있는 곳에서 계산 가능하고 reduction 항등식으로 모든 node에 옮겨지는 conditional marginal은 어느 '
  '값을 시도할지에, 탐색이 backtrack하면서 산출하는 결정 아래 node 수는 어느 변수로 분기할지에 답한다. 고전 규칙 7종과 learned-branching 세 계열의 전이형에 대해, 한 탐색 루프에서, '
  '100개 세트로, 학습 규칙은 모든 크기에서 단일 relaxation 규칙 중 node를 가장 적게 쓰고 28×80에서는 가장 빠르기도 하다; <i>n</i>≤60에서 학습한 그 우위는 줄지 않고 크기와 함께 커진다; 그리고 '
  'node 우위를 SCIP 안으로 가져가 conflict analysis on/off의 reliability branching에 맞선다. ablation은 레시피가 무엇인지 말한다: 변수 선택이 이득을 지니고, 비용 조정 순서는 설계가 아니라 학습된 '
  'fail-first이며, 동결된 marginal trunk는 fine-tuning을 안정하게 유지하는 것이고, 정확한 marginal 값은 변수를 잘 고르고 나면 LP 반올림으로 대체 가능하며 가장 큰 크기에서 '
  '기껏 1.4배 차이다.', body_s)
P('세 한계가 결과를 제한한다. 주장은 하나의 instance 계열, 즉 3.1절의 심어진 0/1 market-split 구성에 대한 것이며, 다른 제약 계열이나 실제 스캐너 구조의 투영 행렬에서도 성립하는지는 검증하지 않았다. 학습 탐색은 '
  'end-to-end로 경쟁력이 없다: 기본 설정의 Gurobi·CP-SAT·SCIP가 <i>n</i>≤70에서 자릿수 차이로 더 빠르므로 기여는 solver가 아니라 branching 규칙에 있고, lattice enumeration도 우리가 다루는 모든 '
  'instance를 풀되 큰 두 크기에서는 학습 탐색보다 느리다. 그리고 정확한 지도는 Stage 1을 열거되는 크기로 묶으므로 값 신호는 배포되는 곳에서 학습될 수 없다; LP 값이 그 역할을 하므로 그럴 필요가 '
  '없다는 것이 이 한계를 덜어 주는 한 결과다.', body_s)
P('다음 단계는 부정적 결과에서 따른다. 값 신호의 역할이 표현을 학습시키는 것이라면 정확한 marginal보다 싼 target으로도 될 수 있고, Stage 2의 비용 신호는 열거가 전혀 필요 없으므로 열거가 '
  '실패하는 크기에서, 그리고 결정 아래 node 수는 subtree에 해가 있든 없든 정의되므로 본 논문이 제외한 infeasible instance에서 순서를 fine-tuning할 수 있다. solver 안의 시간 격차는 forward pass '
  '예산이다: plug-in은 network가 callback의 60~80%인 채 시간에서 지므로, 더 작은 network나 node의 부분집합에서만 평가하는 학습 규칙이 두 지표 모두에서 이기는 규칙으로 가는 길이다.', body_s)

# ═══════════════ 부록 ═══════════════
H1('부록 A. 작은 두 크기')
TT('small', ['규칙', '해결', 'node 중앙값', '초 중앙값', 'PAR-2 s', 'node, 규칙 / Stage 1+2', '시간, 규칙 / Stage 1+2'], [3.2*cm, 1.4*cm, 1.8*cm, 1.4*cm, 1.4*cm, 3.9*cm, 3.9*cm],
   '표 A1. 작은 두 크기(각 30개): 표 2와 같은 열. tree는 수십 node이며 어떤 규칙도 점수의 비용 외에는 시간에서 갈라지지 않는다.', font=6.8)
P('표 A1은 10×25와 18×50을 보고한다. 참조 신호가 여기에 나타나는 것은 oracle이 모든 node에서 <i>S</i>(<i>F</i>,<b>v</b>)를 열거해 10×25에서만 감당 가능하고 belief propagation이 18×50에서 예산 '
  '안에 끝나기 때문이다.', body_s)
H1('부록 B. 모든 규칙의 반증 구조')
P('표 A2는 모든 규칙과 모든 해결 instance에 대해 탐색이 node를 어떻게 썼는지 기록한다. <i>경로</i>는 해에 이르는 branching 결정 수, <i>반증 비중</i>은 해 없이 소진된 subtree에 속한 node의 비율, '
  '<i>첫 자식 실패율</i>은 첫 값이 반증된 경로 결정의 비율(값 선택이 틀린 빈도), <i>반증 비용</i>은 그런 틀린 첫 자식이 소진되기까지 든 평균 node 수, <i>깊이 비율</i>은 그 틀린 결정들의 깊이 중앙값을 경로 '
  '길이로 나눈 것이다. 세 가지 읽기가 5.4절의 논증을 떠받친다. 첫째, 모든 규칙이 21×60에서 node의 90% 이상, 28×80에서 99% 이상을 반증된 subtree에 쓰므로 규칙은 얼마나 드물게 틀리느냐가 아니라 '
  '얼마나 싸게 반증하느냐로 평가된다. 둘째, Stage 1 규칙은 가장 드물게 틀리고 실수당 가장 많이 지불하는 규칙이며, 비용 조정 규칙은 2~3배 자주 틀리고 3~5배 적게 지불한다; Stage 2 변수에 LP 값을 붙인 '
  'arm도 같은 양상이므로 변화는 변수 선택에서 온다. 셋째, 비용 조정 규칙의 해 경로는 모든 크기에서 단일 relaxation 규칙 중 가장 짧은데 고르는 변수가 더 많은 propagation과 relaxation 고정을 '
  '촉발하기 때문이고, LP probing은 변수를 직접 고정하므로 더 짧다.', body_s)
TT('trace', ['규칙', '<i>n</i>', 'node', '경로', '반증 비중', '첫 자식 실패율', '반증 비용', '깊이 비율'], [2.9*cm, 1.2*cm, 1.9*cm, 1.1*cm, 1.8*cm, 2.2*cm, 1.9*cm, 1.7*cm],
   '표 A2. 규칙별 반증 구조(해결 instance; instance 간 중앙값): node, 해 경로의 결정 수, 반증된 subtree의 node 비중, 첫 값이 반증된 경로 결정의 비율, 그런 반증 한 번의 평균 비용, '
   '반증된 결정의 깊이 중앙값(경로 길이 대비).', font=8.0)
H1('부록 C. 고전 규칙의 구현')
P('모든 규칙은 알고리즘 1을 공유하고 5단계만 다르며, 각 규칙은 축소 instance (<i>A</i>′, <b>b</b>′)와 relaxation vertex <b>x</b><sup>LP</sup>를 받는다. <i>most-integral</i>과 <i>most-fractional</i>은 '
  'argmax<sub><i>j</i></sub> |<i>x</i><sup>LP</sup><sub><i>j</i></sub>−1/2|와 argmin<sub><i>j</i></sub> |<i>x</i><sup>LP</sup><sub><i>j</i></sub>−1/2|로 분기하고 반올림 값을 먼저 시도한다. <i>maxSD</i>: 자유 변수 '
  '<i>r<sub>i</sub></i>개와 우변 <i>b</i>′<sub><i>i</i></sub>를 가진 0/1 등식 행에서 <i>x<sub>j</sub></i>=1의 solution density는 그 행의 모든 변수에 대해 <i>b</i>′<sub><i>i</i></sub>/<i>r<sub>i</sub></i>, <i>x<sub>j</sub></i>=0은 '
  '1−<i>b</i>′<sub><i>i</i></sub>/<i>r<sub>i</sub></i>다; 규칙은 모든 행과 변수에 걸쳐 밀도가 최대인 변수–값 쌍을 취하고, 동률은 그 변수를 포함하는 행들의 밀도 합으로 깬다. <i>dom/wdeg</i>: 모든 자유 변수의 '
  'domain 크기가 2이므로 규칙은 conflict 가중 차수의 최대로 환원된다; 원래 행마다 가중치(초기 0)가 있어 2~3단계의 bound check나 propagation이 그 행 때문에 node를 버릴 때마다 1씩 늘고, 변수의 점수는 '
  '그 변수가 나타나는 행들의 (1+가중치) 합이며 반올림 LP 값을 먼저 시도한다. <i>impact</i>: node에서 모든 자유 변수와 두 값에 대해 값을 고정하고 propagation을 돌린다; 쌍의 impact는 1 더하기 '
  'propagation이 고정하는 변수 수이고, 모순인 쌍은 반증된 것으로 기록해(LP probing처럼) 반대 값을 고정하며, 규칙은 impact 합이 최대인 변수로 분기하고 impact가 작은 값을 먼저 시도한다. <i>LP probing</i>: '
  '모든 자유 변수와 두 값에 대해 그 값을 고정한 relaxation을 다시 풀고, relaxation이 infeasible한 값은 반증해 반대 값을 고정하며, 반증이 없으면 most-integral 선택으로 돌아간다. <i>uniform</i>: 자유 변수와 '
  '값을 균등하게 뽑는다. <i>oracle</i>: 모든 node에서 CP-SAT로 <i>S</i>(<i>F</i>,<b>v</b>)를 열거하고 정확한 marginal에 규칙 (5)를 적용한다. <i>BP</i>: 정확한 cardinality-factor message를 prefix–suffix '
  'convolution으로 계산하는 damped sum-product(damping 0.9, 허용오차 10<sup>−5</sup>, 최대 300회)이며 추정치에 규칙 (5)를 적용한다.', body_s)
H1('부록 D. 타이밍 복제')
TT('seeds', ['크기', 'arm', 'seed 0', 'seed 1', 'seed 2'], [1.8*cm, 2.2*cm, 3.0*cm, 3.0*cm, 3.0*cm],
   '표 A3. 평가 seed 셋에 걸친 학습 arm의 타이밍 복제(seed 0이 본문 표에 보고한 것): 해결 instance 수와 wall-clock 중앙값. 탐색은 relaxation vertex가 주어지면 결정적이므로 node 수는 '
   'seed 간에 동일하다.', right_from=2)
P('node 수는 seed 불변이다; 표 A3은 학습 arm의 wall-clock을 평가 seed 두 개로 더 복제한다. 작은 두 크기의 seed 0 run은 더 가벼운 동시 부하에서 돈 seed 1, 2보다 20~30% 느리며, 본문의 비율은 전부 '
  'seed 0을 쓰므로 학습 arm에 보수적이다. 28×80에서 세 seed는 1% 안에서 일치한다.', body_s)

H1('데이터와 코드 공개')
P('instance 생성기, 학습·평가 코드, 학습된 network, 모든 표의 instance별 기록은 출판 시 공개 저장소에 공개한다.', body_s)
H1('참고문헌')
for r in sorted([
 'Hottung, A., Tanaka, S., and Tierney, K. (2020). Deep learning assisted heuristic tree search for the container pre-marshalling problem. <i>Computers & Operations Research</i>, 113:104781.',
 'Mazyavkina, N., Sviridov, S., Ivanov, S., and Burnaev, E. (2021). Reinforcement learning for combinatorial optimization: A survey. <i>Computers & Operations Research</i>, 134:105400.',
 'Pryor, J. and Chinneck, J. W. (2011). Faster integer-feasibility in mixed-integer linear programs by branching to force change. <i>Computers & Operations Research</i>, 38(8):1143–1152.',
 'Smit, I. G., Zhou, J., Reijnen, R., Wu, Y., Chen, J., Zhang, C., Bukhsh, Z., Zhang, Y., and Nuijten, W. (2025). Graph neural networks for job shop scheduling problems: A survey. <i>Computers & Operations Research</i>, 176:106914.',
 'Yan, F., Clautiaux, F., Froger, A., and Albar, B. (2025). Generic machine-learning-augmented beam search for resource-constrained shortest path reformulations of combinatorial optimization problems. <i>Computers & Operations Research</i>, 187:107339.',
 'Achterberg, T., Koch, T., and Martin, A. (2005). Branching rules revisited. <i>Operations Research Letters</i>, 33(1):42–54.',
 'Bestuzheva, K., Besançon, M., Chen, W.-K., et al. (2021). The SCIP Optimization Suite 8.0. Technical Report 21-41, Zuse Institute Berlin.',
 'Boussemart, F., Hemery, F., Lecoutre, C., and Sais, L. (2004). Boosting systematic search by weighting constraints. In <i>ECAI</i>, pages 146–150.',
 'The FPLLL development team (2023). fpylll, a Python wrapper for fplll. https://github.com/fplll/fpylll.',
 'Gurobi Optimization, LLC (2024). Gurobi Optimizer Reference Manual.',
 'Perron, L. and Didier, F. (2023). OR-Tools CP-SAT. Google.',

 'Aardal, K., Hurkens, C. A. J., and Lenstra, A. K. (2000). Solving a system of linear Diophantine equations with lower and upper bounds on the variables. <i>Mathematics of Operations Research</i>, 25(3):427–442.',
 'Aardal, K., Bixby, R. E., Hurkens, C. A. J., Lenstra, A. K., and Smeltink, J. W. (2000). Market split and basis reduction: Towards a solution of the Cornuéjols–Dawande instances. <i>INFORMS Journal on Computing</i>, 12(3):192–202.',
 'Amos, B. (2023). Tutorial on amortized optimization. <i>Foundations and Trends in Machine Learning</i>, 16(5):592–732.',
 'Cappart, Q., Moisan, T., Rousseau, L.-M., Prémont-Schwarz, I., and Cire, A. A. (2021). Combining reinforcement learning and constraint programming for combinatorial optimization. In <i>AAAI Conference on Artificial Intelligence</i>.',
 'Chen, Z., Liu, J., Wang, X., Lu, J., and Yin, W. (2023). On representing mixed-integer linear programs by graph neural networks. In <i>International Conference on Learning Representations</i>.',
 'Chu, G. and Stuckey, P. J. (2015). Learning value heuristics for constraint programming. In <i>CPAIOR</i>.',
 'Cornuéjols, G. and Dawande, M. (1999). A class of hard small 0-1 programs. In <i>Integer Programming and Combinatorial Optimization</i>, pages 284–293.',
 'Etheve, M., Alès, Z., Bissuel, C., Juan, O., and Kedad-Sidhoum, S. (2020). Reinforcement learning for variable selection in a branch and bound algorithm. In <i>CPAIOR</i>, pages 176–185.',
 'Feng, S. and Yang, Y. (2025). SORREL: Suboptimal-demonstration-guided reinforcement learning for learning to branch. In <i>AAAI Conference on Artificial Intelligence</i>.',
 'Gasse, M., Chételat, D., Ferroni, N., Charlin, L., and Lodi, A. (2019). Exact combinatorial optimization with graph convolutional neural networks. In <i>Advances in Neural Information Processing Systems</i>, volume 32.',
 'Gershman, S. J. and Goodman, N. D. (2014). Amortized inference in probabilistic reasoning. In <i>Proceedings of the Annual Meeting of the Cognitive Science Society</i>, volume 36.',
 'Han, Q., Yang, L., Chen, Q., Zhou, X., Zhang, D., Wang, A., Sun, R., and Luo, X. (2023). A GNN-guided predict-and-search framework for mixed-integer linear programming. In <i>International Conference on Learning Representations</i>.',
 'Haralick, R. M. and Elliott, G. L. (1980). Increasing tree search efficiency for constraint satisfaction problems. <i>Artificial Intelligence</i>, 14(3):263–313.',
 'Khalil, E. B., Le Bodic, P., Song, L., Nemhauser, G., and Dilkina, B. (2016). Learning to branch in mixed integer programming. In <i>AAAI Conference on Artificial Intelligence</i>.',
 'Khalil, E. B., Morris, C., and Lodi, A. (2022). MIP-GNN: A data-driven framework for guiding combinatorial solvers. In <i>AAAI Conference on Artificial Intelligence</i>.',
 'Lenstra, A. K., Lenstra, H. W., and Lovász, L. (1982). Factoring polynomials with rational coefficients. <i>Mathematische Annalen</i>, 261(4):515–534.',
 'Li, Z. and Si, X. (2022). NSNet: A general neural probabilistic framework for satisfiability problems. In <i>Advances in Neural Information Processing Systems</i>, volume 35.',
 'Li, Z., Guo, J., and Si, X. (2023). G4SATBench: Benchmarking and advancing SAT solving with graph neural networks. <i>Transactions on Machine Learning Research</i>.',
 'Marques-Silva, J. P. and Sakallah, K. A. (1999). GRASP: A search algorithm for propositional satisfiability. <i>IEEE Transactions on Computers</i>, 48(5):506–521.',
 'Montanari, A., Ricci-Tersenghi, F., and Semerjian, G. (2007). Solving constraint satisfaction problems through belief propagation-guided decimation. In <i>Allerton Conference on Communication, Control, and Computing</i>.',
 'Nair, V., Bartunov, S., Gimeno, F., et al. (2020). Solving mixed integer programs using neural networks. arXiv:2012.13349.',
 'Ohrimenko, O., Stuckey, P. J., and Codish, M. (2009). Propagation via lazy clause generation. <i>Constraints</i>, 14(3):357–391.',
 'Parsonson, C. W. F., Laterre, A., and Barrett, T. D. (2023). Reinforcement learning for branch-and-bound optimisation using retrospective trajectories. In <i>AAAI Conference on Artificial Intelligence</i>.',
 'Pesant, G., Quimper, C.-G., and Zanarini, A. (2012). Counting-based search: Branching heuristics for constraint satisfaction problems. <i>Journal of Artificial Intelligence Research</i>, 43:173–210.',
 'Refalo, P. (2004). Impact-based search strategies for constraint programming. In <i>Principles and Practice of Constraint Programming (CP)</i>, pages 557–571.',
 'Scavuzzo, L., Chen, F. Y., Chételat, D., Gasse, M., Lodi, A., Yorke-Smith, N., and Aardal, K. (2022). Learning to branch with tree MDPs. In <i>Advances in Neural Information Processing Systems</i>, volume 35.',
 'Schnorr, C. P. and Euchner, M. (1994). Lattice basis reduction: Improved practical algorithms and solving subset sum problems. <i>Mathematical Programming</i>, 66:181–199.',
 'Selsam, D., Lamm, M., Bünz, B., Liang, P., de Moura, L., and Dill, D. L. (2019). Learning a SAT solver from single-bit supervision. In <i>International Conference on Learning Representations</i>.',
 'Vaezipoor, P., Lederman, G., Wu, Y., Maddison, C., Grosse, R. B., Seshia, S. A., and Bacchus, F. (2021). Learning branching heuristics for propositional model counting. In <i>AAAI Conference on Artificial Intelligence</i>.',
 'Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. (2018). Graph attention networks. In <i>International Conference on Learning Representations</i>.',
 'Wassermann, A. (2025). Solving the market split problem with lattice enumeration. arXiv:2508.08702; <i>Mathematical Programming Computation</i>, to appear.',
 'Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. <i>Machine Learning</i>, 8(3–4):229–256.'], key=lambda x: x.split('(')[0].lower()):
	P(r, ref_s)


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.0*cm, rightMargin=2.0*cm, topMargin=2.0*cm, bottomMargin=2.0*cm,
                        title='Generalisable Learned Branching for Binary Linear Systems (KR, 261012)', author='Ko, Cheong, Choi')
def footer(canvas, doc_):
	canvas.saveState(); canvas.setFont('NotoKR', 8); canvas.setFillColor(colors.black)
	canvas.drawCentredString(A4[0] / 2.0, 1.1 * cm, str(doc_.page)); canvas.restoreState()
doc.build(E, onFirstPage=footer, onLaterPages=footer)
print('saved:', os.path.abspath(OUT))
