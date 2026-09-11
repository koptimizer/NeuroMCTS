#!/usr/bin/env python3
"""Builds the v11 EN and KR pptx decks directly via python-pptx (bypasses the
broken Korean xelatex/lualatex toolchain in this environment)."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

COLORS = dict(bg='FFFFFF', primary='1F4E79', accent='2E75B6', body='2D2D2D',
              muted='777777', rule='CCCCCC', good='2E7D32', bad='C62828')
FIG_DIR = '/home/kopt/neuroMCTS/figures/'


def rgb(hexstr):
	return RGBColor.from_string(hexstr)


def new_deck():
	p = Presentation()
	p.slide_width = Inches(10)
	p.slide_height = Inches(5.625)
	return p


def blank_slide(p):
	return p.slides.add_slide(p.slide_layouts[6])


def add_title(slide, text, color=COLORS['primary'], y=0.25, size=24):
	box = slide.shapes.add_textbox(Inches(0.5), Inches(y), Inches(9.0), Inches(0.8))
	tf = box.text_frame; tf.word_wrap = True
	pr = tf.paragraphs[0]; pr.text = text
	pr.font.size = Pt(size); pr.font.bold = True; pr.font.color.rgb = rgb(color)
	line = slide.shapes.add_shape(1, Inches(0.5), Inches(y + 0.75), Inches(9.0), Pt(1.5))
	line.fill.solid(); line.fill.fore_color.rgb = rgb(COLORS['rule']); line.line.fill.background()
	return box


def add_bullets(slide, items, x=0.5, y=1.15, w=9.0, h=3.9, size=16, color=COLORS['body']):
	box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
	tf = box.text_frame; tf.word_wrap = True
	for i, (text, level, bold) in enumerate(items):
		pr = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
		pr.text = text; pr.level = level
		pr.font.size = Pt(size - level * 2); pr.font.bold = bold
		pr.font.color.rgb = rgb(color if level == 0 else COLORS['muted'])
		pr.space_after = Pt(10)
	return box


def add_footer(slide, text, num):
	box = slide.shapes.add_textbox(Inches(0.5), Inches(5.3), Inches(9.0), Inches(0.3))
	tf = box.text_frame
	pr = tf.paragraphs[0]; pr.text = f"{text}    ·    {num}"
	pr.font.size = Pt(9); pr.font.color.rgb = rgb(COLORS['muted'])


def title_slide(p, title, subtitle, meta):
	s = blank_slide(p)
	s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(COLORS['primary'])
	box = s.shapes.add_textbox(Inches(0.7), Inches(1.6), Inches(8.6), Inches(1.8))
	tf = box.text_frame; tf.word_wrap = True
	pr = tf.paragraphs[0]; pr.text = title
	pr.font.size = Pt(30); pr.font.bold = True; pr.font.color.rgb = rgb('FFFFFF')
	box2 = s.shapes.add_textbox(Inches(0.7), Inches(3.3), Inches(8.6), Inches(0.8))
	tf2 = box2.text_frame; tf2.word_wrap = True
	pr2 = tf2.paragraphs[0]; pr2.text = subtitle
	pr2.font.size = Pt(16); pr2.font.color.rgb = rgb('A0BBDD')
	box3 = s.shapes.add_textbox(Inches(0.7), Inches(4.9), Inches(8.6), Inches(0.4))
	tf3 = box3.text_frame
	pr3 = tf3.paragraphs[0]; pr3.text = meta
	pr3.font.size = Pt(13); pr3.font.color.rgb = rgb('CADCFC')
	return s


def add_table(slide, data, x, y, w, h, header_bg=COLORS['accent'], col_widths=None, font_size=12, highlight_row=None):
	rows, cols = len(data), len(data[0])
	gt = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h))
	tbl = gt.table
	if col_widths:
		total = sum(col_widths)
		for i, cw in enumerate(col_widths):
			tbl.columns[i].width = Inches(w * cw / total)
	for r in range(rows):
		for c in range(cols):
			cell = tbl.cell(r, c)
			cell.text = str(data[r][c])
			para = cell.text_frame.paragraphs[0]
			para.font.size = Pt(font_size)
			para.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
			if r == 0:
				cell.fill.solid(); cell.fill.fore_color.rgb = rgb(header_bg)
				para.font.color.rgb = rgb('FFFFFF'); para.font.bold = True
			else:
				cell.fill.solid()
				cell.fill.fore_color.rgb = rgb('FFF2CC') if highlight_row == r else rgb('FFFFFF')
				para.font.color.rgb = rgb(COLORS['body'])
	return gt


def add_image(slide, path, x, y, w=None, h=None):
	return slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def build(lang):
	KR = lang == 'kr'
	p = new_deck()

	# 1. Title
	if KR:
		title_slide(p, "학습 기반 feasibility 판별의 실패와, CP-SAT 강화로 얻은 첫 성공",
		            "v11 — 10가지 부정 결과의 기제 진단, baseline 재검증, CP-SAT augmentation",
		            "neuroMCTS project  ·  2026년 7월")
	else:
		title_slide(p, "Why Learned Feasibility Classification Fails Here",
		            "v11 — Ten diagnosed negative results, a baseline reality check, and a first win from augmenting CP-SAT",
		            "neuroMCTS project  ·  July 2026")

	# 2. Motivation
	s = blank_slide(p)
	add_title(s, "동기: feasibility 판별을 1차 목표로" if KR else "Motivation: Feasibility Classification, Directly")
	if KR:
		items = [
			("Solution 구성은 보류 — 0/1 해가 존재하는지(feasibility)를 GNN/RL로 직접 판별", 0, True),
			("필수조건 A: 고전 솔버(MILP/CP-SAT)보다 빠른 추론", 0, False),
			("필수조건 B: 학습하지 않은 크기에서도 유지되는 zero-shot 정확도", 0, False),
			("본 사이클: 두 조건 모두 프로젝트 자신의 실험 + 문헌으로 정면 검증", 0, True),
		]
	else:
		items = [
			("Solution construction set aside — classify feasibility (does a 0/1 solution exist) directly", 0, True),
			("Requirement A: inference faster than classical MILP/CP-SAT solvers", 0, False),
			("Requirement B: accuracy that survives zero-shot transfer to untrained sizes", 0, False),
			("This cycle tests both requirements head-on, against our own experiments and the literature", 0, True),
		]
	add_bullets(s, items)
	add_footer(s, "v11", 2)

	# 3. Ten negative results scoreboard
	s = blank_slide(p)
	add_title(s, "열 가지 독립적 부정 결과" if KR else "Ten Independent Negative Results")
	add_image(s, FIG_DIR + 'v11_negative_scoreboard.png', 2.35, 1.05, h=4.1)
	add_footer(s, "v11", 3)

	# 4. Sanity check / ruling out bugs
	s = blank_slide(p)
	add_title(s, "최적화 버그 배제: positive-control sanity check" if KR else "Ruling Out Optimization Bugs")
	if KR:
		items = [
			("모든 부정 결과는 학습 전 sanity check으로 검증", 0, True),
			("이중노드 GNN: 대리 라벨(mean(b)>median)로 25 epoch만에 loss 0.693→0.607", 0, False),
			("이분그래프 GNN: pooling-보존 상수 신호로 6 epoch만에 loss 0.614→0.04", 0, False),
			("→ 그래디언트·최적화기 정상 작동. 실패는 학습 절차 문제가 아님", 0, True),
		]
	else:
		items = [
			("Every negative result checked against a sanity task before being trusted", 0, True),
			("Dual-node GNN: loss 0.693→0.607 in 25 epochs on an easy continuous proxy", 0, False),
			("Bipartite GNN: loss 0.614→0.04 in 6 epochs on a pooling-preserved signal", 0, False),
			("→ Gradients and optimizer work fine. The failure is not a training bug.", 0, True),
		]
	add_bullets(s, items)
	add_footer(s, "v11", 4)

	# 5. Root cause diagnosis
	s = blank_slide(p)
	add_title(s, "근본 원인: near-miss 짝은 표현상 동일하다" if KR else "Root Cause: Near-Miss Pairs Are Representationally Identical")
	if KR:
		items = [
			("BKZ Gram-Schmidt 프로파일 시퀀스 모델도 AUC 0.5000", 0, True),
			("진단 결과: 2,529개 near-miss 짝의 96.8%가 8회 시도 전부에서 비트 단위로 동일한 프로파일", 0, False),
			("b는 n+1개 기저 행 중 1개에만 영향 — 나머지는 전부 A(짝 간 공유)로 결정됨", 0, False),
			("→ 학습·용량 문제가 아니라, 표현 자체가 정보를 안 담고 있음(수학적 사실)", 0, True),
		]
	else:
		items = [
			("A sequence model on the raw BKZ Gram-Schmidt profile also lands at AUC 0.5000", 0, True),
			("Diagnosis: 96.8% of 2,529 near-miss pairs are bit-identical across all 8 BKZ tries", 0, False),
			("b affects only 1 of n+1 basis rows — the rest are entirely determined by A (shared)", 0, False),
			("→ Not a training/capacity issue: the representation itself carries no information", 0, True),
		]
	add_bullets(s, items)
	add_footer(s, "v11", 5)

	# 6. Literature grounding
	s = blank_slide(p)
	add_title(s, "문헌적 근거" if KR else "Literature Grounding")
	if KR:
		items = [
			("Chen et al. 2023 (ICLR): 일부 feasible/infeasible MILP 쌍은 표준 GNN이 원리적으로 구분 불가", 0, False),
			("→ 처방(랜덤 노드 특징)도 제시 — 이번 사이클에 테스트했으나 우리 구조에선 무효(d=0/8/32 전부 0.5001)", 1, False),
			("G4SATBench (TMLR 2024): SAT 분류기, 크기 4-5배 전이 시 정확도 15-25%p 붕괴", 0, False),
			("RL이 feasibility 판정 자체를 출력하는 문헌: 조사 범위 내 사실상 부재", 0, False),
		]
	else:
		items = [
			("Chen et al. 2023 (ICLR): some feasible/infeasible MILP pairs are provably GNN-indistinguishable", 0, False),
			("→ Their fix (random node features) tested this cycle, no effect here (d=0/8/32 all 0.5001)", 1, False),
			("G4SATBench (TMLR 2024): SAT classifiers lose 15-25 accuracy points under 4-5x size transfer", 0, False),
			("RL whose output is a feasibility verdict: essentially absent from the literature surveyed", 0, False),
		]
	add_bullets(s, items)
	add_footer(s, "v11", 6)

	# 7. Baseline reality check
	s = blank_slide(p)
	add_title(s, "Baseline 재검증" if KR else "Baseline Reality Check")
	add_image(s, FIG_DIR + 'v11_baseline_reality_check.png', 0.6, 1.0, w=8.8)
	add_footer(s, "v11", 7)

	# 8. Baseline table
	s = blank_slide(p)
	add_title(s, "정확도·시간: Classical Solver 전 크기 우위" if KR else "Accuracy & Time by Size")
	hdr = ["크기", "Gurobi", "SCIP", "CP-SAT", "제안 파이프라인"] if KR else ["Size", "Gurobi", "SCIP", "CP-SAT", "Ours"]
	data = [hdr,
	        ["10x25", "100.0% / 0.0023s", "100.0% / 0.0053s", "100.0% / 0.0011s", "100.0% / 0.23s"],
	        ["20x50", "100.0% / 0.028s", "100.0% / 0.141s", "100.0% / 0.031s", "78.25% / 3.78s"],
	        ["40x100", "99.4% / 1.875s", "93.2% / 9.24s", "90.9% / 8.03s", "66.67% / 22.0s"],
	        ["60x150", "—", "—", "16.67% / 17.72s", "6.67% / 107.95s"]]
	add_table(s, data, 0.5, 1.2, 9.0, 2.6, col_widths=[1, 1.6, 1.6, 1.6, 1.8], font_size=13, highlight_row=4)
	note = ("40x100까지 baseline이 최대 200배 빠르고 더 정확. 60x150은 이번에 처음 실측 — "
	        "CP-SAT도 어렵지만(2/12) 여전히 제안 파이프라인(2/30)보다 나음." if KR else
	        "Baselines up to 200x faster and more accurate through 40x100. 60x150 tested for the "
	        "first time this cycle -- hard for CP-SAT too (2/12), but still ahead of ours (2/30).")
	tb = s.shapes.add_textbox(Inches(0.5), Inches(4.1), Inches(9.0), Inches(1.0))
	tf = tb.text_frame; tf.word_wrap = True
	pr = tf.paragraphs[0]; pr.text = note; pr.font.size = Pt(14); pr.font.color.rgb = rgb(COLORS['muted'])
	add_footer(s, "v11", 8)

	# 9. CP-SAT augmentation
	s = blank_slide(p)
	add_title(s, "CP-SAT를 대체가 아닌 강화로" if KR else "Augmenting, Not Replacing, CP-SAT")
	hdr = ["변형", "Solution 정확도", "Infeasible 증명", "평균 시간"] if KR else ["Variant", "Sol. Acc.", "Infeas. Proved", "Avg Time"]
	data = [hdr,
	        ["Plain CP-SAT", "16.67% (2/12)", "5/6", "17.72s"],
	        ["+ pump hint (AddHint)", "16.67% (2/12)", "6/6", "17.70s"],
	        ["linearization_level=2", "8.33% (1/12)", "5/6", "21.22s"],
	        ["PORTFOLIO_SEARCH", "16.67% (2/12)", "5/6", "19.01s"],
	        ["LP_SEARCH", "16.67% (2/12)", "5/6", "18.00s"]]
	add_table(s, data, 0.5, 1.2, 9.0, 2.8, col_widths=[2.4, 1.6, 1.6, 1.2], font_size=13, highlight_row=2)
	note = ("Hint는 해를 더 찾진 못하지만 infeasible 증명 5/6→6/6. 파라미터 튜닝 3종은 전부 무효 — "
	        "이번 사이클 유일한 재현 가능한 긍정 결과." if KR else
	        "The hint doesn't find more solutions but closes an infeasibility proof, 5/6 -> 6/6. "
	        "All 3 parameter tweaks were neutral or worse -- the cycle's only reproducible positive result.")
	tb = s.shapes.add_textbox(Inches(0.5), Inches(4.25), Inches(9.0), Inches(1.0))
	tf = tb.text_frame; tf.word_wrap = True
	pr = tf.paragraphs[0]; pr.text = note; pr.font.size = Pt(14); pr.font.color.rgb = rgb(COLORS['muted'])
	add_footer(s, "v11", 9)

	# 10. Why learning fails - logical argument
	s = blank_slide(p)
	add_title(s, "왜 학습이 여기서 전이되지 않는가 (논리적 근거)" if KR else "Why Learning Doesn't Transfer Here")
	if KR:
		items = [
			("(i) 느린 수렴이 아니라 표현 수준의 불가능성 — 96.8%가 비트 단위로 동일한 입력", 0, True),
			("(ii) 정수 feasibility는 불연속·정수론적 성질 — parity와 같은 부류의 장애물", 0, True),
			("(iii) Market-split은 매끄러운 신호를 없애도록 설계됨 — LP완화뿐 아니라 그래디언트 하강도 무력화", 0, True),
			("스케일(깊이·폭·데이터)을 키워도 AUC가 정확히 0.5±0.0003 — 학습 곡선이 아니라 단단한 상한", 0, False),
		]
	else:
		items = [
			("(i) Representation-level impossibility, not slow convergence — 96.8% bit-identical inputs", 0, True),
			("(ii) Exact integer feasibility is discontinuous, number-theoretic — same obstruction as parity", 0, True),
			("(iii) Market-split is built to remove smooth signal -- for LP relaxation AND gradient descent alike", 0, True),
			("Scaling depth/width/data leaves AUC at exactly 0.5±0.0003 -- a hard ceiling, not a learning curve", 0, False),
		]
	add_bullets(s, items)
	add_footer(s, "v11", 10)

	# 11. Conclusion
	s = blank_slide(p)
	add_title(s, "결론 및 다음 단계" if KR else "Conclusion & Next Steps")
	if KR:
		items = [
			("Feasibility 판별: 프로젝트 실험(10건) + 문헌 양쪽 모두 검증 실패", 0, True),
			("학습의 유일한 긍정 기여: CP-SAT에 심볼릭 부산물을 hint로 제공(대체 아닌 강화)", 0, True),
			("Classical solver는 대조군이 아니라 40x100까지 지배적 방법 — 60x150에서도 우위 유력", 0, False),
			("다음: hint 방향 확장(더 큰 표본, AHL hint, 부분 hint) — 판정 대체 재시도는 보류", 0, True),
		]
	else:
		items = [
			("Feasibility classification failed both our own experiments (10 results) and the literature", 0, True),
			("Learning's one positive contribution: hinting CP-SAT with symbolic byproducts, not replacing it", 0, True),
			("Classical solvers dominate through 40x100 and likely still lead at 60x150", 0, False),
			("Next: expand the hint direction (larger samples, AHL hints, partial hints); hold off on replacement", 0, True),
		]
	add_bullets(s, items)
	add_footer(s, "v11", 11)

	return p


for lang, suffix in [('en', ''), ('kr', '_KR')]:
	deck = build(lang)
	out = f'/home/kopt/neuroMCTS/docs/pptx/LPneuroBLS_v11_deck{suffix}.pptx'
	deck.save(out)
	print('wrote', out)
