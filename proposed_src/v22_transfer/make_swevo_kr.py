#!/usr/bin/env python3
"""Emits docs/tex/SWEVO_KR_v22.tex (elsarticle, 5p two-column) from the same Korean content that
make_paper_kr.py renders with reportlab, so the two Korean editions cannot drift apart.

The generator body is executed with P/H1/H2/EQ/TBL replaced by LaTeX emitters and the
reportlab markup (<b>, <i>, <sub>, <sup>, <br/>, &nbsp;) translated. Table captions keep
the generator's own numbering (there is a table 4b), so they are typeset by hand rather
than with \\caption. Compiling needs xelatex + xeCJK (or lualatex + luaotfload) and the
Noto Sans KR font; this machine has neither TeX engine set up, so the file is emitted and
syntax-checked only.
"""
import io, re, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'make_paper_kr.py')
OUT = os.path.join(HERE, '../../docs/tex/SWEVO_KR_v22.tex')

src = io.open(SRC, encoding='utf-8').read()
body = src[src.index('# ═══════════════════════ 표지'):src.index('\ndoc = SimpleDocTemplate')]

TAG = {'<b>': '\\textbf{', '</b>': '}', '<i>': '\\textit{', '</i>': '}',
       '<sub>': '\\textsubscript{', '</sub>': '}', '<sup>': '\\textsuperscript{', '</sup>': '}',
       '</font>': '}'}

def esc(t):
	t = t.replace('\\', '\\textbackslash{}')
	t = t.replace('&nbsp;', '~').replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
	for a, b in [('{', '\\{'), ('}', '\\}'), ('%', '\\%'), ('&', '\\&'), ('#', '\\#'), ('_', '\\_'),
	             ('$', '\\$'), ('^', '\\^{}'), ('~', '\\textasciitilde{}')]:
		t = t.replace(a, b)
	return t.replace('\\textasciitilde\\{\\}', '~')  # &nbsp; became ~ before escaping

def tx(t, br='\\\\ '):
	out = []
	for piece in re.split(r'(<[^>]+>)', t):
		if not piece: continue
		if piece.startswith('<'):
			if piece == '<br/>': out.append(br)
			elif piece.startswith('<font'): out.append('{\\ttfamily\\footnotesize ')
			else: out.append(TAG[piece])
		else:
			out.append(esc(piece))
	return ''.join(out)

class Preformatted:
	def __init__(self, t, s): self.t = t
class PageBreak: pass
def P(t, s='body'): E.append(('P', s, t))
def H1(t): E.append(('H1', t))
def H2(t): E.append(('H2', t))
def EQ(t): E.append(('EQ', t))
def TBL(data, widths, caption=None, right_from=1, font=8.5): E.append(('TBL', data, caption, right_from))
class _E2(list):
	def append(self, x): super().append(('CODE', x.t) if isinstance(x, Preformatted) else x)
E = _E2()
ns = dict(P=P, H1=H1, H2=H2, EQ=EQ, TBL=TBL, E=E, Preformatted=Preformatted, PageBreak=PageBreak,
          title_s='title', meta_s='meta', abs_s='abs', body_s='body', prop_s='prop', code_s='code', ref_s='ref', cm=1.0)
exec(body, ns)
items = [x for x in E if not isinstance(x, PageBreak)]

L = []
abstract, refs = [], []
mode = None
for it in items:
	k = it[0]
	if k == 'H1':
		t = it[1]
		if t == '초록': mode = 'abs'; continue
		if t == '참고문헌': mode = 'ref'; continue
		mode = None
		m = re.match(r'\d+\.\s*(.*)', t)
		L.append('\\section{%s}' % tx(m.group(1)) if m else '\\section*{%s}' % tx(t))
	elif k == 'H2':
		m = re.match(r'\d+\.\d+\s*(.*)', it[1])
		L.append('\\subsection{%s}' % tx(m.group(1) if m else it[1]))
	elif k == 'P':
		s, t = it[1], it[2]
		if s in ('title', 'meta'): continue
		if s == 'abs' or mode == 'abs': abstract.append(tx(t)); continue
		if s == 'ref' or mode == 'ref': refs.append(tx(t)); continue
		if s == 'prop':
			L.append('\\noindent\\fbox{\\parbox{\\dimexpr\\columnwidth-2\\fboxsep-2\\fboxrule\\relax}{\\small %s}}\\par\\medskip' % tx(t))
		else:
			L.append(tx(t) + '\n')
	elif k == 'EQ':
		L.append('\\begin{center}\\small %s\\end{center}' % tx(it[1]))
	elif k == 'CODE':
		L.append('{\\footnotesize\\begin{verbatim}\n%s\n\\end{verbatim}}' % it[1])
	elif k == 'TBL':
		data, cap, rf = it[1], it[2], it[3]
		n = len(data[0]); wide = n >= 5
		spec = 'l' * rf + 'r' * (n - rf)
		rows = [' & '.join(tx(c, br=' ') for c in r) + ' \\\\' for r in data]
		env = 'table*' if wide else 'table'; fit = '\\fitw' if wide else '\\fitc'
		L.append('\\begin{%s}[t]\\centering\n%s{\\begin{tabular}{%s}\n\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}}' % (
			env, fit, spec, rows[0], '\n'.join(rows[1:])))
		if cap: L.append('\\par\\vspace{3pt}{\\footnotesize %s\\par}' % tx(cap))
		L.append('\\end{%s}\n' % env)

PRE = r"""\documentclass[final,5p,twocolumn]{elsarticle}
% Korean edition. Compile with xelatex (xeCJK) or lualatex (luaotfload); needs Noto Sans KR.
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{etoolbox}
\usepackage{fontspec}
\IfFileExists{xeCJK.sty}{\usepackage{xeCJK}\setCJKmainfont{Noto Sans KR}\xeCJKsetup{CJKspace=true}}{}
\setmainfont{Noto Sans KR}
\AtBeginEnvironment{table}{\small\setlength{\tabcolsep}{4pt}}
\AtBeginEnvironment{table*}{\small\setlength{\tabcolsep}{4pt}}
\newcommand{\fitc}[1]{\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{#1}}
\newcommand{\fitw}[1]{\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{#1}}
% author footnote marks as dagger symbols instead of numerals
\makeatletter\def\thefnote{\ifcase\c@fnote\or$\dagger$\or$\ddagger$\fi}\makeatother
\journal{Swarm and Evolutionary Computation}

\begin{document}
\begin{frontmatter}

\title{Binary linear system을 위한 amortized deduction:\\
정확한 conditional marginal로부터 branching guidance를 학습하기\tnoteref{t1}}
\tnotetext[t1]{영문 제목: Amortized Deduction for Binary Linear Systems: Learning Branching Guidance from Exact Conditional Marginals.}

\author[ku]{Gwang-Jong Ko\corref{cor1}}
\author[ku]{Taesu Cheong\fnref{fn1}}
\author[ku]{In-Chan Choi\fnref{fn1}}
\cortext[cor1]{교신저자.}
\fntext[fn1]{\dag{} 표시 각주 내용 기재 필요.}
\affiliation[ku]{organization={School of Industrial and Management Engineering, Korea University},
                city={Seoul},
                country={Republic of Korea}}

\begin{abstract}
@@ABSTRACT@@
\end{abstract}

\begin{keyword}
Binary linear systems \sep Learning to branch \sep Amortized optimization \sep Exact conditional marginals \sep Market split \sep Graph neural networks
\end{keyword}

\end{frontmatter}

"""
doc = PRE.replace('@@ABSTRACT@@', '\n\n'.join(abstract))
doc += '\n'.join(L)
doc += '\n\\section*{참고문헌}\n{\\small\n' + '\n'.join('\\noindent\\hangindent=1em\\hangafter=1 %s\\par\\smallskip' % r for r in refs) + '}\n\n\\end{document}\n'
io.open(OUT, 'w', encoding='utf-8').write(doc)
print('saved', os.path.abspath(OUT), 'abstract paras', len(abstract), 'refs', len(refs), 'blocks', len(L))
