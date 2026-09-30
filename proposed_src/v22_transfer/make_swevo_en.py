#!/usr/bin/env python3
"""Derives docs/tex/SWEVO_EN_v22.tex (Swarm and Evolutionary Computation, elsarticle 5p two-column)
from docs/tex/LPneuroBLS_paper_els.tex (the single-column preprint that is the English source).

The body is shared; this applies only format-level changes: document class and journal, author
block, table placement (>= 5 columns -> table*), width-fitting of tabulars, smaller verbatim,
no line numbers, one display equation split for the column width, and the long URL wrapped.
Edit the preprint, then run this; never edit the SWEVO file by hand.
"""
import io, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '../../docs/tex/detailed/LPneuroBLS_paper_els.tex')
OUT = os.path.join(HERE, '../../docs/tex/detailed/SWEVO_EN_v22.tex')

s = io.open(SRC, encoding='utf-8').read()
s = s.replace("\\documentclass[preprint,12pt]{elsarticle}", "\\documentclass[final,5p,times,twocolumn]{elsarticle}", 1)
s = s.replace("\\usepackage{lineno}\n", "", 1)
s = s.replace("\\linenumbers\n", "", 1)
s = s.replace("\\AtBeginEnvironment{table}{\\small\\setlength{\\tabcolsep}{4pt}}",
              "\\AtBeginEnvironment{table}{\\small\\setlength{\\tabcolsep}{4pt}}\n"
              "\\AtBeginEnvironment{table*}{\\small\\setlength{\\tabcolsep}{4pt}}\n"
              "% shrink a tabular only when it is wider than the available width\n"
              "\\newcommand{\\fitc}[1]{\\resizebox{\\ifdim\\width>\\columnwidth\\columnwidth\\else\\width\\fi}{!}{#1}}\n"
              "\\newcommand{\\fitw}[1]{\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{#1}}", 1)
s = s.replace("\\journal{European Journal of Operational Research}",
              "% author footnote marks as dagger symbols instead of numerals\n"
              "\\makeatletter\\def\\thefnote{\\ifcase\\c@fnote\\or$\\dagger$\\or$\\ddagger$\\fi}\\makeatother\n"
              "\\journal{Swarm and Evolutionary Computation}", 1)
old = s[s.index("\\author[1]{neuroMCTS project}"):s.index("\\begin{abstract}")]
s = s.replace(old, "\\author[ku]{Gwang-Jong Ko\\corref{cor1}}\n\\author[ku]{Taesu Cheong\\fnref{fn1}}\n\\author[ku]{In-Chan Choi\\fnref{fn1}}\n"
                   "\\cortext[cor1]{Corresponding author.}\n\\fntext[fn1]{Footnote text for the \\dag{} mark to be supplied by the authors.}\n"
                   "\\affiliation[ku]{organization={School of Industrial and Management Engineering, Korea University},\n"
                   "                city={Seoul},\n                country={Republic of Korea}}\n\n", 1)
# strip the preprint's resizebox wrappers, then re-wrap every tabular; wide tables go to table*
s = s.replace("\\resizebox{\\textwidth}{!}{\\begin{tabular}", "\\begin{tabular}")
s = re.sub(r"\\end\{tabular\}\}", "\\\\end{tabular}", s)

def fix(m):
	body = m.group(0)
	spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", body).group(1)
	wide = len(re.findall(r"[lrc]", spec)) >= 5
	body = body.replace("\\begin{tabular}", ("\\fitw{" if wide else "\\fitc{") + "\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
	if wide: body = body.replace("\\begin{table}[h]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
	else: body = body.replace("\\begin{table}[h]", "\\begin{table}[t]")
	return body

s = re.sub(r"\\begin\{table\}\[h\].*?\\end\{table\}", fix, s, flags=re.S)
s = s.replace("{\\small\\begin{verbatim}", "{\\footnotesize\\begin{verbatim}", 1)
s = s.replace("Unrefereed technical note, \\url{https://www.beren.io/2022-09-25-Deconfusing-direct-vs-amortized-optimization/}.",
              "Unrefereed technical note, \\url{https://www.beren.io/}, post \\emph{2022-09-25-Deconfusing-direct-vs-amortized-optimization}.")
old_eq = "\\begin{equation}\n\\text{propagation} \;\\to\; \\text{LP infeasibility rule} \;\\to\; \\text{kernel pump} \;\\to\; \\text{AHL/BKZ} \;\\to\; \\text{guided backtracking search}.\n\\end{equation}"
new_eq = "\\begin{equation}\n\\begin{split}\n\\text{propagation} \;\\to\; \\text{LP infeasibility rule} \;\\to\; \\text{kernel pump}\\\\\n\;\\to\; \\text{AHL/BKZ} \;\\to\; \\text{guided backtracking search}.\n\\end{split}\n\\end{equation}"
s = s.replace(old_eq, new_eq, 1)
io.open(OUT, 'w', encoding='utf-8').write(s)
print('saved', os.path.abspath(OUT))
