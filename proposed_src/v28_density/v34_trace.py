"""RQ3 sub-question "what did Stage 2 learn": refutation structure of the search per branching rule.
From the per-instance records (path_len = decisions on the solution path, dead_first = (depth, cost)
of every path decision whose first child was refuted, refuted_nodes = nodes in refuted subtrees):
  refuted share           refuted_nodes / nodes
  first-child failure rate  len(dead_first) / path_len   (how often the value choice was wrong)
  mean refutation cost    mean cost of a refuted first child  (how expensive a wrong value was)
  refutation depth        median depth of refuted first children, as a fraction of path length
A rule that branches fail-first should refute wrong values early (shallow, cheap).
Usage: v34_trace.py --dirs <label>=<dir> ... [--out ../../runs/v34/trace.md]
"""
import argparse, glob, json
import numpy as np


def stats(d):
	rs = [json.load(open(f)) for f in glob.glob(d + '/inst-*.json')]
	rs = [r for r in rs if r['solved'] and r.get('path_len')]
	if not rs: return None
	share = np.median([r['refuted_nodes'] / max(r['nodes'], 1) for r in rs])
	fail = np.median([r['n_refuted'] / max(r['path_len'], 1) for r in rs])
	cost = np.median([np.mean([c for _, c in r['dead_first']]) if r['dead_first'] else 0 for r in rs])
	depth = np.nanmedian([np.median([dd / max(r["path_len"], 1) for dd, _ in r["dead_first"]]) if r["dead_first"] else np.nan for r in rs])
	return len(rs), np.median([r['nodes'] for r in rs]), np.median([r['path_len'] for r in rs]), share, fail, cost, depth


def main():
	ap = argparse.ArgumentParser(); ap.add_argument('--dirs', nargs='+', required=True); ap.add_argument('--title', default=''); ap.add_argument('--out', default=None)
	a = ap.parse_args()
	L = [f'## {a.title}', '', '| rule | n | median nodes | path length | refuted share | first-child failure rate | mean refutation cost | refutation depth (frac of path) |', '|---|---|---|---|---|---|---|---|']
	for item in a.dirs:
		lab, d = item.split('=', 1); s = stats(d)
		if s: L.append(f'| {lab} | {s[0]} | {s[1]:,.0f} | {s[2]:.0f} | {s[3]:.3f} | {s[4]:.3f} | {s[5]:,.0f} | {s[6]:.2f} |')
	txt = '\n'.join(L); print(txt)
	if a.out: open(a.out, 'a').write(txt + '\n\n')


if __name__ == '__main__':
	main()
