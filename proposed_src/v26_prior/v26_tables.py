"""v26 result tables from runs/v26 (and the most-fractional arm in runs/v25): SCIP plugin blocks per
(set, config), local-search blocks, end-to-end blocks. Writes runs/v26/tables.md.
"""
import glob, json, os, re
import numpy as np

R = '../../runs/v26'
BUD = dict(A=60, B=300, C=600, H=600, G70=1200, G80=1200)


def load(d):
	return [json.load(open(f)) for f in glob.glob(d + '/inst-*.json')]


def row(name, rs, budget, extra=''):
	sol = [r for r in rs if r['solved']]
	mn = f"{np.median([r['nodes'] for r in sol]):,.0f}" if sol and 'nodes' in sol[0] else '—'
	ms = f"{np.median([r['sec'] for r in sol]):.2f}" if sol else '—'
	par = np.mean([r['sec'] if r['solved'] else 2 * budget for r in rs])
	return f"| {name} | {len(sol)}/{len(rs)} | {mn} | {ms} | {par:.1f} |{extra}"


def main():
	L = []
	groups = {}
	for d in sorted(glob.glob(R + '/SCIP_*')):
		m = re.match(r'.*/SCIP_(\w+?)_S(\d)_(c\dp\dh\d)_(.+)$', d)
		if m: groups.setdefault((m.group(1), m.group(2), m.group(3)), []).append((m.group(4), d))
	for (tag, seed, cfg), items in sorted(groups.items()):
		L.append(f"\n## SCIP {tag} seed {seed} {cfg}  (budget {BUD.get(tag, 600)}s; cfg = conflict/presolve/heuristics)\n")
		L.append("| rule | solved | median nodes | median sec | PAR-2 | callback share |"); L.append("|---|---|---|---|---|---|")
		for rule, d in items:
			rs = load(d); sol = [r for r in rs if r['solved']]
			cb = f" {np.median([r['br_sec'] / max(r['sec'], 1e-9) for r in sol]):.2f} |" if sol and sol[0].get('br_sec') is not None else ' — |'
			L.append(row(rule, rs, BUD.get(tag, 600), cb))
	groups = {}
	for d in sorted(glob.glob(R + '/LS_*')):
		m = re.match(r'.*/LS_(\w+?)_S(\d)_(\w+)$', d)
		if m: groups.setdefault((m.group(1), m.group(2)), []).append((m.group(3), d))
	for (tag, seed), items in sorted(groups.items()):
		L.append(f"\n## Local search {tag} seed {seed}\n"); L.append("| init | solved | median flips | median sec | PAR-2 |"); L.append("|---|---|---|---|---|")
		for kind, d in items:
			rs = load(d); sol = [r for r in rs if r['solved']]
			L.append(f"| {kind} | {len(sol)}/{len(rs)} | {np.median([r['flips'] for r in sol]) if sol else float('nan'):,.0f} | {np.median([r['sec'] for r in sol]) if sol else float('nan'):.2f} | {np.mean([r['sec'] if r['solved'] else 2 * BUD.get(tag, 600) for r in rs]):.1f} |")
	groups = {}
	for d in sorted(glob.glob(R + '/E2E_*')):
		m = re.match(r'.*/E2E_(\w+?)_S(\d)_(\w+)$', d)
		if m: groups.setdefault((m.group(1), m.group(2)), []).append((m.group(3), d))
	for (tag, seed), items in sorted(groups.items()):
		L.append(f"\n## End-to-end {tag} seed {seed}\n"); L.append("| method | solved | median sec | PAR-2 |"); L.append("|---|---|---|---|")
		for mth, d in items:
			rs = load(d); sol = [r for r in rs if r['solved']]
			L.append(f"| {mth} | {len(sol)}/{len(rs)} | {np.median([r['sec'] for r in sol]) if sol else float('nan'):.2f} | {np.mean([r['sec'] if r['solved'] else 2 * BUD.get(tag, 600) for r in rs]):.1f} |")
	open(R + '/tables.md', 'w').write('\n'.join(L)); print('\n'.join(L))


if __name__ == '__main__':
	main()
