"""Paired comparison between a v24 arm directory (inst-*_r.json) and a v25 arm directory (inst-*.json)
on commonly solved instances: median per-instance ratio, bootstrap CI, sign test, PAR-2.
Usage: v25_cross_compare.py <budget> <dirA> <dirB> [<dirA> <dirB> ...]   (ratio = A / B)
"""
import sys
sys.path.insert(0, '.')
from v25_aggregate import load, compare


def main():
	budget = float(sys.argv[1]); ds = sys.argv[2:]
	for dA, dB in zip(ds[::2], ds[1::2]):
		A = load(dA, not dA.rstrip('/').split('/')[-1].count('ahl')); B = load(dB, not dB.rstrip('/').split('/')[-1].count('ahl'))
		for key in ['nodes', 'sec']:
			c = compare(A, B, budget, key)
			print(f"{dA.split('/')[-1]} / {dB.split('/')[-1]} {key}: common {c['common']} solved {c['solved_A']}/{c['solved_B']} median {c['median_A']:.0f} vs {c['median_B']:.0f} | median ratio {c['median_ratio']:.2f}x CI[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] A larger {c['wins_A_larger']}/{c['common']-c['ties']} p={c['p_sign']:.2g}" + (f" | PAR-2 {c['par2_A']:.0f} vs {c['par2_B']:.0f}" if key == 'sec' else ''))


if __name__ == '__main__':
	main()
