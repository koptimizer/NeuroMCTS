"""v24: training-set variants that differ ONLY in state distribution.

  M1  deployment-range 21x60 states only
  M2  half 10x25 in-tree states (the v23 distribution), half 21x60 range states
  P06 / P612 / P012  10x25 states restricted to depth 0-6 / 6-12 / 0-12, equal count
      (the pre-diagnostic: how sensitive is in-tree accuracy to the depth range at all?)
Every file is written with the same record schema, so v17_train_conditional.py trains on
each unchanged; architecture, optimizer, epochs and loss are held fixed by construction.
"""
import argparse
import json
import random


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--cond25', default='../../runs/v22/cond_10x25_train.json')
	ap.add_argument('--range', dest='rng', default='../../runs/v24/cond_21x60_range_train.json')
	ap.add_argument('--n_prediag', type=int, default=5000)
	ap.add_argument('--out_dir', default='../../runs/v24')
	a = ap.parse_args()
	random.seed(0)
	c25 = json.load(open(a.cond25))
	rg = json.load(open(a.rng))
	random.shuffle(c25); random.shuffle(rg)
	n_mix = min(len(rg), 5000)
	json.dump(rg, open(f'{a.out_dir}/train_M1.json', 'w'))
	json.dump(c25[:n_mix] + rg[:n_mix], open(f'{a.out_dir}/train_M2.json', 'w'))
	lo = [r for r in c25 if r['depth'] <= 6]
	hi = [r for r in c25 if r['depth'] >= 6]
	json.dump(lo[:a.n_prediag], open(f'{a.out_dir}/train_P06.json', 'w'))
	json.dump(hi[:a.n_prediag], open(f'{a.out_dir}/train_P612.json', 'w'))
	json.dump(c25[:a.n_prediag], open(f'{a.out_dir}/train_P012.json', 'w'))
	print(f"M1={len(rg)}  M2={2*n_mix} ({n_mix}+{n_mix})  P06={min(len(lo),a.n_prediag)} "
	      f"P612={min(len(hi),a.n_prediag)} P012={a.n_prediag}")


if __name__ == '__main__':
	main()
