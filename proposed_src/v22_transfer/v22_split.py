"""Deterministic 80/20 train/test split of the enumerated instance pools (full_{size}.json).

Recovered verbatim from the session transcript of 2026-09-10 17:56: random.seed(0), shuffle,
first 80% train. The evaluation directories instances/v22test_10x25 and _18x50 are the first
30 records of the resulting test split; 21x60 is generated separately (v22_make_testset.py)
with disjoint seeds. Run from this directory.
"""
import json, random

for size in ['10x25', '18x50']:
	r = json.load(open(f'../../runs/v22/full_{size}.json'))
	random.seed(0)
	random.shuffle(r)
	ntr = int(len(r) * 0.8)
	json.dump(r[:ntr], open(f'../../runs/v22/full_{size}_train.json', 'w'))
	json.dump(r[ntr:], open(f'../../runs/v22/full_{size}_test.json', 'w'))
	print(f'{size}: train={ntr} test={len(r)-ntr}')
