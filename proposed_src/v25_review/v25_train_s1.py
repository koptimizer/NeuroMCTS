"""v25 F7 / B2: Stage-1 training with an explicit seed (v17_train_conditional.py fixes
torch.manual_seed(0) and leaves numpy unseeded). Runs v17's main() with the given argv after
seeding both generators, from proposed_src/v17_conditional so that its relative paths hold.

  python3 v25_train_s1.py --seed 1 -- --train ../../runs/v22/cond_10x25_train.json ... --out ../../runs/v25/s1_seed1
"""
import os, runpy, sys
import numpy as np
import torch

if __name__ == '__main__':
	i = sys.argv.index('--')
	seed = int(sys.argv[sys.argv.index('--seed') + 1])
	rest = sys.argv[i + 1:]
	np.random.seed(seed)
	_orig = torch.manual_seed
	torch.manual_seed = lambda s=None: _orig(seed)         # v17 calls torch.manual_seed(0) once; redirect it
	os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'v17_conditional'))
	sys.argv = ['v17_train_conditional.py'] + rest
	runpy.run_path('v17_train_conditional.py', run_name='__main__')
