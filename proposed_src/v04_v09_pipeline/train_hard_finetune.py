# Layout bootstrap: this file lives in a per-cycle folder, so make util/ and the
# sibling cycle folders importable. Paths like '../../instances' resolve from here.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]

#!/usr/bin/env python3
"""
Adapts the proposed pipeline's GNN to the hard instance family. Two things
were confirmed necessary before this could work at all:
  1. Trainer.sample_fresh_feasible() calls gen_planted_instance() directly
     (the natural/easy generator) regardless of --data_dir -- --data_dir
     only feeds the infeasible pool. Pointing it at the hard directory alone
     would train on natural feasible instances with only the infeasible
     labels swapped in, silently defeating the point.
  2. Generating hard feasible instances on the fly (K=20 vertex_spread
     selection) costs ~20 x 10-25ms per instance -- multiple seconds per
     training batch, making from-scratch training impractical in this
     session's time budget.

Both are solved by monkey-patching sample_fresh_feasible to draw from the
already-generated hard feasible pool (train_instances_hard_*, the 8,000
feasible instances per size) instead of generating fresh -- paid once at
load time, not every epoch.

Rather than retrain from scratch (pretrain_epochs=200 by default), this
warm-starts from the project's existing pretrained checkpoint and runs a
bounded expert-imitation fine-tune (the "EXIL" phase, matching how the
original checkpoint's own later stages were produced) on hard data only.

Usage:
  python train_hard_finetune.py --size 10x25 --init_from ../../runs/260710-2026_LP_Hybrid_v7/checkpoint_pretrained.pt --imitation_epochs 20
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '.')
import LPneuroBLS_v7 as m


def load_hard_feasible_pool(train_dir):
	pool = []
	for f in sorted(Path(train_dir).glob('*.json')):
		d = json.load(open(f))
		if d.get('feasible', True):
			pool.append((np.array(d['A'], dtype=np.int8), np.array(d['b'], dtype=np.int32), np.array(d['x'], dtype=np.int8)))
	return pool


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--size', required=True)
	ap.add_argument('--train_n', type=int, default=10000)
	ap.add_argument('--init_from', required=True)
	ap.add_argument('--imitation_epochs', type=int, default=20)
	ap.add_argument('--samples_per_epoch', type=int, default=512)
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--save_dir', default='../../runs')
	a = ap.parse_args()
	mm, nn = (int(v) for v in a.size.split('x'))

	train_dir = f"../../instances/train_instances_hard_{mm}x{nn}_{a.train_n}"
	pool = load_hard_feasible_pool(train_dir)
	print(f"loaded {len(pool)} hard feasible instances from {train_dir}")

	class Args:
		pass
	args = Args()
	args.data_dir = [train_dir]
	args.seed = 0
	args.dfs_epochs = 0
	args.feas_epochs = 0
	args.imitation_epochs = a.imitation_epochs
	args.size_range = None
	args.dfs_samples = 128
	args.dfs_budget_train = 3000
	args.save_dir = a.save_dir
	args.pretrain_epochs = 0
	args.pretrain_minimum_epoch = 20
	args.pretrain_patience_limit = 30
	args.il_epochs = 0
	args.rl_epochs = 0
	args.samples_per_epoch = a.samples_per_epoch
	args.num_simulations = 100
	args.hidden_dim = 256
	args.num_layers = 8
	args.lr = 0.0005
	args.lr_decay_step = 30
	args.lr_decay_gamma = 0.5
	args.entropy_coef = 0.01
	args.save_interval = 5
	args.batch_size = 256
	args.device = a.device
	args.init_from = None  # loaded manually below (Trainer.__init__ needs infeasible_pool first)

	trainer = m.Trainer(args)

	rng = np.random.default_rng(0)

	def sample_fresh_feasible_hard(size=None):
		A, b, x = pool[int(rng.integers(len(pool)))]
		return trainer.featurize(A, b, x, True)

	trainer.sample_fresh_feasible = sample_fresh_feasible_hard

	state = torch.load(a.init_from, map_location=trainer.device, weights_only=False)
	if isinstance(state, dict) and 'model_state_dict' in state:
		state = state['model_state_dict']
	trainer.model.load_state_dict(state)
	print(f"warm-started from {a.init_from}")

	trainer.run_expert_imitation(a.imitation_epochs)

	final_path = trainer.save_dir / f"checkpoint_hard_finetuned_{a.size}.pt"
	torch.save({'model_state_dict': trainer.model.state_dict()}, final_path)
	print(f"saved final checkpoint to {final_path}")


if __name__ == '__main__':
	main()
