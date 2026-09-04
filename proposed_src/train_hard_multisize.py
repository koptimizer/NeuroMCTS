"""Multi-size expert-imitation fine-tune on the hard instance family (10x25/20x50/40x100/60x150),
warm-started from the general pretrained checkpoint, run under a wall-clock deadline (not a fixed
epoch count) so an overnight run stops on time regardless of how per-epoch cost varies with the
size mix. Builds on train_hard_finetune.py's single-size monkey-patch (Trainer.sample_fresh_feasible
calls gen_planted_instance() directly regardless of --data_dir; must be patched to draw from the
pre-generated hard feasible pools instead), extended to sample a random size per instance.

Usage:
  python train_hard_multisize.py --deadline_hours 10.5 --init_from ../runs/260710-2026_LP_Hybrid_v7/checkpoint_pretrained.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '.')
import LPneuroBLS_v7 as m

SIZES = ['10x25', '20x50', '40x100', '60x150']


def load_hard_feasible_pool(train_dir):
	pool = []
	for f in sorted(Path(train_dir).glob('*.json')):
		d = json.load(open(f))
		if d.get('feasible', True):
			pool.append((np.array(d['A'], dtype=np.int8), np.array(d['b'], dtype=np.int32), np.array(d['x'], dtype=np.int8)))
	return pool


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--train_n', type=str, default='10000,10000,10000,1500',
	                 help='comma-separated pool size suffix per size in SIZES order')
	ap.add_argument('--init_from', required=True)
	ap.add_argument('--deadline_hours', type=float, required=True)
	ap.add_argument('--samples_per_epoch', type=int, default=512)
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--save_dir', default='../runs')
	ap.add_argument('--checkpoint_every', type=int, default=10)
	a = ap.parse_args()

	train_ns = a.train_n.split(',')
	train_dirs = {}
	pools = {}
	for sz, tn in zip(SIZES, train_ns):
		d = f"../instances/train_instances_hard_{sz}_{tn}"
		train_dirs[sz] = d
		pool = load_hard_feasible_pool(d)
		pools[sz] = pool
		print(f"{sz}: loaded {len(pool)} hard feasible instances from {d}", flush=True)

	class Args:
		pass
	args = Args()
	args.data_dir = list(train_dirs.values())
	args.seed = 0
	args.dfs_epochs = 0
	args.feas_epochs = 0
	args.imitation_epochs = 0
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
	args.save_interval = 10**9  # disable the method's own checkpointing; we save explicitly below
	args.batch_size = 256
	args.device = a.device
	args.init_from = None

	trainer = m.Trainer(args)
	print(f"sizes seen by infeasible pool: {trainer.sizes}", flush=True)

	rng = np.random.default_rng(0)

	def sample_fresh_feasible_multisize(size=None):
		sz = SIZES[int(rng.integers(len(SIZES)))]
		pool = pools[sz]
		A, b, x = pool[int(rng.integers(len(pool)))]
		return trainer.featurize(A, b, x, True)

	trainer.sample_fresh_feasible = sample_fresh_feasible_multisize

	state = torch.load(a.init_from, map_location=trainer.device, weights_only=False)
	if isinstance(state, dict) and 'model_state_dict' in state:
		state = state['model_state_dict']
	trainer.model.load_state_dict(state)
	print(f"warm-started from {a.init_from}", flush=True)

	deadline = time.time() + a.deadline_hours * 3600
	global_epoch = 0
	t_start = time.time()
	while time.time() < deadline:
		trainer.run_expert_imitation(1)
		global_epoch += 1
		elapsed_h = (time.time() - t_start) / 3600
		remaining_h = (deadline - time.time()) / 3600
		print(f"[multisize] global_epoch={global_epoch}  elapsed={elapsed_h:.2f}h  remaining={remaining_h:.2f}h", flush=True)
		if global_epoch % a.checkpoint_every == 0:
			ckpt_path = Path(trainer.save_dir) / f"checkpoint_hard_multisize_ep{global_epoch}.pt"
			torch.save({'model_state_dict': trainer.model.state_dict(), 'global_epoch': global_epoch}, ckpt_path)
			print(f"[multisize] periodic checkpoint saved: {ckpt_path}", flush=True)

	final_path = Path(trainer.save_dir) / "checkpoint_hard_multisize_final.pt"
	torch.save({'model_state_dict': trainer.model.state_dict(), 'global_epoch': global_epoch}, final_path)
	print(f"[multisize] DONE: {global_epoch} epochs in {(time.time()-t_start)/3600:.2f}h -> {final_path}", flush=True)


if __name__ == '__main__':
	main()
