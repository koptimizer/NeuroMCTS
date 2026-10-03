"""v26 E1: an NSNet-style marginal network (Li & Si 2022) trained on the same exact in-tree
marginals as Stage 1, to test whether the BP-structured architecture learns the marginal better
than the bipartite GAT. Messages live on the edges of the variable-factor graph and are updated
like belief propagation (each outgoing message excludes the incoming message on the same edge);
the readout aggregates factor-to-variable messages per variable. Parameter count is matched to
MarginalNet (~20k). Same forward signature as MarginalNet, so it drops into v25_search.
Usage: v26_nsnet.py --train ... --test ... --out ... [--seed N]
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_p) for _p in sorted(_Path(__file__).resolve().parents[1].iterdir())
                 if _p.is_dir() and not _p.name.startswith('.')]
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class NSNetMarginal(nn.Module):
	"""BP-structured edge message passing on the variable/row factor graph, one logit per variable."""

	def __init__(self, hidden=48, iters=8):
		super().__init__()
		self.h = hidden; self.iters = iters
		self.var_in = nn.Linear(3, hidden); self.con_in = nn.Linear(3, hidden)
		self.msg_fv = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
		self.msg_vf = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
		self.ln_fv = nn.LayerNorm(hidden); self.ln_vf = nn.LayerNorm(hidden)
		self.read = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

	def forward(self, xv, xc, ev2c, ec2v):
		vi, ci = ev2c[0], ev2c[1]                       # edge e = (variable vi[e], row ci[e])
		n, mm, E = xv.shape[0], xc.shape[0], vi.shape[0]
		hv = self.var_in(xv); hc = self.con_in(xc)
		m_vf = torch.zeros(E, self.h); m_fv = torch.zeros(E, self.h)
		for _ in range(self.iters):
			tot = torch.zeros(mm, self.h).index_add_(0, ci, m_vf)            # all variable->row messages at each row
			m_fv = self.ln_fv(self.msg_fv(torch.cat([tot[ci] - m_vf, hc[ci]], -1)) + m_fv)   # exclude the edge's own message
			tot = torch.zeros(n, self.h).index_add_(0, vi, m_fv)
			m_vf = self.ln_vf(self.msg_vf(torch.cat([tot[vi] - m_fv, hv[vi]], -1)) + m_vf)
		agg = torch.zeros(n, self.h).index_add_(0, vi, m_fv)
		return self.read(torch.cat([agg, hv], -1)).squeeze(-1)


def main():
	from v15_train_marginal import featurize
	from v17_train_conditional import evaluate_by_depth
	ap = argparse.ArgumentParser()
	ap.add_argument('--train', required=True); ap.add_argument('--test', required=True)
	ap.add_argument('--n_train', type=int, default=10000); ap.add_argument('--n_test', type=int, default=2000)
	ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
	a = ap.parse_args()
	torch.manual_seed(a.seed); np.random.seed(a.seed)
	device = torch.device('cpu')
	tr_recs = json.load(open(a.train))[:a.n_train]; te_recs = json.load(open(a.test))[:a.n_test]
	tr = [f for f in (featurize(r, device) for r in tr_recs) if f is not None]
	te_pairs = [(f, r) for f, r in ((featurize(r, device), r) for r in te_recs) if f is not None]
	te, te_recs = [p[0] for p in te_pairs], [p[1] for p in te_pairs]
	model = NSNetMarginal()
	print(f"train={len(tr)} test={len(te)} params={sum(p.numel() for p in model.parameters())}", flush=True)
	opt = torch.optim.Adam(model.parameters(), lr=a.lr)
	for ep in range(1, a.epochs + 1):
		tot = 0.0; t0 = time.time()
		for i in np.random.permutation(len(tr)):
			f = tr[i]
			loss = F.binary_cross_entropy_with_logits(model(f['xv'], f['xc'], f['ev2c'], f['ec2v']), f['soft'])
			opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
			tot += loss.item()
		print(f"  ep{ep:>3} loss={tot/len(tr):.4f} ({time.time()-t0:.0f}s)", flush=True)
	rows = evaluate_by_depth(model, te, te_recs, None)
	print(f"{'depth':>6}{'states':>8}{'ceiling':>10}{'nsnet':>9}{'LP':>8}{'L1':>8}")
	for d in sorted(rows):
		e = rows[d]
		print(f"{d:>6}{len(e['ceil']):>8}{100*np.mean(e['ceil']):>9.1f}%{100*np.mean(e['model']):>8.1f}%{100*np.mean(e['lp']) if e['lp'] else float('nan'):>7.1f}%{np.mean(e['l1']):>8.3f}")
	allc = np.mean([v for e in rows.values() for v in e['ceil']]); allm = np.mean([v for e in rows.values() for v in e['model']])
	print(f"{'ALL':>6}{'':>8}{100*allc:>9.1f}%{100*allm:>8.1f}%")
	out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
	torch.save({'model_state_dict': model.state_dict(), 'arch': 'nsnet'}, out / 'conditional.pt')
	json.dump({str(d): {k: float(np.mean(v)) for k, v in e.items() if v} for d, e in rows.items()}, open(out / 'v17_by_depth.json', 'w'), indent=2)
	print('saved', out / 'conditional.pt', flush=True)


if __name__ == '__main__':
	main()
