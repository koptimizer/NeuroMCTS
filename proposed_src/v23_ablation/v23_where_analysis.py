"""Where the transferred network helps (paper Table 'tab:where').

Builds in-tree states on the 21x60 evaluation instances at several numbers of free variables
(prefix of the LP-confidence order fixed to the planted solution), enumerates the exact
conditional marginals where that completes within 15 s, and compares the network's and
relaxation rounding's accuracy on FORCED variables, bucketed by free-variable count.
Recovered verbatim from the session transcript of 2026-09-15 10:52 and saved here so the
table is reproducible from the repository. Run from this directory.
"""
import sys, json
from pathlib import Path
sys.path[:0] = [str(p) for p in sorted(Path.cwd().parent.iterdir()) if p.is_dir() and not p.name.startswith('.')]
import numpy as np, torch
from v15_train_marginal import MarginalNet, featurize
import LPneuroBLS_v7 as m
from v15_bayes_limit import enumerate_solutions
torch.set_num_threads(1)
dev=torch.device('cpu'); net=MarginalNet()
net.load_state_dict(torch.load('../../runs/v22/model_n25/conditional.pt',map_location=dev,weights_only=False)['model_state_dict']); net.eval()

# 21x60 인스턴스에서 여러 n_c의 in-tree 상태를 만들어, 모델이 LP와 얼마나 다른지 + 실제 정확도
rng=np.random.default_rng(0)
buckets={}
for f in sorted(Path('../../instances/v22test_21x60').glob('*.json'))[:14]:
    d=json.load(open(f))
    A0=np.array(d['A'],np.int64); b0=np.array(d['b'],np.int64); x0=np.array(d['x'],np.int64)
    x_lp0,ok=m.solve_lp_relaxation(A0.astype(np.int32),b0.astype(np.int32))
    if not ok: continue
    order=np.argsort(-np.abs(x_lp0-0.5))
    for dep in [0,10,20,30,35,40]:            # n_c = 60,50,40,30,25,20
        fx=order[:dep]; keep=order[dep:]
        Ar=A0[:,keep]; br=b0-A0[:,fx].dot(x0[fx])
        live=Ar.sum(1)>0
        if live.sum()==0: continue
        Ar,br2=Ar[live],br[live]
        S,cok=enumerate_solutions(Ar,br2,20000,15.0)
        if not cok or len(S)==0: continue
        p_true=S.mean(0)
        x_lp,ok2=m.solve_lp_relaxation(Ar.astype(np.int32),br2.astype(np.int32))
        if not ok2: continue
        rec=dict(A=Ar.tolist(),b=br2.tolist(),x=S[0].tolist(),marginals=p_true.tolist())
        fe=featurize(rec,dev)
        if fe is None: continue
        with torch.no_grad(): pm=torch.sigmoid(net(fe['xv'],fe['xc'],fe['ev2c'],fe['ec2v'])).numpy()
        plp=np.clip(x_lp,0,1)
        forced=(p_true<1e-9)|(p_true>1-1e-9)
        if forced.sum()==0: continue
        fv=(p_true>0.5).astype(np.int64)
        k=Ar.shape[1]
        buckets.setdefault(k,[]).append((
            ((pm>=0.5).astype(np.int64)[forced]==fv[forced]).mean(),     # model FORCED 정확도
            ((plp>=0.5).astype(np.int64)[forced]==fv[forced]).mean(),    # LP FORCED 정확도
            np.corrcoef(pm,plp)[0,1]))                                    # model vs LP 상관
print("n_c 구간별: 모델이 LP를 넘어서는가 (21x60 in-tree 상태, 전수열거 가능분)")
print(f"{'n_c':>5}{'학습범위?':>10}{'상태수':>7}{'model FORCED':>14}{'LP FORCED':>11}{'차이':>8}{'model~LP 상관':>14}")
for k in sorted(buckets):
    v=np.array(buckets[k]); tag='안' if 13<=k<=25 else '밖'
    print(f"{k:>5}{tag:>9}{len(v):>7}{100*v[:,0].mean():>13.1f}%{100*v[:,1].mean():>10.1f}%{100*(v[:,0]-v[:,1]).mean():>+7.1f}%{v[:,2].mean():>13.3f}")
