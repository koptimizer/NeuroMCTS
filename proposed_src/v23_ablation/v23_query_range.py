"""Free-variable count (n_c) of the states the deployed search actually queries the network on,
against the n_c range of the training states. Recovered verbatim from the session transcript
of 2026-09-15 10:47 (it drives the original SCIP-backed loop, v20_guided_search, on the first 8
evaluation instances at 21x60 for 60 s each). Source of the "80.2% of queries above the
training range" statement. Run from this directory.
"""
import sys, json, time
from pathlib import Path
sys.path[:0] = [str(p) for p in sorted(Path.cwd().parent.iterdir()) if p.is_dir() and not p.name.startswith('.')]
import numpy as np, torch
from v15_train_marginal import MarginalNet, featurize
import LPneuroBLS_v7 as m
import v20_guided_search as G
torch.set_num_threads(1)

# 학습 상태의 n_c 분포
tr=json.load(open('../../runs/v22/cond_10x25_train.json'))
nc_train=np.array([len(r['marginals']) for r in tr])

# 추론 시 model이 실제로 질의받는 n_c 분포를 계측
seen=[]
orig=G.guidance_model
def spy(A,b,rng,model,device):
    seen.append(A.shape[1]); return orig(A,b,rng,model,device)
G.GUIDES['model']=spy

dev=torch.device('cpu'); net=MarginalNet()
net.load_state_dict(torch.load('../../runs/v22/model_n25/conditional.pt',map_location=dev,weights_only=False)['model_state_dict']); net.eval()
for f in sorted(Path('../../instances/v22test_21x60').glob('*.json'))[:8]:
    d=json.load(open(f)); A=np.array(d['A'],np.int64); b=np.array(d['b'],np.int64)
    G.search(A,b,G.GUIDES['model'],net,dev,60.0,0)
nc_inf=np.array(seen)

print("model이 보는 자유변수 수(n_c) 분포")
print(f"  학습 상태 ({len(nc_train):,}개): 범위 [{nc_train.min()}, {nc_train.max()}]  중앙 {np.median(nc_train):.0f}")
print(f"  추론 질의 ({len(nc_inf):,}회): 범위 [{nc_inf.min()}, {nc_inf.max()}]  중앙 {np.median(nc_inf):.0f}")
lo,hi=nc_train.min(),nc_train.max()
inside=((nc_inf>=lo)&(nc_inf<=hi)).mean()
print(f"\n  추론 질의 중 학습 범위 [{lo},{hi}] 안에 드는 비율: {100*inside:.1f}%")
print(f"    n_c > {hi} (학습보다 큼): {100*(nc_inf>hi).mean():5.1f}%")
print(f"    n_c < {lo} (학습보다 작음): {100*(nc_inf<lo).mean():5.1f}%")
