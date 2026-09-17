#!/bin/bash
# v24 Track 3, R2: waits for R1, applies the critic gate, fine-tunes the policy on exact subtree
# costs, selects the epoch by the held-out monitor (range TEST states, never the evaluation
# instances), and evaluates the pv arm on both 21x60 sets with 3 seeds.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S() { date '+%y-%m-%d %H:%M:%S'; }
R=../../runs/v24
until grep -q "V24 R1 DONE" $R/r1.log 2>/dev/null; do sleep 60; done
echo "[$(S)] ===== R2 GATE ====="
GATE=$(python3 -c "
import json
f=json.load(open('$R/critic_frozen/critic_eval.json')); u=json.load(open('$R/critic_unfrozen/critic_eval.json'))
print(f\"frozen {f['spearman']:.3f} unfrozen {u['spearman']:.3f} size-only {f['size_only_spearman']:.3f}\")
best=max(f['spearman'],u['spearman'])
print('PASS' if best>=0.5 else 'FAIL', 'UNFREEZE' if u['spearman']-f['spearman']>=0.1 else 'FROZEN')
")
echo "$GATE"
if echo "$GATE" | grep -q FAIL; then echo "[$(S)] critic gate failed -> R2 not started"; echo "[$(S)] ===== V24 R2 DONE (gate fail) ====="; exit 0; fi
UNF=""; echo "$GATE" | grep -q UNFREEZE && UNF="--unfreeze"
echo "[$(S)] ===== R2 STAGE A: fine-tune (init critic_frozen, $UNF) ====="
python3 v24_r2_finetune.py --init_pv $R/critic_frozen/pv.pt --epochs 6 --states_per_epoch 400 --n_monitor 150 --budget 90 --jobs 6 $UNF --out $R/r2
BEST=$(python3 -c "
import json; L=json.load(open('$R/r2/log.json')); b=min(L,key=lambda r:r['sum_ratio']); print(b['epoch'])")
echo "[$(S)] selected epoch $BEST by held-out total-node ratio"
cp $R/r2/pv_ep$BEST.pt $R/r2/pv_best.pt
echo "[$(S)] ===== R2 STAGE B: evaluation (pv arm, 3 seeds) ====="
cd ../v23_ablation
PV=../../runs/v22/model_n25/conditional.pt,../../runs/v24/r2/pv_best.pt
for SEED in 0 1 2; do
  python3 v23_cascade_warm.py --data_dir ../../instances/v22test_21x60 --tag C_PV_S$SEED --time_limit 600 --block 12 --tries 10 \
    --arms pv --ckpt $PV --ahl off on --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_21x60_100 --tag H_PV_S$SEED --time_limit 600 --block 12 --tries 10 \
    --arms pv --ckpt $PV --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
done
cd ../v24_deploy_range
python3 v24_aggregate.py --tags M0 M1 M2 SW PV
echo "[$(S)] ===== V24 R2 DONE ====="
