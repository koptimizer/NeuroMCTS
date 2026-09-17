#!/bin/bash
# v24 follow-up: does the cost-aware fine-tuning transfer UPWARD in size? Fine-tune at 18x50 only
# (states with 35-50 free variables from the 18x50 training pool), evaluate at 21x60 (30 and 100
# instances) and 18x50; plus a probe at 24x70 (12 instances, |S| unverified) for lp / M0 / PV60 / PV50.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S() { date '+%y-%m-%d %H:%M:%S'; }
R=../../runs/v24; N25=../../runs/v22/model_n25/conditional.pt
echo "[$(S)] ===== UP STAGE 1: 18x50 range states ====="
python3 - <<'PY'
import json; r=json.load(open('../../runs/v22/full_18x50_train.json')); json.dump(r[:500], open('../../runs/v24/pool_18x50_train.json','w'))
t=json.load(open('../../runs/v22/full_18x50_test.json')); json.dump(t[:100], open('../../runs/v24/pool_18x50_test.json','w')); print('pools', len(r[:500]), len(t[:100]))
PY
python3 v24_make_range_states.py --pool $R/pool_18x50_train.json --sizes 35 40 45 50 --time_limit 60 --jobs 12 --out $R/cond_18x50_range_train.json
python3 v24_make_range_states.py --pool $R/pool_18x50_test.json  --sizes 35 40 45 50 --time_limit 60 --jobs 12 --out $R/cond_18x50_range_test.json
echo "[$(S)] ===== UP STAGE 2: R1 at 18x50 ====="
python3 v24_r1_collect.py --states $R/cond_18x50_range_train.json --n_free 40 45 50 --out $R/r1_18x50_train.json --budget 60 --per_state 25 --jobs 6 --selfcheck 1
python3 v24_r1_collect.py --states $R/cond_18x50_range_test.json  --n_free 40 45 50 --out $R/r1_18x50_test.json  --budget 60 --per_state 25 --jobs 6 --selfcheck 0
python3 v24_r1_train_critic.py --train $R/r1_18x50_train.json --test $R/r1_18x50_test.json --out $R/critic_18x50
echo "[$(S)] ===== UP STAGE 3: R2 at 18x50 ====="
python3 v24_r2_finetune.py --train $R/cond_18x50_range_train.json --test $R/cond_18x50_range_test.json --init_pv $R/critic_18x50/pv.pt \
  --n_free 40 45 50 --epochs 6 --states_per_epoch 400 --n_monitor 150 --budget 60 --jobs 6 --out $R/r2_18x50
BEST=$(python3 -c "import json; L=json.load(open('$R/r2_18x50/log.json')); print(min(L,key=lambda r:r['sum_ratio'])['epoch'])")
echo "[$(S)] selected epoch $BEST"; cp $R/r2_18x50/pv_ep$BEST.pt $R/r2_18x50/pv_best.pt
echo "[$(S)] ===== UP STAGE 4: evaluate PV50 at 21x60 (30, 100) and 18x50 ====="
cd ../v23_ablation
PV50=../../runs/v22/model_n25/conditional.pt,../../runs/v24/r2_18x50/pv_best.pt
PV60=../../runs/v22/model_n25/conditional.pt,../../runs/v24/r2/pv_best.pt
for SEED in 0 1 2; do
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_21x60_100 --tag H_PV50_S$SEED --time_limit 600 --block 12 --tries 10 --arms pv --ckpt $PV50 --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v22test_21x60 --tag C_PV50_S$SEED --time_limit 600 --block 12 --tries 10 --arms pv --ckpt $PV50 --ahl off on --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v22test_18x50 --tag B_PV50_S$SEED --time_limit 300 --block 10 --tries 10 --arms pv --ckpt $PV50 --ahl off on --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
done
echo "[$(S)] ===== UP STAGE 5: 24x70 probe (12 instances, |S| unverified) ====="
cd ../v22_transfer
python3 v22_make_testset.py --m 24 --n 70 --count 12 --time_limit 60 --jobs 12 --out_dir ../../instances/v24probe_24x70
cd ../v23_ablation
for SEED in 0; do
  python3 v23_cascade_warm.py --data_dir ../../instances/v24probe_24x70 --tag P70_M0_S$SEED --time_limit 900 --block 12 --tries 10 --arms lp model --ckpt ../../runs/v22/model_n25/conditional.pt --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24probe_24x70 --tag P70_PV_S$SEED --time_limit 900 --block 12 --tries 10 --arms pv --ckpt $PV60 --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24probe_24x70 --tag P70_PV50_S$SEED --time_limit 900 --block 12 --tries 10 --arms pv --ckpt $PV50 --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
done
cd ../v24_deploy_range
python3 v24_aggregate.py --tags M0 PV PVb PV50
echo "[$(S)] ===== V24 UP DONE ====="
