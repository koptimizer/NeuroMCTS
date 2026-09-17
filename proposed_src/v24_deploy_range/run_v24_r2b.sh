#!/bin/bash
# v24 R2 replication: second training seed (rollout sampling + init differ), plus the 18x50
# check for the pv arm that the first chain omitted.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S() { date '+%y-%m-%d %H:%M:%S'; }
R=../../runs/v24
echo "[$(S)] ===== R2b: fine-tune seed 1 ====="
python3 v24_r2_finetune.py --init_pv $R/critic_frozen/pv.pt --epochs 6 --states_per_epoch 400 --n_monitor 150 --budget 90 --jobs 6 --seed 1 --out $R/r2_seed1
BEST=$(python3 -c "import json; L=json.load(open('$R/r2_seed1/log.json')); print(min(L,key=lambda r:r['sum_ratio'])['epoch'])")
echo "[$(S)] selected epoch $BEST"; cp $R/r2_seed1/pv_ep$BEST.pt $R/r2_seed1/pv_best.pt
cd ../v23_ablation
PV1=../../runs/v22/model_n25/conditional.pt,../../runs/v24/r2_seed1/pv_best.pt
PV0=../../runs/v22/model_n25/conditional.pt,../../runs/v24/r2/pv_best.pt
for SEED in 0 1 2; do
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_21x60_100 --tag H_PVb_S$SEED --time_limit 600 --block 12 --tries 10 --arms pv --ckpt $PV1 --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v22test_21x60 --tag C_PVb_S$SEED --time_limit 600 --block 12 --tries 10 --arms pv --ckpt $PV1 --ahl off on --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v22test_18x50 --tag B_PV_S$SEED --time_limit 300 --block 10 --tries 10 --arms pv --ckpt $PV0 --ahl off on --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
done
cd ../v24_deploy_range
python3 v24_aggregate.py --tags M0 PV PVb
echo "[$(S)] ===== V24 R2b DONE ====="
