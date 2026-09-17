#!/bin/bash
# v24 follow-up: route nodes with >= 50 free variables to M1 (better near the top), the rest to M0.
set -u
cd "$(dirname "$(readlink -f "$0")")/../v23_ablation"
S() { date '+%y-%m-%d %H:%M:%S'; }
SW=../../runs/v22/model_n25/conditional.pt,../../runs/v24/M1/conditional.pt,50
for SEED in 0 1 2; do
  echo "[$(S)] ===== switch seed $SEED ====="
  python3 v23_cascade_warm.py --data_dir ../../instances/v22test_21x60 --tag C_SW_S$SEED --time_limit 600 --block 12 --tries 10 \
    --arms switch --ckpt $SW --ahl off on --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_21x60_100 --tag H_SW_S$SEED --time_limit 600 --block 12 --tries 10 \
    --arms switch --ckpt $SW --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
done
echo "[$(S)] ===== V24 SWITCH DONE ====="
