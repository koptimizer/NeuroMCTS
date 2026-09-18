#!/bin/bash
# 24x70 replication: seeds 1 and 2 for lp / M0 / PV60 / PV50 (28x80 stays single-seed: 5.5 h per seed).
set -u
cd "$(dirname "$(readlink -f "$0")")/../v23_ablation"
S() { date '+%y-%m-%d %H:%M:%S'; }
N25=../../runs/v22/model_n25/conditional.pt; PV60=$N25,../../runs/v24/r2/pv_best.pt; PV50=$N25,../../runs/v24/r2_18x50/pv_best.pt
for SEED in 1 2; do
  echo "[$(S)] ===== 24x70 seed $SEED ====="
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_24x70 --tag G70_PV_S$SEED   --time_limit 1200 --block 12 --tries 10 --arms pv --ckpt $PV60 --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_24x70 --tag G70_PV50_S$SEED --time_limit 1200 --block 12 --tries 10 --arms pv --ckpt $PV50 --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_24x70 --tag G70_M0_S$SEED   --time_limit 1200 --block 12 --tries 10 --arms lp model --ckpt $N25 --ahl off --jobs 6 --seed_offset $SEED --out_root ../../runs/v24
done
echo "[$(S)] ===== V24 BIG SEEDS DONE ====="
