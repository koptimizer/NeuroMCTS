#!/bin/bash
# Full re-run of every search experiment on the warm-started loop, three seeds each, so
# all reported search numbers come from ONE implementation. Runs are sequential at
# concurrency 6 because wall-clock comparisons are sensitive to CPU contention.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S() { date '+%y-%m-%d %H:%M:%S'; }
OUT=../../runs/v23
N25=../../runs/v22/model_n25/conditional.pt
N50=../../runs/v22/model_n50/conditional.pt
run() { python3 v23_cascade_warm.py --data_dir ../../instances/v22test_$1 --tag $2_S$7 \
          --time_limit $3 --block $4 --tries 10 --arms $5 --ckpt $6 --ahl ${8:-off on} \
          --jobs 6 --seed_offset $7 --out_root $OUT || exit 1; }
for SEED in 0 1 2; do
  echo "[$(S)] ===== seed $SEED ====="
  run 10x25 A_n25to25   60  8 "lp model" $N25 $SEED
  run 18x50 B_n25to50  300 10 "lp model" $N25 $SEED
  run 18x50 R_n50to50  300 10 "model"    $N50 $SEED
  run 21x60 C_n25to60  600 12 "lp model" $N25 $SEED
  run 21x60 D_n50to60  600 12 "model"    $N50 $SEED
  run 21x60 Crand      600 12 "random"   $N25 $SEED "on"
done
echo "[$(S)] ===== REVISION2 ALL DONE ====="
