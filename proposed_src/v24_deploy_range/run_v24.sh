#!/bin/bash
# v24 chain: deployment-range supervision. Every stage logs to runs/v24/chain.log; a failed
# stage is marked but the chain continues so that independent stages still produce results.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S() { date '+%y-%m-%d %H:%M:%S'; }
R=../../runs/v24; mkdir -p $R
N25=../../runs/v22/model_n25/conditional.pt
stage() { echo "[$(S)] ===== STAGE $1 START: $2 ====="; }
done_() { echo "[$(S)] ===== STAGE $1 DONE (exit $2) ====="; }

stage 1 "21x60 pool"
python3 v24_gen_pool.py --count 500 --out_train $R/pool_21x60_train.json --out_test $R/pool_21x60_test.json; done_ 1 $?

stage 2 "range states with exact marginals"
python3 v24_make_range_states.py --pool $R/pool_21x60_train.json --out $R/cond_21x60_range_train.json --time_limit 90 --jobs 12; e1=$?
python3 v24_make_range_states.py --pool $R/pool_21x60_test.json  --out $R/cond_21x60_range_test.json  --time_limit 90 --jobs 12; done_ 2 $((e1+$?))

stage 3 "training-set variants"
python3 v24_mix.py --out_dir $R; done_ 3 $?

stage 4 "train M1 M2 P06 P612 P012 (parallel, single-thread each)"
cd ../v17_conditional
for K in M1 M2; do
  python3 v17_train_conditional.py --train ../../runs/v24/train_$K.json --test ../../runs/v24/cond_21x60_range_test.json \
    --n_train 100000 --n_test 3000 --epochs 20 --target soft --out ../../runs/v24/$K > ../../runs/v24/train_$K.log 2>&1 &
done
for K in P06 P612 P012; do
  python3 v17_train_conditional.py --train ../../runs/v24/train_$K.json --test ../../runs/v22/cond_10x25_test.json \
    --n_train 100000 --n_test 2500 --epochs 20 --target soft --out ../../runs/v24/$K > ../../runs/v24/train_$K.log 2>&1 &
done
wait; cd ../v24_deploy_range
for K in M1 M2 P06 P612 P012; do echo "--- $K"; grep -E "^ +ALL|train=" $R/train_$K.log; done
done_ 4 0

stage 5 "where-table, paired over checkpoints"
python3 v24_where_eval.py --ckpts $N25 $R/M1/conditional.pt $R/M2/conditional.pt --names M0 M1 M2 --out $R/where_M012.json; done_ 5 $?

stage 6 "30-instance evaluation, 3 seeds, warm loop (lp, M0, M1, M2)"
cd ../v23_ablation
run() { python3 v23_cascade_warm.py --data_dir ../../instances/$(if [ "$1" = 21x60_100X ]; then echo v24test_21x60_100; else echo v22test_$1; fi) --tag $2_S$7 --time_limit $3 --block $4 --tries 10 \
          --arms $5 --ckpt $6 --ahl ${8:-off on} --jobs 6 --seed_offset $7 --out_root ../../runs/v24; }
for SEED in 0 1 2; do
  run 21x60 C_M0 600 12 "lp model" ../../runs/v22/model_n25/conditional.pt $SEED
  run 21x60 C_M1 600 12 "model"    ../../runs/v24/M1/conditional.pt      $SEED
  run 21x60 C_M2 600 12 "model"    ../../runs/v24/M2/conditional.pt      $SEED
  run 18x50 B_M0 300 10 "lp model" ../../runs/v22/model_n25/conditional.pt $SEED
  run 18x50 B_M1 300 10 "model"    ../../runs/v24/M1/conditional.pt      $SEED
  run 18x50 B_M2 300 10 "model"    ../../runs/v24/M2/conditional.pt      $SEED
done
cd ../v24_deploy_range; done_ 6 0

stage 7 "100-instance 21x60 evaluation set (seeds 700000-700099; first 30 coincide with v22test_21x60)"
cd ../v22_transfer
python3 v22_make_testset.py --m 21 --n 60 --count 100 --time_limit 120 --jobs 10 --out_dir ../../instances/v24test_21x60_100; e=$?
cd ../v24_deploy_range
python3 - <<'PY'
import json,glob,hashlib,numpy as np
k=lambda d: hashlib.md5(np.array(d['A'],dtype=np.int64).tobytes()+np.array(d['b'],dtype=np.int64).tobytes()).hexdigest()
a={k(json.load(open(f))) for f in glob.glob('../../instances/v22test_21x60/*.json')}
b={k(json.load(open(f))) for f in glob.glob('../../instances/v24test_21x60_100/*.json')}
print(f"100-set contains the original 30: {len(a&b)}/30; total unique {len(b)}")
PY
done_ 7 $e

stage 8 "100-instance evaluation, AHL off, 3 seeds (lp, M0, M1, M2)"
cd ../v23_ablation
for SEED in 0 1 2; do
  run 21x60_100X H_M0 600 12 "lp model" ../../runs/v22/model_n25/conditional.pt $SEED "off"
  run 21x60_100X H_M1 600 12 "model"    ../../runs/v24/M1/conditional.pt      $SEED "off"
  run 21x60_100X H_M2 600 12 "model"    ../../runs/v24/M2/conditional.pt      $SEED "off"
done
cd ../v24_deploy_range; done_ 8 0

stage 9 "R0: residual for a cost-aware policy (M0 and M1)"
python3 v24_r0_residual.py --states $R/cond_21x60_range_test.json --ckpt $N25 --n_states 100 --out $R/r0_M0.json; e1=$?
python3 v24_r0_residual.py --states $R/cond_21x60_range_test.json --ckpt $R/M1/conditional.pt --n_states 100 --out $R/r0_M1.json; done_ 9 $((e1+$?))

echo "[$(S)] ===== V24 ALL DONE ====="
