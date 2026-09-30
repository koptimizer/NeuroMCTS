#!/bin/bash
# v25 Phase 3 chain (training-dependent items): waits for Phase 2, then
#   B2  learned baselines: Stage 2 from the LP rule, from random init, posterior + unfrozen trunk; hard-label Stage 1
#   A1  dense TorchScript inference path: wall-clock re-measurement of the network arms
#   F7  extra training seeds for Stage 1 (2) and M1 (1)
#   F5  28x80 seeds 1-2
#   G3  infeasible 18x50 appendix (refutation trees)
# Every stage logs to runs/v25/p3.log; a failed stage is marked and the chain continues.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25; mkdir -p $R
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
until grep -q "V25 P2 DONE" $R/p2.log 2>/dev/null; do sleep 120; done

best_epoch() { $PY -c "
import json; L=json.load(open('$1/log.json')); L=[r for r in L if r.get('sum_ratio') is not None]; b=min(L,key=lambda r:r['sum_ratio']); print(b['epoch'])"; }

stage "P3-A Stage 2 variants (B2): lp init, random init, posterior+unfreeze"
run $PY v25_r2_variants.py --mode lp --out $R/rl_lp > $R/rl_lp.log 2>&1
run $PY v25_r2_variants.py --mode random --out $R/rl_random > $R/rl_random.log 2>&1
run $PY v25_r2_variants.py --mode posterior --unfreeze --out $R/rl_unfrozen > $R/rl_unfrozen.log 2>&1
for V in lp random unfrozen; do
  E=$(best_epoch $R/rl_$V); cp $R/rl_$V/pv_ep$E.pt $R/rl_$V/pv_best.pt; echo "[$(S)] rl_$V: selected epoch $E"
done
stage "P3-B hard-label Stage 1 (B2/C3) and Stage-1 seeds 1-2 (F7)"
run $PY v25_train_s1.py --seed 0 -- --train ../../runs/v22/cond_10x25_train.json --test ../../runs/v22/cond_10x25_test.json --n_train 10000 --n_test 2000 --epochs 20 --target hard --out ../../runs/v25/s1_hard > $R/s1_hard.log 2>&1
for SD in 1 2; do
  run $PY v25_train_s1.py --seed $SD -- --train ../../runs/v22/cond_10x25_train.json --test ../../runs/v22/cond_10x25_test.json --n_train 10000 --n_test 2000 --epochs 20 --target soft --out ../../runs/v25/s1_seed$SD > $R/s1_seed$SD.log 2>&1
done
stage "P3-C M1 seed 1 (F7)"
run $PY v25_train_s1.py --seed 1 -- --train ../../runs/v24/train_M1.json --test ../../runs/v24/cond_21x60_range_test.json --n_train 100000 --n_test 3000 --epochs 20 --target soft --out ../../runs/v25/M1_seed1 > $R/M1_seed1.log 2>&1
stage "P3-D evaluation of the variants on 21x60 (100 instances, 3 seeds)"
for SEED in 0 1 2; do
  for V in lp random unfrozen; do
    run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 --arms pv \
        --ckpt_pv ../../runs/v22/model_n25/conditional.pt,$R/rl_$V/pv_best.pt --out_root $R/rl_$V
  done
  for V in s1_hard s1_seed1 s1_seed2; do
    run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 --arms model \
        --ckpt_s1 $R/$V/conditional.pt --out_root $R/$V
  done
  run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 --arms model \
      --ckpt_s1 $R/M1_seed1/conditional.pt --out_root $R/M1_seed1
done
stage "P3-E dense inference path, wall-clock re-measurement (A1): 21x60 100 x 3 seeds, 24x70 30 x 3 seeds"
for SEED in 0 1 2; do
  run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 --arms model pv --fast
  run $PY v25_run.py --data_dir ../../instances/v24test_24x70 --tag G70_S$SEED --time_limit 1200 --seed_offset $SEED --jobs 6 --arms lp model pv --fast
done
stage "P3-F 28x80 seeds 1-2 (F5)"
for SEED in 1 2; do
  run $PY v25_run.py --data_dir ../../instances/v24test_28x80 --tag G80_S$SEED --time_limit 1200 --seed_offset $SEED --jobs 6 --arms lp model pv
done
stage "P3-G infeasible 18x50 appendix (G3)"
run $PY v25_gen_infeasible.py --jobs 6
for SEED in 0 1 2; do
  run $PY v25_run.py --data_dir ../../instances/v25inf_18x50 --tag INF_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 --arms lp model pv
done
stage "V25 P3 DONE"
