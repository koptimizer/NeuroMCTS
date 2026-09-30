#!/bin/bash
# v25 Phase 1+2 chain (docs/review_plan_261001.md): everything that needs no training, in one go.
# Every stage logs to runs/v25/p2.log; a failed stage is marked and the chain continues.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25; mkdir -p $R
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }

stage "P2-A cross ablation + baselines, 21x60 100 instances, 3 seeds (C1, D1)"
for SEED in 0 1 2; do
  run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 \
      --arms lp model pv lpvar_s1val s1var_lpval pvvar_lpval
done
stage "P2-B classical CP rules, 21x60 30 instances + 18x50, 3 seeds (B1)"
for SEED in 0 1 2; do
  run $PY v25_run.py --data_dir ../../instances/v22test_21x60 --tag C_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 \
      --arms lp model pv maxsd domwdeg impact lp_probe
  run $PY v25_run.py --data_dir ../../instances/v22test_18x50 --tag B_S$SEED --time_limit 300 --seed_offset $SEED --jobs 6 \
      --arms lp model pv lpvar_s1val s1var_lpval pvvar_lpval maxsd domwdeg impact lp_probe
done
stage "P2-C random arm on the current instance family (F8)"
run $PY v25_run.py --data_dir ../../instances/v22test_18x50 --tag B_S0 --time_limit 300 --seed_offset 0 --jobs 6 --arms random
run $PY v25_run.py --data_dir ../../instances/v22test_21x60 --tag C_S0 --time_limit 600 --seed_offset 0 --jobs 6 --arms random
stage "P2-D oracle marginals at 10x25 (C3) and the 10x25 arms"
run $PY v25_run.py --data_dir ../../instances/v22test_10x25 --tag A_S0 --time_limit 60 --seed_offset 0 --jobs 6 --arms lp model oracle maxsd domwdeg impact lp_probe random
stage "P2-E stronger lattice settings (B3)"
(cd ../.. && run $PY proposed_src/v25_review/v25_lattice_strong.py --jobs 6 --enum_cap 300 --out runs/v25/lattice_strong.json)
stage "P2-F distribution variants (G1)"
run $PY v25_gen_variants.py --jobs 6
for V in dense03 dense07 plant03 nosel; do
  run $PY v25_run.py --data_dir ../../instances/v25var_$V --tag V_${V}_S0 --time_limit 600 --seed_offset 0 --jobs 6 --arms lp model pv
done
stage "P2-G size-matched control for M1: Stage 1 on 2,691 10x25 states (C2)"
(cd ../v17_conditional && run $PY v17_train_conditional.py --train ../../runs/v22/cond_10x25_train.json --test ../../runs/v22/cond_10x25_test.json \
     --n_train 2691 --n_test 2000 --epochs 20 --out ../../runs/v25/ctrl_2691 > ../../runs/v25/ctrl_2691.log 2>&1)
for SEED in 0 1 2; do
  run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 \
      --arms model --ckpt_s1 ../../runs/v25/ctrl_2691/conditional.pt --out_root ../../runs/v25/ctrl2691
done
stage "P2-H Gurobi complete solver (B4)"
run $PY v25_gurobi_baseline.py --jobs 6
stage "P2-I per-node cost profile (A2)"
run $PY v25_profile.py --set v24test_21x60_100 --k 6
stage "V25 P2 DONE"
