#!/bin/bash
# v33: complete the evidence at the single paper density p=0.3 (every table in the paper must come
# from this density). Waits for v32. Stages:
#   S0 10x25 evaluation set
#   S1 training variants on the p=0.3 data: hard-label Stage 1, Stage-1 seeds 1-2, size-matched control
#      (10x25, 2,771 states), M1 (21x60-trained Stage 1), NSNet-style Stage 1, Stage 2 from random / LP
#      init and with the trunk unfrozen
#   S2 our loop: missing classical arms, value x variable cross ablation, oracle / BP reference signals,
#      all variants, timing seeds 1-2 for the main arms
#   S4 SCIP (rules + plugin, conflict off/on) on the 100-instance 24x70 / 28x80 sets
#   S5 end-to-end solvers (SCIP default, CP-SAT, Gurobi, BKZ-40x30 + enumeration)
#   S3 root marginal + local search (NSNet usage)
set -u
cd "$(dirname "$(readlink -f "$0")")"
export BLS_DENSITY=0.3
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v33; mkdir -p $R; I=../../instances; D=../../runs/v28
V22=../v22_transfer; V25=../v25_review; V26=../v26_prior
S1=$D/s1_p03/conditional.pt; PV=$D/r2/pv_best.pt; CR=$D/critic_frozen/pv.pt
TR=$D/cond_10x25_train.json; TE=$D/cond_10x25_test.json; RTR=$D/cond_21x60_range_train.json; RTE=$D/cond_21x60_range_test.json
A=v33lo_10x25; B=v28lo_18x50; H=v28lo_21x60_100; G70=v32d03_24x70_100; G80=v32d03_28x80_100
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
best_epoch() { $PY -c "
import json; L=json.load(open('$1/log.json')); L=[r for r in L if r.get('sum_ratio') is not None]; b=min(L,key=lambda r:r['sum_ratio']); print(b['epoch'])"; }
until grep -q "V32 DONE" ../../runs/v32/chain.log 2>/dev/null; do sleep 120; done

stage "S0 10x25 evaluation set (p=0.3)"
run $PY $V22/v22_make_testset.py --m 10 --n 25 --count 30 --time_limit 60 --jobs 10 --out_dir $I/$A

stage "S1 training variants (p=0.3 data)"
run $PY $V25/v25_train_s1.py --seed 0 -- --train $TR --test $TE --n_train 10000 --n_test 2000 --epochs 20 --target hard --ref_ckpt $S1 --out $R/s1_hard > $R/s1_hard.log 2>&1
for SD in 1 2; do run $PY $V25/v25_train_s1.py --seed $SD -- --train $TR --test $TE --n_train 10000 --n_test 2000 --epochs 20 --target soft --ref_ckpt $S1 --out $R/s1_seed$SD > $R/s1_seed$SD.log 2>&1; done
run $PY $V25/v25_train_s1.py --seed 0 -- --train $TR --test $TE --n_train 2771 --n_test 2000 --epochs 20 --target soft --ref_ckpt $S1 --out $R/s1_ctrl > $R/s1_ctrl.log 2>&1
run $PY $V25/v25_train_s1.py --seed 0 -- --train $RTR --test $RTE --n_train 100000 --n_test 3000 --epochs 20 --target soft --ref_ckpt $S1 --out $R/M1 > $R/M1.log 2>&1
run $PY $V26/v26_nsnet.py --train $TR --test $TE --n_train 10000 --n_test 2000 --epochs 20 --seed 0 --out $R/nsnet > $R/nsnet.log 2>&1
for V in random lp; do
  run $PY $V25/v25_r2_variants.py --mode $V --train $RTR --test $RTE --base $S1 --init_pv $CR --out $R/rl_$V > $R/rl_$V.log 2>&1
done
run $PY $V25/v25_r2_variants.py --mode posterior --unfreeze --train $RTR --test $RTE --base $S1 --init_pv $CR --out $R/rl_unfrozen > $R/rl_unfrozen.log 2>&1
for V in random lp unfrozen; do E=$(best_epoch $R/rl_$V); cp $R/rl_$V/pv_ep$E.pt $R/rl_$V/pv_best.pt; echo "[$(S)] rl_$V: selected epoch $E"; done

stage "S2 our loop (p=0.3)"
run $PY $V25/v25_run.py --data_dir $I/$A --tag A_S0 --time_limit 60 --jobs 6 --arms lp lpfrac maxsd domwdeg random lp_probe impact oracle bp --ckpt_s1 $S1 --out_root $R/main
run $PY $V25/v25_run.py --data_dir $I/$A --tag A_S0 --time_limit 60 --jobs 6 --fast --arms model pv lpvar_s1val s1var_lpval pvvar_lpval --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/main
run $PY $V25/v25_run.py --data_dir $I/$B --tag B_S0 --time_limit 300 --jobs 6 --arms bp impact --ckpt_s1 $S1 --out_root $R/main
run $PY $V25/v25_run.py --data_dir $I/$B --tag B_S0 --time_limit 300 --jobs 6 --fast --arms model pv lpvar_s1val s1var_lpval pvvar_lpval --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/main
run $PY $V25/v25_run.py --data_dir $I/$H --tag H_S0 --time_limit 600 --jobs 6 --arms impact --out_root $R/main
run $PY $V25/v25_run.py --data_dir $I/$H --tag H_S0 --time_limit 600 --jobs 6 --fast --arms lpvar_s1val s1var_lpval pvvar_lpval --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/main
for T in "$G70 G70_S0 1200" "$G80 G80_S0 1200"; do
  set -- $T
  run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --arms lp lp_probe --out_root $R/main
  run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --fast --arms lpvar_s1val s1var_lpval pvvar_lpval --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/main
done
stage "S2b variants and timing seeds"
for T in "$H H_S0 600" "$G70 G70_S0 1200" "$G80 G80_S0 1200"; do
  set -- $T
  for V in s1_hard s1_seed1 s1_seed2 s1_ctrl M1; do
    run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --fast --arms model --ckpt_s1 $R/$V/conditional.pt --out_root $R/$V
  done
  run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --arms model --ckpt_s1 $R/nsnet/conditional.pt --out_root $R/nsnet
  for V in random lp unfrozen; do
    run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --arms pv --ckpt_pv $S1,$R/rl_$V/pv_best.pt --out_root $R/rl_$V
  done
  for SD in 1 2; do
    run $PY $V25/v25_run.py --data_dir $I/$1 --tag ${2%_S0}_S$SD --time_limit $3 --seed_offset $SD --jobs 6 --fast --arms model pv --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/main
  done
done

stage "S4 SCIP on the 100-instance 24x70 / 28x80 (p=0.3)"
for C in 0 1; do
  run $PY $V26/v26_scip_branch.py --data_dir $I/$G70 --tag G70 --time_limit 1200 --conflict $C --rules scip_relpscost scip_mostinf scip_inference pv --ckpt_s1 $S1 --ckpt_pv $PV --out_root $R/scip
  run $PY $V26/v26_scip_branch.py --data_dir $I/$G80 --tag G80 --time_limit 1200 --conflict $C --rules scip_relpscost scip_mostinf scip_inference pv --ckpt_s1 $S1 --ckpt_pv $PV --out_root $R/scip
done

stage "S5 end-to-end solvers (p=0.3)"
run $PY $V26/v26_e2e.py --data_dir $I/$H --tag H --time_limit 600 --methods scip_default cpsat lattice --out_root $R/e2e
run $PY $V26/v26_e2e.py --data_dir $I/$G70 --tag G70 --time_limit 1200 --methods scip_default cpsat lattice --out_root $R/e2e
run $PY $V26/v26_e2e.py --data_dir $I/$G80 --tag G80 --time_limit 1200 --methods scip_default cpsat lattice --out_root $R/e2e
(cd $V25 && run $PY v25_gurobi_baseline.py --sets $A $B $H $G70 $G80 --budgets 60 300 600 1200 1200 --jobs 6 > ../../runs/v33/gurobi.log 2>&1)

stage "S3 root marginal + local search (p=0.3)"
run $PY $V26/v26_local_search.py --data_dir $I/$A --tag A --time_limit 60 --ckpt_s1 $S1 --out_root $R/ls
run $PY $V26/v26_local_search.py --data_dir $I/$B --tag B --time_limit 300 --ckpt_s1 $S1 --out_root $R/ls
run $PY $V26/v26_local_search.py --data_dir $I/$H --tag H --time_limit 120 --inits s1 lp uniform --ckpt_s1 $S1 --out_root $R/ls
stage "V33 DONE"
