#!/bin/bash
# v28: the low-density question — does learned branching win where the LP is weak? Entry probability
# p=0.3 throughout (BLS_DENSITY env read by generate_hard_instances.gen_hard_feasible). Pipeline:
#   S1  10x25 pool with exact marginals -> solutions -> 80/20 split -> in-tree states -> Stage 1 (soft, seed 0)
#   S2  21x60 pool -> range states with exact marginals -> R1 critic (frozen) -> R2 policy (posterior init)
#   S3  evaluation sets 18x50 (30), 21x60 (100), 24x70 (30), 28x80 (30)
#   S4  our loop: classical rules + p=0.5-trained and p=0.3-trained Stage 1 / Stage 1+2
#   S5  SCIP: relpscost / mostinf / inference vs the p=0.3 Stage 1+2 plugin, conflict off and on
set -u
cd "$(dirname "$(readlink -f "$0")")"
export BLS_DENSITY=0.3
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v28; mkdir -p $R; I=../../instances
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
V15=../v15_bayes; V17=../v17_conditional; V22=../v22_transfer; V24=../v24_deploy_range; V25=../v25_review; V26=../v26_prior

stage "S1-a 10x25 pool (p=0.3) with marginals, solutions, split"
run $PY $V15/v15_gen_by_m.py --m 10 --n 25 --count 1600 --seed0 910000 --jobs 12 --out $R/full_10x25.json
run $PY $V22/v22_add_solutions.py --inp $R/full_10x25.json --out $R/sols_10x25.json --jobs 12
$PY - <<'PYEOF'
import json; r = json.load(open('../../runs/v28/sols_10x25.json')); n = int(0.8 * len(r))
json.dump(r[:n], open('../../runs/v28/sols_10x25_train.json', 'w')); json.dump(r[n:], open('../../runs/v28/sols_10x25_test.json', 'w')); print('split', n, len(r) - n)
PYEOF
stage "S1-b in-tree states and Stage 1 training"
run $PY $V17/v17_make_conditional_data.py --sols $R/sols_10x25_train.json --per_instance 8 --jobs 12 --out $R/cond_10x25_train.json
run $PY $V17/v17_make_conditional_data.py --sols $R/sols_10x25_test.json --per_instance 8 --jobs 12 --out $R/cond_10x25_test.json
run $PY $V25/v25_train_s1.py --seed 0 -- --train $R/cond_10x25_train.json --test $R/cond_10x25_test.json --n_train 10000 --n_test 2000 --epochs 20 --target soft --ref_ckpt ../../runs/v22/model_n25/conditional.pt --out $R/s1_p03 > $R/s1_p03.log 2>&1
tail -n 3 $R/s1_p03.log

stage "S2-a 21x60 pool and range states"
run $PY $V24/v24_gen_pool.py --count 500 --seed0 810000 --jobs 12 --out_train $R/pool_21x60_train.json --out_test $R/pool_21x60_test.json
run $PY $V24/v24_make_range_states.py --pool $R/pool_21x60_train.json --out $R/cond_21x60_range_train.json --time_limit 90 --jobs 12
run $PY $V24/v24_make_range_states.py --pool $R/pool_21x60_test.json --out $R/cond_21x60_range_test.json --time_limit 90 --jobs 12
stage "S2-b R1 critic and R2 policy"
run $PY $V24/v24_r1_collect.py --states $R/cond_21x60_range_train.json --ckpt $R/s1_p03/conditional.pt --out $R/r1_train.json --n_free 45 50 55 60 --budget 90 --per_state 25 --jobs 6
run $PY $V24/v24_r1_collect.py --states $R/cond_21x60_range_test.json --ckpt $R/s1_p03/conditional.pt --out $R/r1_test.json --n_free 45 50 55 60 --budget 90 --per_state 25 --jobs 6 --selfcheck 0
run $PY $V24/v24_r1_train_critic.py --train $R/r1_train.json --test $R/r1_test.json --ckpt $R/s1_p03/conditional.pt --out $R/critic_frozen
run $PY $V24/v24_r2_finetune.py --train $R/cond_21x60_range_train.json --test $R/cond_21x60_range_test.json --base $R/s1_p03/conditional.pt --init_pv $R/critic_frozen/pv.pt --epochs 6 --states_per_epoch 400 --n_monitor 150 --budget 90 --jobs 6 --out $R/r2 > $R/r2.log 2>&1
E=$($PY -c "
import json; L=json.load(open('$R/r2/log.json')); L=[r for r in L if r.get('sum_ratio') is not None]; b=min(L,key=lambda r:r['sum_ratio']); print(b['epoch'])")
cp $R/r2/pv_ep$E.pt $R/r2/pv_best.pt; echo "[$(S)] r2 (p=0.3): selected epoch $E"

stage "S3 evaluation sets (p=0.3)"
run $PY $V22/v22_make_testset.py --m 18 --n 50 --count 30 --time_limit 120 --jobs 10 --out_dir $I/v28lo_18x50
run $PY $V22/v22_make_testset.py --m 21 --n 60 --count 100 --time_limit 120 --jobs 10 --out_dir $I/v28lo_21x60_100
run $PY $V22/v22_make_testset.py --m 24 --n 70 --count 30 --time_limit 120 --jobs 10 --out_dir $I/v28lo_24x70
run $PY $V22/v22_make_testset.py --m 28 --n 80 --count 30 --time_limit 120 --jobs 10 --out_dir $I/v28lo_28x80

stage "S4 our loop"
S1_05=../../runs/v22/model_n25/conditional.pt; PV_05=../../runs/v24/r2/pv_best.pt; S1_03=$R/s1_p03/conditional.pt; PV_03=$R/r2/pv_best.pt
for T in "v28lo_18x50 B_S0 300" "v28lo_21x60_100 H_S0 600" "v28lo_24x70 G70_S0 1200" "v28lo_28x80 G80_S0 1200"; do
  set -- $T
  run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --arms lp lpfrac maxsd domwdeg random lp_probe model pv --ckpt_s1 $S1_05 --ckpt_pv $S1_05,$PV_05 --out_root $R/p05
  run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --arms model pv --ckpt_s1 $S1_03 --ckpt_pv $S1_03,$PV_03 --out_root $R/p03
done

stage "S5 SCIP"
for C in 0 1; do
  for T in "v28lo_21x60_100 H 600" "v28lo_24x70 G70 1200" "v28lo_28x80 G80 1200"; do
    set -- $T
    run $PY $V26/v26_scip_branch.py --data_dir $I/$1 --tag $2 --time_limit $3 --conflict $C --rules scip_relpscost scip_mostinf scip_inference pv --ckpt_s1 $S1_03 --ckpt_pv $PV_03 --out_root $R/scip
  done
done
stage "V28 DONE"
