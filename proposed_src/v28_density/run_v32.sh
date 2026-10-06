#!/bin/bash
# v32: raise 24x70 and 28x80 to 100 instances at the two densities that matter (0.3, 0.4), so that
# the comparison rests on more than one 100-instance set. Classical rules and the learned arms,
# both trained models, dense path. Runs after v31.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v32; mkdir -p $R; I=../../instances; V22=../v22_transfer; V25=../v25_review
S1_03=../../runs/v28/s1_p03/conditional.pt; PV_03=../../runs/v28/r2/pv_best.pt
S1_05=../../runs/v22/model_n25/conditional.pt; PV_05=../../runs/v24/r2/pv_best.pt
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
until grep -q "V31 DONE" ../../runs/v31/chain.log 2>/dev/null; do sleep 120; done
for P in 0.3 0.4; do
  TP=$(echo $P | tr -d '.')
  export BLS_DENSITY=$P
  stage "v32 p=$P: 100-instance sets at 24x70 and 28x80"
  run $PY $V22/v22_make_testset.py --m 24 --n 70 --count 100 --time_limit 120 --jobs 10 --out_dir $I/v32d${TP}_24x70_100
  run $PY $V22/v22_make_testset.py --m 28 --n 80 --count 100 --time_limit 120 --jobs 10 --out_dir $I/v32d${TP}_28x80_100
  for T in "v32d${TP}_24x70_100 G70_S0 1200" "v32d${TP}_28x80_100 G80_S0 1200"; do
    set -- $T
    stage "v32 p=$P $1"
    run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --arms lpfrac maxsd domwdeg random --out_root $R/d$TP
    for M in 03 05; do
      eval "S1=\$S1_$M; PV=\$PV_$M"
      run $PY $V25/v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --fast --arms pv pv_k2 --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/d${TP}_m$M
    done
  done
done
stage "V32 DONE"
