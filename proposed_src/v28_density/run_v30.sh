#!/bin/bash
# v30: density sweep — is the gap between learned branching and the classical rules a monotone
# function of row density? Evaluation sets at p in {0.2, 0.4} (0.3 from v28, 0.5 from v25/v24),
# 21x60 (100 instances) and 24x70 (30). Arms: the classical rules, and the Stage 1 / Stage 1+2
# models trained at p=0.3 and at p=0.5, all on the dense TorchScript path.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v30; mkdir -p $R; I=../../instances; V22=../v22_transfer; V25=../v25_review
S1_03=../../runs/v28/s1_p03/conditional.pt; PV_03=../../runs/v28/r2/pv_best.pt
S1_05=../../runs/v22/model_n25/conditional.pt; PV_05=../../runs/v24/r2/pv_best.pt
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
until grep -q "V29 DONE" ../../runs/v29/chain.log 2>/dev/null; do sleep 120; done

for P in 0.2 0.4; do
  TAGP=$(echo $P | tr -d '.')
  stage "v30 p=$P evaluation sets"
  export BLS_DENSITY=$P
  run $PY $V22/v22_make_testset.py --m 21 --n 60 --count 100 --time_limit 120 --jobs 10 --out_dir $I/v30d${TAGP}_21x60_100
  run $PY $V22/v22_make_testset.py --m 24 --n 70 --count 30 --time_limit 120 --jobs 10 --out_dir $I/v30d${TAGP}_24x70
  stage "v30 p=$P arms"
  run $PY $V25/v25_run.py --data_dir $I/v30d${TAGP}_21x60_100 --tag H_S0 --time_limit 600 --jobs 6 \
      --arms lpfrac maxsd domwdeg lp_probe random --out_root $R/d${TAGP}
  for M in 03 05; do
    eval "S1=\$S1_$M; PV=\$PV_$M"
    run $PY $V25/v25_run.py --data_dir $I/v30d${TAGP}_21x60_100 --tag H_S0 --time_limit 600 --jobs 6 --fast \
        --arms model pv pv_k2 --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/d${TAGP}_m$M
  done
  run $PY $V25/v25_run.py --data_dir $I/v30d${TAGP}_24x70 --tag G70_S0 --time_limit 1200 --jobs 6 \
      --arms lpfrac maxsd domwdeg lp_probe random --out_root $R/d${TAGP}
  for M in 03 05; do
    eval "S1=\$S1_$M; PV=\$PV_$M"
    run $PY $V25/v25_run.py --data_dir $I/v30d${TAGP}_24x70 --tag G70_S0 --time_limit 1200 --jobs 6 --fast \
        --arms model pv pv_k2 --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R/d${TAGP}_m$M
  done
done
stage "V30 DONE"
