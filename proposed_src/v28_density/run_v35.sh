#!/bin/bash
# v35: arms still missing from the single-density (p=0.3) paper tables on the 100-instance 24x70 / 28x80 sets:
#   Stage 1 only (seed 0, dense path) for the RQ3 "is Stage 2 necessary" row, and the impact rule for RQ1.
# Waits for v34 so the timing is not contaminated. Logs to runs/v35/chain.log.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
V25=../v25_review; I=../../instances; R=../../runs/v33; S1=../../runs/v28/s1_p03/conditional.pt
S() { date '+%y-%m-%d %H:%M:%S'; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
until grep -q "V34 DONE" ../../runs/v34/chain.log 2>/dev/null; do sleep 120; done
echo "[$(S)] ===== v35 Stage 1 only, 24x70 / 28x80 (100, p=0.3) ====="
run $PY $V25/v25_run.py --data_dir $I/v32d03_24x70_100 --tag G70_S0 --time_limit 1200 --jobs 6 --fast --arms model --ckpt_s1 $S1 --out_root $R/main
run $PY $V25/v25_run.py --data_dir $I/v32d03_28x80_100 --tag G80_S0 --time_limit 1200 --jobs 6 --fast --arms model --ckpt_s1 $S1 --out_root $R/main
echo "[$(S)] ===== v35 impact rule, 24x70 then 28x80 ====="
run $PY $V25/v25_run.py --data_dir $I/v32d03_24x70_100 --tag G70_S0 --time_limit 1200 --jobs 6 --arms impact --out_root $R/main
run $PY $V25/v25_run.py --data_dir $I/v32d03_28x80_100 --tag G80_S0 --time_limit 1200 --jobs 6 --arms impact --out_root $R/main
echo "[$(S)] ===== V35 DONE ====="
