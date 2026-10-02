#!/bin/bash
# v25 Phase 5: NSNet-style inference baseline — damped sum-product BP marginals (no learning) in the
# Stage-1 slot, on every evaluation set (seed 0; node counts are seed-invariant). 3 workers so the
# P2W/P3 chains (6 workers) keep their timing. Logs to runs/v25/p5.log.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25
S() { date '+%y-%m-%d %H:%M:%S'; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
echo "[$(S)] ===== P5 BP marginal arm (NSNet inference baseline) ====="
run $PY v25_run.py --data_dir ../../instances/v22test_18x50 --tag B_S0 --time_limit 300 --jobs 3 --arms bp
run $PY v25_run.py --data_dir ../../instances/v22test_21x60 --tag C_S0 --time_limit 600 --jobs 3 --arms bp
run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S0 --time_limit 600 --jobs 3 --arms bp
run $PY v25_run.py --data_dir ../../instances/v24test_24x70 --tag G70_S0 --time_limit 1200 --jobs 3 --arms bp
echo "[$(S)] ===== V25 P5 DONE ====="
