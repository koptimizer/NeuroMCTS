#!/bin/bash
# v25 Phase 2 extension 4: the strong classical rules and random at 24x70 and 28x80 (3 seeds at 70,
# seed 0 at 80), so the upward-transfer claim is made against them. Waits for P2Z, gates P3.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V25 P2Z DONE" $R/p2z.log 2>/dev/null; do sleep 120; done
echo "[$(S)] ===== P2W classical rules + random at 24x70 (3 seeds) and 28x80 (seed 0) (B1 upward) ====="
for SEED in 0 1 2; do
  $PY v25_run.py --data_dir ../../instances/v24test_24x70 --tag G70_S$SEED --time_limit 1200 --seed_offset $SEED --jobs 6 --arms maxsd domwdeg random lp_probe || echo "[$(S)] !!! FAILED 70 seed $SEED"
done
$PY v25_run.py --data_dir ../../instances/v24test_28x80 --tag G80_S0 --time_limit 1200 --seed_offset 0 --jobs 6 --arms maxsd domwdeg random lp_probe || echo "[$(S)] !!! FAILED 80"
echo "[$(S)] ===== V25 P2W DONE ====="
