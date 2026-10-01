#!/bin/bash
# v25 Phase 2 extension 3: classical rules and random on the distribution variants (G1) and random on
# the 100-instance set (F8), so every table has the strong non-learned arms. Waits for P2Y, gates P3.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V25 P2Y DONE" $R/p2y.log 2>/dev/null; do sleep 120; done
echo "[$(S)] ===== P2Z classical rules + random on variants (G1) and random on 21x60 100 (F8) ====="
for V in dense03 dense07 plant03 nosel; do
  $PY v25_run.py --data_dir ../../instances/v25var_$V --tag V_${V}_S0 --time_limit 600 --seed_offset 0 --jobs 6 --arms maxsd domwdeg lp_probe random || echo "[$(S)] !!! FAILED $V"
done
for SEED in 0 1 2; do
  $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 --arms random || echo "[$(S)] !!! FAILED random seed $SEED"
done
echo "[$(S)] ===== V25 P2Z DONE ====="
