#!/bin/bash
# v25 Phase 2 extension: the classical CP rules on the 100-instance 21x60 set (3 seeds), so that the
# strongest non-learned rules are compared on the same set as the main result. Waits for P2, gates P3.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V25 P2 DONE" $R/p2.log 2>/dev/null; do sleep 120; done
echo "[$(S)] ===== P2X classical CP rules on 21x60 100 instances, 3 seeds (B1) ====="
for SEED in 0 1 2; do
  $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S$SEED --time_limit 600 --seed_offset $SEED --jobs 6 \
      --arms maxsd domwdeg lp_probe impact || echo "[$(S)] !!! FAILED seed $SEED"
done
echo "[$(S)] ===== V25 P2X DONE ====="
