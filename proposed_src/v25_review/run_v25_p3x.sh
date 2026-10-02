#!/bin/bash
# v25 P3X: re-evaluate the lp / random Stage-2 variants at seed 0 with their own policy formula
# (the first P3-D pass loaded them through PolicyValueNet, i.e. with the posterior formula: invalid).
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V25 P3 DONE" $R/p3.log 2>/dev/null; do sleep 120; done
echo "[$(S)] ===== P3X re-evaluation of rl_lp / rl_random at seed 0 (mode-aware loader) ====="
for V in lp random; do
  $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S0 --time_limit 600 --seed_offset 0 --jobs 6 --arms pv \
      --ckpt_pv ../../runs/v22/model_n25/conditional.pt,$R/rl_$V/pv_best.pt --out_root $R/rl_$V || echo "[$(S)] !!! FAILED $V"
done
echo "[$(S)] ===== V25 P3X DONE ====="
