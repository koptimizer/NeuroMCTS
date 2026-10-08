#!/bin/bash
# v34: per-node cost profile at the paper density (p=0.3) for the node-vs-time sub-question of RQ1.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V33 DONE" ../../runs/v33/chain.log 2>/dev/null; do sleep 120; done
echo "[$(S)] ===== v34 per-node cost profile (p=0.3) ====="
cd ../v25_review
for SET in v28lo_21x60_100 v32d03_24x70_100 v32d03_28x80_100; do
  $PY v25_profile.py --set $SET --k 6 --ckpt_s1 ../../runs/v28/s1_p03/conditional.pt --ckpt_pv ../../runs/v28/r2/pv_best.pt || echo "[$(S)] !!! FAILED profile $SET"
done
echo "[$(S)] ===== V34 DONE ====="
