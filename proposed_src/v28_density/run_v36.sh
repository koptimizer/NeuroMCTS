#!/bin/bash
# v36: per-node cost profile on the dense TorchScript path (the deployed one) at p=0.3; v34 measured the PyG path.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V34 DONE" ../../runs/v34/chain.log 2>/dev/null; do sleep 60; done
echo "[$(S)] ===== v36 per-node cost profile, dense path (p=0.3) ====="
cd ../v25_review
for SET in v28lo_21x60_100 v32d03_24x70_100 v32d03_28x80_100; do
  $PY v25_profile.py --set $SET --k 6 --time_limit 1200 --fast --ckpt_s1 ../../runs/v28/s1_p03/conditional.pt --ckpt_pv ../../runs/v28/r2/pv_best.pt || echo "[$(S)] !!! FAILED profile $SET"
done
echo "[$(S)] ===== V36 DONE ====="
