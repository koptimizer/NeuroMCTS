#!/bin/bash
# v29: turning the node-count advantage into a wall-clock advantage at low density.
#   A  re-measure the learned arms on the dense TorchScript path (--fast); v28 ran them through PyG
#   B  reliability-style amortisation of the forward pass: network every k-th node with cached
#      per-variable scores in between (pv_k2/4/8), and depth gating (pv_gate50)
# All arms use the p=0.3-trained Stage 1 / Stage 1+2. Runs alone after v28 so the timings are clean.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v29; mkdir -p $R; I=../../instances
S1=../../runs/v28/s1_p03/conditional.pt; PV=../../runs/v28/r2/pv_best.pt
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
until grep -q "V28 DONE" ../../runs/v28/chain.log 2>/dev/null; do sleep 120; done
stage "v29: dense path + amortised forward (p=0.3 model)"
for T in "v28lo_21x60_100 H_S0 600" "v28lo_24x70 G70_S0 1200" "v28lo_28x80 G80_S0 1200"; do
  set -- $T
  run $PY v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --fast \
      --arms model pv pv_k2 pv_k4 pv_k8 pv_gate50 --ckpt_s1 $S1 --ckpt_pv $S1,$PV --out_root $R
done
stage "V29 DONE"
