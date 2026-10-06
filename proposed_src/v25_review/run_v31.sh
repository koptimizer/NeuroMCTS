#!/bin/bash
# v31: a cheap, reliability-style probing baseline in our own loop, so that the strongest classical
# rule is represented at a per-node cost comparable to ours (full probing = full strong branching is
# the quality bound but is far too slow to be the wall-clock reference).
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v31; mkdir -p $R; I=../../instances
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
until grep -q "V30 DONE" ../../runs/v30/chain.log 2>/dev/null; do sleep 120; done
stage "v31 reliability-style probing (rpb C=10, rpb20 C=20)"
for T in "v24test_21x60_100 H_S0 600" "v24test_24x70 G70_S0 1200" "v24test_28x80 G80_S0 1200" \
         "v28lo_21x60_100 H_S0 600" "v28lo_24x70 G70_S0 1200" "v28lo_28x80 G80_S0 1200"; do
  set -- $T
  OUT=$R/$(echo $1 | grep -q v28lo && echo lo || echo hi)
  run $PY v25_run.py --data_dir $I/$1 --tag $2 --time_limit $3 --jobs 6 --arms rpb rpb20 --out_root $OUT
done
stage "V31 DONE"
