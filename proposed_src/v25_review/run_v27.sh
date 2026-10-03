#!/bin/bash
# P7: most-fractional ideas inside our method.
#   A  cheap arms on 18x50 / 21x60 (100) / 24x70: most-fractional variable + marginal value; Stage-1+2 policy restricted
#      to fractional variables; policy among the 5 / 10 most fractional variables
#   B  Stage 2 fine-tuned from the most-fractional rule (mode mostfrac), best epoch by monitor, evaluated on the same sets + 28x80
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v27; mkdir -p $R
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
ARMS="lpfracvar_s1val pv_frac pv_top5 pv_top10"
stage "P7-A cheap arms"
run $PY v25_run.py --data_dir ../../instances/v22test_18x50 --tag B_S0 --time_limit 300 --jobs 6 --arms $ARMS --out_root $R
run $PY v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S0 --time_limit 600 --jobs 6 --arms $ARMS --out_root $R
run $PY v25_run.py --data_dir ../../instances/v24test_24x70 --tag G70_S0 --time_limit 1200 --jobs 6 --arms $ARMS --out_root $R
stage "P7-B Stage 2 from the most-fractional rule"
run $PY v25_r2_variants.py --mode mostfrac --out $R/rl_mostfrac > $R/rl_mostfrac.log 2>&1
E=$($PY -c "
import json; L=json.load(open('$R/rl_mostfrac/log.json')); L=[r for r in L if r.get('sum_ratio') is not None]; b=min(L,key=lambda r:r['sum_ratio']); print(b['epoch'])")
cp $R/rl_mostfrac/pv_ep$E.pt $R/rl_mostfrac/pv_best.pt; echo "[$(S)] rl_mostfrac: selected epoch $E"
for T in "v22test_18x50 B_S0 300" "v24test_21x60_100 H_S0 600" "v24test_24x70 G70_S0 1200" "v24test_28x80 G80_S0 1200"; do
  set -- $T
  run $PY v25_run.py --data_dir ../../instances/$1 --tag $2 --time_limit $3 --jobs 6 --arms pv --ckpt_pv ../../runs/v22/model_n25/conditional.pt,$R/rl_mostfrac/pv_best.pt --out_root $R/rl_mostfrac
done
stage "V27 DONE"
