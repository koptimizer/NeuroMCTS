#!/bin/bash
# v26 chain (prior-work comparisons requested 2026-10-03):
#   E1 NSNet-style marginal network trained on the Stage-1 data, evaluated as the Stage-1 arm
#   E2 root marginal -> rounding -> WalkSAT local search (NSNet's usage) with four initialisations
#   E3 our rules as a SCIP branching plugin vs SCIP's rules, conflict analysis off and on; LP-vertex diagnostic
#   E4 (only if the plugin beats every SCIP rule on PAR-2 at 24x70 with conflict on) end-to-end solvers
# Logs to runs/v26/chain.log; a failed stage is marked and the chain continues.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v26; mkdir -p $R
S() { date '+%y-%m-%d %H:%M:%S'; }
stage() { echo "[$(S)] ===== $1 ====="; }
run() { "$@" || echo "[$(S)] !!! FAILED: $*"; }
V25=../v25_review

stage "E1 NSNet-style marginal network: train (seed 0) and evaluate as Stage-1 arm"
run $PY v26_nsnet.py --train ../../runs/v22/cond_10x25_train.json --test ../../runs/v22/cond_10x25_test.json --n_train 10000 --n_test 2000 --epochs 20 --seed 0 --out $R/nsnet_s0 > $R/nsnet_s0.log 2>&1
tail -n 3 $R/nsnet_s0.log
run $PY $V25/v25_run.py --data_dir ../../instances/v22test_10x25 --tag A_S0 --time_limit 60 --jobs 6 --arms model --ckpt_s1 $R/nsnet_s0/conditional.pt --out_root $R/nsnet_s0
run $PY $V25/v25_run.py --data_dir ../../instances/v22test_18x50 --tag B_S0 --time_limit 300 --jobs 6 --arms model --ckpt_s1 $R/nsnet_s0/conditional.pt --out_root $R/nsnet_s0
run $PY $V25/v25_run.py --data_dir ../../instances/v24test_21x60_100 --tag H_S0 --time_limit 600 --jobs 6 --arms model --ckpt_s1 $R/nsnet_s0/conditional.pt --out_root $R/nsnet_s0
run $PY $V25/v25_run.py --data_dir ../../instances/v24test_24x70 --tag G70_S0 --time_limit 1200 --jobs 6 --arms model --ckpt_s1 $R/nsnet_s0/conditional.pt --out_root $R/nsnet_s0

stage "E2 root marginal + rounding + local search"
run $PY v26_local_search.py --data_dir ../../instances/v22test_10x25 --tag A --time_limit 60
run $PY v26_local_search.py --data_dir ../../instances/v22test_18x50 --tag B --time_limit 300
run $PY v26_local_search.py --data_dir ../../instances/v22test_21x60 --tag C --time_limit 300
run $PY v26_local_search.py --data_dir ../../instances/v24test_24x70 --tag G70 --time_limit 600

stage "E3 SCIP branching plugin, conflict analysis OFF"
run $PY v26_scip_branch.py --data_dir ../../instances/v24test_21x60_100 --tag H --time_limit 600 --conflict 0
run $PY v26_scip_branch.py --data_dir ../../instances/v24test_24x70 --tag G70 --time_limit 1200 --conflict 0
run $PY v26_scip_branch.py --data_dir ../../instances/v24test_28x80 --tag G80 --time_limit 1200 --conflict 0
stage "E3 SCIP branching plugin, conflict analysis ON"
run $PY v26_scip_branch.py --data_dir ../../instances/v24test_21x60_100 --tag H --time_limit 600 --conflict 1
run $PY v26_scip_branch.py --data_dir ../../instances/v24test_24x70 --tag G70 --time_limit 1200 --conflict 1
run $PY v26_scip_branch.py --data_dir ../../instances/v24test_28x80 --tag G80 --time_limit 1200 --conflict 1
stage "E3 diagnostic: network fed the Gurobi LP vertex inside SCIP (21x60, 30 instances)"
run $PY v26_scip_branch.py --data_dir ../../instances/v22test_21x60 --tag C --time_limit 600 --conflict 0 --rules model_glp pv_glp model pv

stage "E4 decision"
GO=$($PY - <<'PYEOF'
import json, glob, numpy as np
def par2(d, tl=1200):
	rs = [json.load(open(f)) for f in glob.glob(d + '/inst-*.json')]
	return np.mean([r['sec'] if r['solved'] else 2 * tl for r in rs]) if rs else float('inf')
ours = par2('../../runs/v26/SCIP_G70_S0_c1p0h0_pv')
scip = min(par2(f'../../runs/v26/SCIP_G70_S0_c1p0h0_{r}') for r in ['scip_relpscost', 'scip_mostinf', 'scip_inference', 'scip_random'])
print('yes' if ours < scip else 'no', round(ours, 1), round(scip, 1))
PYEOF
)
echo "[$(S)] E4 decision: $GO"
if [[ "$GO" == yes* ]]; then
  stage "E4 end-to-end solvers"
  run $PY v26_e2e.py --data_dir ../../instances/v24test_21x60_100 --tag H --time_limit 600
  run $PY v26_e2e.py --data_dir ../../instances/v24test_24x70 --tag G70 --time_limit 1200
  run $PY v26_e2e.py --data_dir ../../instances/v24test_28x80 --tag G80 --time_limit 1200
else
  stage "E4 skipped (plugin not ahead of SCIP's rules at 24x70); end-to-end reference without the plugin"
  run $PY v26_e2e.py --data_dir ../../instances/v24test_24x70 --tag G70 --time_limit 1200 --methods scip_default cpsat lattice
fi
stage "V26 DONE"
