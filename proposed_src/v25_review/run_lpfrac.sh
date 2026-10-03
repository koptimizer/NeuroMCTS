#!/bin/bash
# most-fractional LP branching (classical B&B rule) in our loop on the remaining sets, 3 workers
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
$PY v25_run.py --data_dir ../../instances/v22test_18x50 --tag B_S0 --time_limit 300 --jobs 3 --arms lpfrac
$PY v25_run.py --data_dir ../../instances/v24test_24x70 --tag G70_S0 --time_limit 1200 --jobs 3 --arms lpfrac
$PY v25_run.py --data_dir ../../instances/v24test_28x80 --tag G80_S0 --time_limit 1200 --jobs 3 --arms lpfrac
$PY v25_run.py --data_dir ../../instances/v25inf_18x50 --tag INF_S0 --time_limit 600 --jobs 3 --arms lpfrac
for V in dense03 dense07 plant03 nosel; do $PY v25_run.py --data_dir ../../instances/v25var_$V --tag V_${V}_S0 --time_limit 600 --jobs 3 --arms lpfrac; done
echo "===== LPFRAC DONE ====="
