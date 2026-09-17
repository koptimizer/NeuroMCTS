#!/bin/bash
# v24 Track 3, R1: collect exact decision costs under M0, train the critic, report the gate.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S() { date '+%y-%m-%d %H:%M:%S'; }
R=../../runs/v24
echo "[$(S)] ===== R1 STAGE A: collect (train) ====="
python3 v24_r1_collect.py --states $R/cond_21x60_range_train.json --out $R/r1_train.json --n_free 45 50 55 60 --budget 90 --per_state 25 --jobs 6
echo "[$(S)] ===== R1 STAGE B: collect (test) ====="
python3 v24_r1_collect.py --states $R/cond_21x60_range_test.json --out $R/r1_test.json --n_free 45 50 55 60 --budget 90 --per_state 25 --jobs 6 --selfcheck 0
echo "[$(S)] ===== R1 STAGE C: critic (value head only) ====="
python3 v24_r1_train_critic.py --train $R/r1_train.json --test $R/r1_test.json --out $R/critic_frozen
echo "[$(S)] ===== R1 STAGE D: critic (trunk unfrozen) ====="
python3 v24_r1_train_critic.py --train $R/r1_train.json --test $R/r1_test.json --out $R/critic_unfrozen --unfreeze --lr 3e-4
echo "[$(S)] ===== V24 R1 DONE ====="
