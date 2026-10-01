#!/bin/bash
# v25 Phase 2 extension 2: the stronger-lattice sweep (B3), re-run after the daemonic-pool fix. Waits for P2X, gates P3.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/kopt/enter/envs/wsl_neuroMCTS/bin/python3
R=../../runs/v25
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V25 P2X DONE" $R/p2x.log 2>/dev/null; do sleep 120; done
echo "[$(S)] ===== P2Y stronger lattice settings (B3) ====="
(cd ../.. && $PY proposed_src/v25_review/v25_lattice_strong.py --jobs 6 --enum_cap 300 --out runs/v25/lattice_strong.json) || echo "[$(S)] !!! FAILED lattice"
echo "[$(S)] ===== V25 P2Y DONE ====="
