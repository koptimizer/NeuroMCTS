#!/bin/bash
# v24 size-transfer extension beyond 21x60: 24x70 and 28x80 (30 instances each, m/n ~ 0.35 as at
# 21x60; |S| cannot be verified by enumeration at these sizes and is reported as a CP-SAT lower
# bound). Arms: lp, M0 (trained 10x25), PV60 (fine-tuned 21x60), PV50 (fine-tuned 18x50); lattice
# off; budget 1200 s; seed 0 first, seeds 1-2 only if the size is solvable. Waits for run_v24_up.sh.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S() { date '+%y-%m-%d %H:%M:%S'; }
until grep -q "V24 UP DONE" ../../runs/v24/up.log 2>/dev/null; do sleep 120; done
R=../../runs/v24; N25=../../runs/v22/model_n25/conditional.pt
PV60=$N25,../../runs/v24/r2/pv_best.pt; PV50=$N25,../../runs/v24/r2_18x50/pv_best.pt
for SPEC in "24 70" "28 80"; do
  set -- $SPEC; M=$1; N=$2; T=${M}x${N}
  echo "[$(S)] ===== BIG $T: instances ====="
  (cd ../v22_transfer && python3 v22_make_testset.py --m $M --n $N --count 30 --time_limit 60 --jobs 10 --out_dir ../../instances/v24test_$T)
  echo "[$(S)] ===== BIG $T: CP-SAT / SCIP baseline ====="
  (cd ../v22_transfer && python3 v22_classical_baseline.py --data_dir ../../instances/v24test_$T --time_limit 1200 --jobs 6 --out ../../runs/revision/classical_$T.json)
  echo "[$(S)] ===== BIG $T: search arms, seed 0 ====="
  cd ../v23_ablation
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_$T --tag G${N}_M0_S0   --time_limit 1200 --block 12 --tries 10 --arms model --ckpt $N25 --ahl off --jobs 6 --seed_offset 0 --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_$T --tag G${N}_PV_S0   --time_limit 1200 --block 12 --tries 10 --arms pv --ckpt $PV60 --ahl off --jobs 6 --seed_offset 0 --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_$T --tag G${N}_PV50_S0 --time_limit 1200 --block 12 --tries 10 --arms pv --ckpt $PV50 --ahl off --jobs 6 --seed_offset 0 --out_root ../../runs/v24
  python3 v23_cascade_warm.py --data_dir ../../instances/v24test_$T --tag G${N}_M0_S0   --time_limit 1200 --block 12 --tries 10 --arms lp --ckpt $N25 --ahl off --jobs 6 --seed_offset 0 --out_root ../../runs/v24
  cd ../v24_deploy_range
  echo "[$(S)] ===== BIG $T: seed 0 done ====="
done
echo "[$(S)] ===== V24 BIG DONE ====="
