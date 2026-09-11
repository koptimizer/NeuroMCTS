#!/bin/bash
# v22 stage 2, chained end to end: test set -> condition C -> condition D -> full report.
# Size fixed at 21x60 (|S| median 12, verified 10/10; n=70 rejected at 0/10 verified).
set -u
cd "$(dirname "$(readlink -f "$0")")"
STAMP() { date "+%y-%m-%d %H:%M:%S"; }

echo "[$(STAMP)] STAGE 1/4: build |S|-matched test set at 21x60"
python3 v22_make_testset.py --m 21 --n 60 --count 30 --K 20 \
  --time_limit 120 --jobs 10 --out_dir ../../instances/v22test_21x60 || exit 1
echo "[$(STAMP)] STAGE 1 done"

echo "[$(STAMP)] STAGE 2/4: condition C (n=25 model -> n=60), arms lp+model, AHL off+on"
python3 v22_cascade_search.py --data_dir ../../instances/v22test_21x60 --tag C_n25to60 \
  --time_limit 600 --block 12 --tries 10 --arms lp model \
  --ckpt ../../runs/v22/model_n25/conditional.pt --ahl off on --jobs 6 || exit 1
echo "[$(STAMP)] STAGE 2 done"

echo "[$(STAMP)] STAGE 3/4: condition D (n=50 model -> n=60), arm model, AHL off+on"
python3 v22_cascade_search.py --data_dir ../../instances/v22test_21x60 --tag D_n50to60 \
  --time_limit 600 --block 12 --tries 10 --arms model \
  --ckpt ../../runs/v22/model_n50/conditional.pt --ahl off on --jobs 6 || exit 1
echo "[$(STAMP)] STAGE 3 done"

echo "[$(STAMP)] STAGE 4/4: cross-arm report"
python3 v22_report.py
echo "[$(STAMP)] ===== V22 CD ALL DONE ====="
