#!/bin/bash
# Reviewer-requested revision: three seeds of the branching comparison at 21x60.
# Seeds are now explicit (--seed_offset) because the previous run derived them from
# hash(), which Python randomizes per process -- every earlier run used different,
# unrecorded seeds, so no reported figure had a reproducible provenance.
set -u
cd "$(dirname "$(readlink -f "$0")")"
for S in 0 1 2; do
  echo "[$(date '+%y-%m-%d %H:%M:%S')] seed_offset=$S"
  python3 v22_cascade_search.py \
    --data_dir ../../instances/v22test_21x60 --tag SEED$S \
    --time_limit 600 --block 12 --tries 10 \
    --arms lp model --ckpt ../../runs/v22/model_n25/conditional.pt \
    --ahl off on --jobs 6 --seed_offset $S \
    --out_root ../../runs/v22_seeds || exit 1
done
echo "===== MULTISEED DONE ====="
