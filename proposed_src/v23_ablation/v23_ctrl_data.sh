#!/bin/bash
# Controlled root-vs-in-tree comparison (paper Table tab:ctrl). Recovered verbatim from the
# session transcript of 2026-09-15 13:12. Trains two MarginalNets on the same pool, labels,
# state count (1,760) and optimizer; only the state distribution differs. Run from v23_ablation/.
cd ../v17_conditional
python3 - <<'PYEOF'
import json, random
random.seed(0)
tr=json.load(open('../../runs/v22/cond_10x25_train.json')); te=json.load(open('../../runs/v22/cond_10x25_test.json'))
root=json.load(open('../../runs/v22/full_10x25_train.json'))   # 같은 풀의 root 상태 (soft label 보유)
n=len(root)
random.shuffle(tr)
json.dump(root, open('../../runs/revision/ctrl_root_train.json','w'))
json.dump(tr[:n], open('../../runs/revision/ctrl_intree_train.json','w'))
print(f"통제 비교용: root-only {n}개 vs in-tree {n}개 (동일 풀, 동일 soft 라벨, 동일 개수)")
PYEOF
for K in root intree; do
  nohup python3 v17_train_conditional.py --train ../../runs/revision/ctrl_${K}_train.json \
    --test ../../runs/v22/cond_10x25_test.json --n_train 100000 --n_test 2500 \
    --epochs 20 --target soft --out ../../runs/revision/ctrl_${K} \
    > ../../runs/revision/ctrl_${K}.log 2>&1 &
done
disown -a; sleep 2; echo "launched: $(pgrep -fc v17_train_conditional) trainings"
