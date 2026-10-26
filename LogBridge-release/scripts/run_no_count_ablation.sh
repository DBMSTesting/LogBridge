#!/usr/bin/env bash
# Count-aware ablation: omit the learned log-count feature.
# 用法:
#   bash scripts/run_no_count_ablation.sh tpc
#   bash scripts/run_no_count_ablation.sh tsbs
#   bash scripts/run_no_count_ablation.sh iotbench
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}

dataset=${1:?dataset required: tpc|tsbs|iotbench}
DATA="$REPO/datasets/$dataset"
KG=$(ls "$DATA"/kg_*.json | head -1)
BERT="$DATA/bert_embeddings.pt"
SUBGRAPHS="$DATA/contexts"
CKPT="$DATA/checkpoints_no_count_aware"   # 单独目录，不覆盖生产 ckpt

mkdir -p "$REPO/results/$dataset"

cd "$REPO" && $PYTHON src/training/train.py \
  --kg-file "$KG" \
  --output-dir "$CKPT" \
  --prebuilt-subgraph-dir "$SUBGRAPHS" \
  --bert-embeddings-file "$BERT" \
  --batch-size 32 --epochs 30 --lr 5e-4 \
  --device cuda --use-amp \
  --num-hops 1 --max-neighbors 5 \
  --inference-mode multi_window \
  --no-count-aware \
  --accumulation-steps 1 --num-workers 4 \
  --early-stopping-patience 100 \
  2>&1 | tee "$REPO/results/$dataset/no_count_aware.log"
