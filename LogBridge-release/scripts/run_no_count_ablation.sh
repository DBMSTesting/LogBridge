#!/usr/bin/env bash
# Count-aware 消融实验：HGT 完整模型 + 禁用 _lt_with_counts 展开
# 用法:
#   bash scripts/run_no_count_ablation.sh tpc
#   bash scripts/run_no_count_ablation.sh tsbs
#   bash scripts/run_no_count_ablation.sh iotbench
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}

dataset=${1:?dataset required: tpc|tsbs|iotbench}
DATA="$REPO/datasets/$dataset"
KG=$(ls "$DATA"/kg_*_v3.json | head -1)
BERT="$DATA/bert_embeddings_v3.pt"
SUBGRAPHS="$DATA/prebuilt_subgraphs_k5_sameclass"
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
  --inference-mode single_window \
  --no-count-aware \
  --accumulation-steps 1 --num-workers 4 \
  --early-stopping-patience 100 \
  --focal-gamma 1.0 --max-class-weight 1.5 \
  2>&1 | tee "$REPO/results/$dataset/no_count_aware.log"
