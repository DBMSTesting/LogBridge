#!/usr/bin/env bash
# 一键脚本：iotbench 训练 / 评估 / 消融
# 用法:
#   bash scripts/run_iotbench.sh train
#   bash scripts/run_iotbench.sh eval_single_window
#   bash scripts/run_iotbench.sh lt_only
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
DATA="$REPO/datasets/iotbench"
KG="$DATA/kg_iotbench_v3.json"
BERT="$DATA/bert_embeddings_v3.pt"
SUBGRAPHS="$DATA/prebuilt_subgraphs_k5_sameclass"
CKPT="$DATA/checkpoints_sameclass"
CKPT_LT="$DATA/checkpoints_lt_only"

mode=${1:-train}

case "$mode" in
  train)
    mkdir -p "$REPO/results/iotbench"
    cd "$REPO" && $PYTHON src/training/train.py \
      --kg-file "$KG" \
      --output-dir "$CKPT" \
      --prebuilt-subgraph-dir "$SUBGRAPHS" \
      --bert-embeddings-file "$BERT" \
      --batch-size 32 --epochs 30 --lr 5e-4 \
      --device cuda --use-amp \
      --num-hops 1 --max-neighbors 5 \
      --inference-mode single_window \
      --accumulation-steps 1 --num-workers 4 \
      --early-stopping-patience 100 \
      --focal-gamma 1.0 --max-class-weight 1.5 \
      2>&1 | tee "$REPO/results/iotbench/train.log"
    ;;

  eval_single_window)
    cd "$REPO" && $PYTHON src/training/eval_single_window.py \
      --data-root "$DATA" \
      2>&1 | tee "$REPO/results/iotbench/eval_single_window.log"
    ;;

  lt_only)
    cd "$REPO" && $PYTHON src/training/train_lt_only.py \
      --input-dir "$DATA" \
      --bert-emb "$BERT" \
      --output-dir "$CKPT_LT" \
      --epochs 30 --lr 5e-4 --batch-size 16 --patience 100 \
      2>&1 | tee "$REPO/results/iotbench/lt_only.log"
    ;;

  *)
    echo "未知模式: $mode"
    echo "用法: $0 {train|eval_single_window|lt_only}"
    exit 1
    ;;
esac
