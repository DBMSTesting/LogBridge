#!/usr/bin/env bash
# 解释生成 demo：6 类各取 1-2 个窗口跑完整 pipeline
# 用法:
#   bash scripts/run_explanation_demo.sh [iotbench|tpc|tsbs] [stub|ollama]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}

dataset=${1:-iotbench}
backend=${2:-stub}

WINDOWS_JSON="$REPO/datasets/$dataset/windows_anomaly_test_compressed.json"

# 1. 确保检索索引已建
INDEX="$REPO/src/explanation/retrieval_index.pt"
if [ ! -f "$INDEX" ]; then
  echo "[1/2] 构建检索索引 ..."
  cd "$REPO/src/explanation" && $PYTHON build_retrieval_index.py --rebuild
else
  echo "[1/2] 检索索引已存在: $INDEX"
fi

# 2. 跑 pipeline
echo "[2/2] 跑解释生成 pipeline (backend=$backend) ..."
cd "$REPO/src/explanation" && $PYTHON explanation_pipeline.py \
  --windows-json "$WINDOWS_JSON" \
  --n-per-class 1 \
  --backend "$backend" \
  ${backend:+--model qwen35-opus-27b:q4km}

echo ""
echo "输出在: $REPO/src/explanation/demo_pipeline_output/"
