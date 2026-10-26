#!/usr/bin/env bash
# Export diagnostics first; then generate one report per window.
# Usage: run_explanation_demo.sh [dataset] [dry-run|generate|judge]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
dataset=${1:-iotbench}
mode=${2:-dry-run}
diagnostics=${LOGBRIDGE_DIAGNOSTICS:-$REPO/datasets/$dataset/diagnostics_test.json}
index=${LOGBRIDGE_KB_INDEX:-$REPO/src/explanation/retrieval_index.pt}
output=${LOGBRIDGE_REPORT_OUTPUT:-$REPO/results/$dataset/reports}
options=()
case "$mode" in
  dry-run|stub) options+=(--dry-run); output="${output}_dry_run" ;;
  generate) ;;
  judge) options+=(--judge) ;;
  *) echo "Usage: $0 [dataset] [dry-run|generate|judge]"; exit 1 ;;
esac
if [[ ! -f "$index" ]]; then
  echo "Build the complete knowledge index first: python src/explanation/build_retrieval_index.py --rebuild"
  exit 1
fi
"$PYTHON" "$REPO/src/explanation/explanation_pipeline.py" \
  --diagnostics "$diagnostics" --output-root "$output" --index "$index" "${options[@]}"
