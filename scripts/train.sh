#!/usr/bin/env bash
# LoRA fine-tune of Qwen3.6-27B (MLX 4-bit) on the full cl-macros dataset.
#
# Run overnight detached:
#   nohup bash scripts/train.sh > training.log 2>&1 &
#   tail -f training.log
#
# Resume from a saved adapter:
#   RESUME_FROM=models/checkpoints/baseline-no-traces/0000400_adapters.safetensors \
#       bash scripts/train.sh

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/lora_baseline.yaml}"
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p models/checkpoints/baseline-no-traces

echo "=== mlx-lm LoRA fine-tune ==="
echo "config:  $CONFIG"
echo "data:    data/processed/full ($(wc -l < data/processed/full/train.jsonl) train / $(wc -l < data/processed/full/valid.jsonl) valid)"
echo "started: $(date)"
echo

resume_args=()
if [[ -n "$RESUME_FROM" ]]; then
  echo "resuming from: $RESUME_FROM"
  resume_args=(--resume-adapter-file "$RESUME_FROM")
fi

uv tool run --from "mlx-lm==0.31.3" mlx_lm.lora \
  --config "$CONFIG" \
  "${resume_args[@]}"

echo
echo "=== finished: $(date) ==="
ls -lh models/checkpoints/baseline-no-traces/
