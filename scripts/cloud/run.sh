#!/bin/bash
# End-to-end RunPod training pipeline for cl-macro-llm.
#
#   1. install python deps (Unsloth + others)
#   2. verify HF access (gated base model + dataset)
#   2.5. verify Unsloth recognises Qwen3.6's model_type
#   3. train (LoRA SFT via Unsloth, full bf16)
#   4. smoke-test generation
#   5. push adapter to HF (if HF_REPO is set)
#   6. print summary
#
# Unsloth manages the transformers/peft/trl/torch stack itself
# (it requires transformers v5 for Qwen3.5/3.6 — the official
# AutoModelForCausalLM path can't load these models).
#
# See scripts/cloud/ENV.md for the env vars this script reads.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./output}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B}"
DATASET="${DATASET:-j14i/cl-macros-thinking}"
SMOKE_PROMPT="${SMOKE_PROMPT:-Write a Common Lisp macro \`when-let\` that binds the result of an expression to a name and runs a body only when the value is non-nil.}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
fail() { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

step "1/5  Installing python requirements (Unsloth + deps)"
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install -r scripts/cloud/requirements.txt -q

step "2/5  Verifying HuggingFace access"
if [[ -z "${HF_TOKEN:-}" ]]; then
    fail "HF_TOKEN is not set — export it before running (needed for adapter upload + private repos)."
fi
python - <<PY
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    api.model_info("${BASE_MODEL}")
    print("  base model OK: ${BASE_MODEL}")
except Exception as e:
    sys.exit(f"cannot access base model ${BASE_MODEL}: {e}")
try:
    api.dataset_info("${DATASET}")
    print("  dataset OK:    ${DATASET}")
except Exception as e:
    sys.exit(f"cannot access dataset ${DATASET}: {e}")
PY

step "2.5/5  Verifying Unsloth + Qwen3.6 architecture compatibility"
python - <<'PY'
import os, sys, importlib
try:
    import unsloth
except Exception as e:
    sys.exit(f"Unsloth import failed — `pip install unsloth unsloth_zoo` did not succeed: {e}")

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(os.environ["BASE_MODEL"], token=os.environ.get("HF_TOKEN"))
print(f"  unsloth version: {getattr(unsloth, '__version__', 'unknown')}")
print(f"  model_type:      {cfg.model_type}")
print(f"  architectures:   {cfg.architectures}")
if cfg.model_type != "qwen3_5":
    sys.exit(
        f"BASE_MODEL has model_type={cfg.model_type!r}; this pipeline targets "
        f"qwen3_5 (Qwen3.5/Qwen3.6 family). For other models override BASE_MODEL "
        f"and adjust target_modules in train_sft.py if needed."
    )
print("  qwen3_5 model_type matches Unsloth's Qwen3.5/3.6 fine-tune path.")
PY

step "3/5  Training (LoRA SFT via Unsloth, full bf16)"
# Unsloth wraps device placement and gradient checkpointing internally;
# plain `python` is the documented invocation, not `accelerate launch`.
BASE_MODEL="$BASE_MODEL" DATASET="$DATASET" OUTPUT_DIR="$OUTPUT_DIR" \
    python scripts/cloud/train_sft.py

ADAPTER_DIR="${OUTPUT_DIR}/final_adapter"
[[ -d "$ADAPTER_DIR" ]] || fail "Training finished but $ADAPTER_DIR is missing."

step "4/5  Smoke-test generation"
python scripts/cloud/generate.py \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER_DIR" \
    --prompt "$SMOKE_PROMPT" \
    --max-new-tokens 512 \
    || warn "Smoke-test generation failed — adapter is still saved on disk."

step "5/5  Upload adapter to HuggingFace"
HF_URL=""
if [[ -n "${HF_REPO:-}" ]]; then
    python scripts/cloud/upload_to_hf.py
    HF_URL="https://huggingface.co/${HF_REPO}"
else
    warn "HF_REPO not set — skipping upload."
fi

step "Done"
echo "  adapter:        $ADAPTER_DIR"
echo "  checkpoints:    $OUTPUT_DIR/checkpoint-*"
[[ -n "$HF_URL" ]] && echo "  huggingface:    $HF_URL"
