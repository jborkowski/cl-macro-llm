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

# PHASE selects which pipeline runs after dep install:
#   sft  (default) — original Phase 1 SFT pipeline (steps 2-5 below)
#   grpo           — exec into scripts/cloud/run_grpo.sh after step 1
PHASE="${PHASE:-sft}"
case "$PHASE" in
    sft|grpo) ;;
    *) echo "ERROR: PHASE must be 'sft' or 'grpo' (got: $PHASE)" >&2; exit 1 ;;
esac

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
fail() { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

step "1/5  Installing python requirements (Unsloth from image; pip-installing direct deps)"
# If /workspace is a RunPod network volume, route HF model cache + pip
# wheel cache there so they survive pod recreation. Saves ~10-15 min per
# subsequent boot once the base model and wheels are seeded.
if [[ -d /workspace && -w /workspace ]]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
    mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
    echo "  HF_HOME=$HF_HOME"
    echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"
fi
# Drop --no-cache-dir when PIP_CACHE_DIR is set: caching wheels on the
# volume is the entire point of the network-volume optimization.
NO_CACHE_FLAG="--no-cache-dir"
[[ -n "${PIP_CACHE_DIR:-}" ]] && NO_CACHE_FLAG=""
# When launched via the official Unsloth template (pzr9tt3vvq), unsloth +
# unsloth_zoo + a verified torch/CUDA combo are baked into the image — we
# skip the --force-reinstall to avoid breaking that. Direct deps still need pip.
python -c "import unsloth" 2>/dev/null \
    && echo "  unsloth already installed: $(python -c 'import unsloth; print(unsloth.__version__)')" \
    || pip install --upgrade --force-reinstall $NO_CACHE_FLAG unsloth unsloth_zoo
pip install -r scripts/cloud/requirements.txt -q

# Best-effort: fast kernels for Qwen3.6's Gated DeltaNet layers (3 of every
# 4 layers). Without these, training falls back to a PyTorch implementation
# that pegs CPU at 100% and idles the GPU at ~11% — first step ~190s, run
# ETA ~36 hours. With them, ~5-30s/step.
#
# causal-conv1d has no prebuilt wheel for torch 2.10+cu128, so it must
# build from source. The runpod/pytorch:*-devel image has CUDA 12.4 nvcc
# at /usr/local/cuda-12.4/bin but doesn't put it on PATH — export it
# before the install. The CUDA 12.4 nvcc vs torch 12.8 CUDA libs minor-
# version mismatch is tolerated. See TROUBLESHOOTING.md §1.
echo "  -- optional: fast DeltaNet kernels (best-effort, ~5-10 min if they build) --"
# Find the newest CUDA toolkit on the image and put its nvcc on PATH.
# Required for source-building causal-conv1d. The cu1281 image has CUDA
# 12.8 (sm_120 / Blackwell-capable); older images have 12.4 (A100/H100).
for d in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda; do
    if [[ -x "$d/bin/nvcc" ]]; then
        export PATH="$d/bin:$PATH"
        export CUDA_HOME="$d"
        echo "  using nvcc at $d/bin/nvcc ($($d/bin/nvcc --version | tail -1))"
        break
    fi
done
# Detect GPU arch at runtime and target it precisely (smaller wheel, faster
# build). Falls back to a broad list covering A100 / H100 / Blackwell.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    [[ -n "$GPU_CAP" ]] && echo "  detected GPU compute capability: $GPU_CAP"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_CAP:-8.0;9.0;12.0}}"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
export MAX_JOBS="${MAX_JOBS:-4}"

python -c "import fla" 2>/dev/null \
    && echo "  flash-linear-attention already installed" \
    || pip install flash-linear-attention 2>&1 | tail -3 \
    || warn "flash-linear-attention install failed; DeltaNet uses slow PyTorch fallback."
python -c "from causal_conv1d import causal_conv1d_fn" 2>/dev/null \
    && echo "  causal-conv1d already installed (fast kernel imports cleanly)" \
    || pip install --no-build-isolation --no-cache-dir causal-conv1d 2>&1 | tail -3 \
    || warn "causal-conv1d install failed (likely no nvcc on PATH); see TROUBLESHOOTING.md §1."

# ─── Phase 2 hand-off ─────────────────────────────────────────────────
# run_grpo.sh runs its own gates (A: deps recheck + sbcl + macro-gym, B:
# kata gen, C: baseline, D: smoke, E: full train, F: upload). It depends
# on the Unsloth/trl install layer above, then takes over end-to-end.
if [[ "$PHASE" == "grpo" ]]; then
    step "PHASE=grpo — installs complete, handing off to run_grpo.sh"
    exec bash scripts/cloud/run_grpo.sh
fi

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
