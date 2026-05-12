#!/bin/bash
# Phase 2 GRPO orchestrator. Runs on the pod after run.sh has installed
# Unsloth + the fast DeltaNet kernels.
#
# Three linear steps, no skip flags:
#   1. prereqs  — sbcl, macro-gym v0.3, kata corpus (regenerates if missing)
#   2. train    — GRPO over `FULL_STEPS` steps, samples + checkpoints in
#                 `OUTPUT_DIR/full/`, push to HF Hub every save_steps
#   3. upload   — push the final adapter to `HF_REPO`
#
# The baseline-measurement and smoke-train gates were retired (2026-05-12):
# they doubled model-load time and the only failure mode they ever caught
# was "the model is producing nonsense", which the first 25 steps of
# training surface clearly enough via the rolling reward in metrics.jsonl.
#
# Override anything via env:
#   FULL_STEPS=200 OUTPUT_DIR=./alt-output bash scripts/cloud/run_grpo.sh

set -uo pipefail
trap 'echo "[FATAL] line $LINENO exited $?" >&2; exit 1' ERR

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-./grpo-output}"
KATA_ROOT="${KATA_ROOT:-/workspace/katas}"
HF_REPO="${HF_REPO:-j14i/cl-macro-27b-grpo}"
SFT_ADAPTER="${SFT_ADAPTER:-j14i/cl-macro-27b-lora}"
FULL_STEPS="${FULL_STEPS:-500}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m[FAIL] %s\033[0m\n' "$*" >&2; exit 1; }

# ─── 1. prereqs ───────────────────────────────────────────────────────
step "1/3 prereqs"
[[ -n "${HF_TOKEN:-}" ]]              || fail "HF_TOKEN not set"
if ! command -v sbcl >/dev/null; then
    echo "  sbcl missing — invoking install_sbcl.sh"
    bash "$REPO_ROOT/scripts/cloud/install_sbcl.sh" || fail "install_sbcl.sh failed"
fi
command -v sbcl >/dev/null            || fail "sbcl still not on PATH after install_sbcl.sh"

# macro-gym v0.3: ships pyproject.toml + the verifier-first MacroGrader
# API. Plain `pip install -e` is enough — no downstream patching.
MACRO_GYM_DIR="${MACRO_GYM_DIR:-/workspace/macro-gym}"
if ! python -c "from macro_gym import MacroGrader" 2>/dev/null; then
    if [[ ! -d "$MACRO_GYM_DIR" ]]; then
        echo "  cloning macro-gym to $MACRO_GYM_DIR"
        git clone https://github.com/jborkowski/macro-gym "$MACRO_GYM_DIR" || \
            fail "macro-gym clone failed"
    else
        # Pull v0.3+ into an existing checkout.
        (cd "$MACRO_GYM_DIR" && git pull --ff-only) || \
            echo "  warning: macro-gym git pull failed; using local checkout"
    fi
    pip install --quiet -e "$MACRO_GYM_DIR" || fail "macro-gym editable install failed"
fi
python -c "from macro_gym import MacroGrader, get_grader, shutdown_grader" \
    || fail "macro_gym v0.3 API not importable — old fork?"
python -c "from unsloth import FastLanguageModel; from trl import GRPOTrainer, GRPOConfig" \
    || fail "unsloth/trl not importable — run scripts/cloud/run.sh's pip install first"

# Katas: regenerate the cl-ds + creative corpora only if either is missing
# or thin. The trainer adds its own setup.lisp/tests.lisp validation.
if [[ ! -d "$KATA_ROOT/cl-ds" || $(find "$KATA_ROOT/cl-ds" -maxdepth 1 -type d | wc -l) -lt 100 ]]; then
    echo "  generating cl-ds katas..."
    uv run scripts/cloud/cl_ds_to_katas.py \
        --dataset j14i/cl-ds --split train \
        --output-dir "$KATA_ROOT/cl-ds" \
        --validate --workers 8 || fail "cl-ds kata generation failed"
fi
if [[ ! -d "$KATA_ROOT/creative" || $(find "$KATA_ROOT/creative" -maxdepth 1 -type d | wc -l) -lt 50 ]]; then
    echo "  generating creative katas..."
    uv run scripts/cloud/cl_ds_to_katas.py \
        --dataset j14i/cl-macros-creative --split train \
        --output-dir "$KATA_ROOT/creative" \
        --validate --workers 8 || fail "creative kata generation failed"
fi
N_KATAS=$(find "$KATA_ROOT" -mindepth 2 -maxdepth 2 -type d -not -name '_*' | wc -l)
[[ $N_KATAS -ge 200 ]] || fail "only $N_KATAS katas survived validation — need ≥ 200"
echo "  $N_KATAS katas across cl-ds + creative"

df -h /workspace 2>/dev/null | tail -1 | awk '{print "  /workspace: " $4 " free of " $2}' || true
echo "  prereqs OK"

# ─── 2. train ─────────────────────────────────────────────────────────
step "2/3 train ($FULL_STEPS steps)"
mkdir -p "$OUTPUT_DIR/full"
# Memory-tight Unsloth knobs. Our OOM trace lands in
# `chunked_hidden_states_selective_log_softmax` (2.37 GB scratch on top of
# 77.89 GB used / 79.26 GB total). Setting both knobs to 1 forces the
# smallest log-softmax chunk and one-completion-at-a-time GRPO inner loop —
# slower, but trades time for VRAM at full bf16. Override either knob from
# the caller if a future GPU has slack.
KATA_ROOT="$KATA_ROOT" SFT_ADAPTER="$SFT_ADAPTER" OUTPUT_DIR="$OUTPUT_DIR/full" \
    MAX_STEPS="$FULL_STEPS" \
    UNSLOTH_LOGIT_CHUNK_MULTIPLIER="${UNSLOTH_LOGIT_CHUNK_MULTIPLIER:-1}" \
    UNSLOTH_GRPO_MINI_BATCH="${UNSLOTH_GRPO_MINI_BATCH:-1}" \
    python scripts/cloud/grpo_train.py 2>&1 | tee "$OUTPUT_DIR/full.log"
grep -q "Training summary" "$OUTPUT_DIR/full.log" || \
    fail "train aborted before final save — see $OUTPUT_DIR/full.log"

# ─── 3. upload ────────────────────────────────────────────────────────
step "3/3 upload adapter to $HF_REPO"
OUTPUT_DIR="$OUTPUT_DIR/full" HF_REPO="$HF_REPO" python scripts/cloud/upload_to_hf.py
echo
echo "DONE. adapter at https://huggingface.co/$HF_REPO"
