#!/bin/bash
# End-to-end Phase 2 GRPO orchestrator. Runs on the pod.
#
#   Gate A — prerequisites
#   Gate B — generate katas (cl-ds + creative)
#   Gate C — baseline measurement on eval split
#   Gate D — 5-step smoke train
#   Gate E — full 500-step training
#   Gate F — save + upload adapter
#
# Any gate failure halts the run with a clear marker; the pod stays alive
# for inspection. Set FORCE=1 to skip baseline + smoke gates (use only
# after they've passed once).

set -uo pipefail
trap 'echo "[FATAL] line $LINENO exited $?" >&2; exit 1' ERR

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-./grpo-output}"
KATA_ROOT="${KATA_ROOT:-/workspace/katas}"
HF_REPO="${HF_REPO:-j14i/cl-macro-27b-grpo}"
SFT_ADAPTER="${SFT_ADAPTER:-j14i/cl-macro-27b-lora}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
FULL_STEPS="${FULL_STEPS:-500}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m[GATE FAIL] %s\033[0m\n' "$*" >&2; exit 1; }

# ─── Gate A: prerequisites ─────────────────────────────────────────────
step "Gate A: prerequisites"
[[ -n "${HF_TOKEN:-}" ]]              || fail "HF_TOKEN not set"
if ! command -v sbcl >/dev/null; then
    echo "  sbcl missing — invoking install_sbcl.sh"
    bash "$REPO_ROOT/scripts/cloud/install_sbcl.sh" || fail "install_sbcl.sh failed"
fi
command -v sbcl       >/dev/null     || fail "sbcl still not on PATH after install_sbcl.sh"
# macro-gym install: the upstream repo doesn't have a setup.py/pyproject.toml,
# so `pip install git+...` rejects it. Clone, drop a minimal pyproject.toml,
# `pip install -e`. Idempotent — the clone is skipped on subsequent boots.
if ! python -c "import macro_gym" 2>/dev/null; then
    MACRO_GYM_DIR="${MACRO_GYM_DIR:-/workspace/macro-gym}"
    if [[ ! -d "$MACRO_GYM_DIR" ]]; then
        echo "  cloning macro-gym to $MACRO_GYM_DIR"
        git clone https://github.com/jborkowski/macro-gym "$MACRO_GYM_DIR" || \
            fail "macro-gym clone failed"
    fi
    if [[ ! -f "$MACRO_GYM_DIR/pyproject.toml" ]]; then
        echo "  writing minimal pyproject.toml (upstream doesn't ship one)"
        cat > "$MACRO_GYM_DIR/pyproject.toml" <<'PYPROJ'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "macro-gym"
version = "0.1.0"
description = "Gymnasium env for CL macro generation via SBCL"
requires-python = ">=3.9"
dependencies = ["gymnasium"]

[tool.setuptools]
packages = ["macro_gym"]
PYPROJ
    fi
    pip install --quiet -e "$MACRO_GYM_DIR" || fail "macro-gym editable install failed"
fi
python -c "import macro_gym; from macro_gym import MacroEnv" \
                                      || fail "macro_gym.MacroEnv not importable"
python -c "from unsloth import FastLanguageModel; from trl import GRPOTrainer, GRPOConfig" \
                                      || fail "unsloth/trl not importable — run scripts/cloud/run.sh's pip install first"
df --output=avail -BG /workspace 2>/dev/null | tail -1 | awk '{exit (substr($1,1,length($1)-1) < 60)}' || \
    fail "less than 60 GB free in /workspace"
echo "  all checks passed"

# ─── Gate B: generate katas ───────────────────────────────────────────
step "Gate B: generate katas"
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
[[ $N_KATAS -ge 200 ]] || fail "only $N_KATAS katas survived validation — need >= 200"
echo "  $N_KATAS katas across cl-ds + creative"

# ─── Gate C: baseline measurement ──────────────────────────────────────
if [[ "${FORCE:-0}" != "1" ]]; then
    step "Gate C: baseline measurement (eval split, no training)"
    KATA_ROOT="$KATA_ROOT" SFT_ADAPTER="$SFT_ADAPTER" OUTPUT_DIR="$OUTPUT_DIR" \
        MAX_STEPS=0 BASELINE_ONLY=1 \
        python scripts/cloud/grpo_train.py 2>&1 | tee "$OUTPUT_DIR/baseline.log"
    BASELINE_MEAN=$(grep -oE 'baseline_mean=[0-9.-]+' "$OUTPUT_DIR/baseline.log" | tail -1 | cut -d= -f2)
    [[ -n "$BASELINE_MEAN" ]] || fail "couldn't parse baseline_mean from log"
    awk "BEGIN { exit !($BASELINE_MEAN > 0.05) }" || \
        fail "baseline mean reward $BASELINE_MEAN < 0.05 — too sparse; filter katas first"
    echo "  baseline mean reward: $BASELINE_MEAN  (>0.05, proceeding)"
fi

# ─── Gate D: smoke train ───────────────────────────────────────────────
if [[ "${FORCE:-0}" != "1" ]]; then
    step "Gate D: smoke train ($SMOKE_STEPS steps)"
    KATA_ROOT="$KATA_ROOT" SFT_ADAPTER="$SFT_ADAPTER" OUTPUT_DIR="$OUTPUT_DIR/smoke" \
        MAX_STEPS="$SMOKE_STEPS" \
        python scripts/cloud/grpo_train.py 2>&1 | tee "$OUTPUT_DIR/smoke.log"
    grep -q "Training summary" "$OUTPUT_DIR/smoke.log" || \
        fail "smoke train did not complete cleanly — see $OUTPUT_DIR/smoke.log"
    echo "  smoke train OK"
fi

# ─── Gate E: full training ─────────────────────────────────────────────
step "Gate E: full training ($FULL_STEPS steps)"
KATA_ROOT="$KATA_ROOT" SFT_ADAPTER="$SFT_ADAPTER" OUTPUT_DIR="$OUTPUT_DIR/full" \
    MAX_STEPS="$FULL_STEPS" \
    python scripts/cloud/grpo_train.py 2>&1 | tee "$OUTPUT_DIR/full.log"
grep -q "Training summary" "$OUTPUT_DIR/full.log" || \
    fail "full train aborted before save — see $OUTPUT_DIR/full.log"

# ─── Gate F: upload ───────────────────────────────────────────────────
step "Gate F: upload adapter to $HF_REPO"
OUTPUT_DIR="$OUTPUT_DIR/full" HF_REPO="$HF_REPO" python scripts/cloud/upload_to_hf.py
echo ""
echo "DONE. adapter at https://huggingface.co/$HF_REPO"
