#!/bin/bash
# Boot an A100 80GB pod on RunPod, run the SFT pipeline end-to-end,
# tear the pod down on exit.
#
# Prerequisites:
#   - `runpodctl` installed locally (https://github.com/runpod/runpodctl)
#   - `jq` installed locally
#   - `.env` in the repo root (copy from `.env.example`) with:
#       RUNPOD_API_KEY, RUNPOD_EXPECTED_EMAIL, HF_TOKEN, HF_REPO
#
# Safeties:
#   1. Authenticates runpodctl with $RUNPOD_API_KEY, then queries the
#      RunPod GraphQL API to confirm the authenticated user's email
#      matches $RUNPOD_EXPECTED_EMAIL — refuses to continue otherwise.
#      Protects against booting on a company account by mistake.
#   2. Refuses to boot if $HF_REPO is unset AND $KEEP_POD is unset —
#      otherwise the adapter dies with the pod.
#   3. Trap on EXIT removes the pod unless $KEEP_POD=1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
fail() { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. Load .env ────────────────────────────────────────────────────
[[ -f "$ENV_FILE" ]] || fail "$ENV_FILE not found — copy .env.example to .env and fill it in."

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing from $ENV_FILE}"
: "${RUNPOD_EXPECTED_EMAIL:?RUNPOD_EXPECTED_EMAIL missing from $ENV_FILE}"
: "${HF_TOKEN:?HF_TOKEN missing from $ENV_FILE}"

# ── 2. Tool checks ──────────────────────────────────────────────────
command -v runpodctl >/dev/null || fail "runpodctl not installed. brew install runpodctl, or see https://github.com/runpod/runpodctl"
command -v jq        >/dev/null || fail "jq not installed."

# ── 3. Verify RunPod account ────────────────────────────────────────
step "Verifying RunPod account"
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null

CURRENT_EMAIL=$(curl -fsS \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query":"{ myself { email id } }"}' \
    https://api.runpod.io/graphql \
    | jq -r '.data.myself.email // empty')

if [[ -z "$CURRENT_EMAIL" ]]; then
    fail "Could not resolve current RunPod user. Bad API key?"
fi
if [[ "$CURRENT_EMAIL" != "$RUNPOD_EXPECTED_EMAIL" ]]; then
    fail "Account mismatch — expected $RUNPOD_EXPECTED_EMAIL, got $CURRENT_EMAIL. Refusing to boot."
fi
echo "  authenticated as: $CURRENT_EMAIL"

# ── 4. Adapter-survival precondition ────────────────────────────────
if [[ -z "${HF_REPO:-}" && -z "${KEEP_POD:-}" ]]; then
    fail "HF_REPO is unset AND KEEP_POD is unset — adapter would be destroyed with the pod. Set HF_REPO in .env, or export KEEP_POD=1 to keep the pod alive after training."
fi

# ── 5. Boot pod ─────────────────────────────────────────────────────
POD_NAME="cl-macro-sft-$(date +%s)"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA A100 80GB PCIe}"
DISK_GB="${RUNPOD_DISK_GB:-100}"

step "Booting pod: $POD_NAME ($GPU_TYPE, $DISK_GB GB disk, image: $IMAGE)"

POD_ENV_ARGS=(--env "HF_TOKEN=$HF_TOKEN")
[[ -n "${HF_REPO:-}"         ]] && POD_ENV_ARGS+=(--env "HF_REPO=$HF_REPO")
[[ -n "${WANDB_API_KEY:-}"   ]] && POD_ENV_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
[[ -n "${BASE_MODEL:-}"      ]] && POD_ENV_ARGS+=(--env "BASE_MODEL=$BASE_MODEL")
[[ -n "${DATASET:-}"         ]] && POD_ENV_ARGS+=(--env "DATASET=$DATASET")
[[ -n "${MAX_SEQ_LENGTH:-}"  ]] && POD_ENV_ARGS+=(--env "MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH")

POD_JSON=$(runpodctl create pod \
    --name "$POD_NAME" \
    --imageName "$IMAGE" \
    --gpuType "$GPU_TYPE" \
    --gpuCount 1 \
    --containerDiskSize "$DISK_GB" \
    --ports "22/tcp" \
    "${POD_ENV_ARGS[@]}" \
    --json)

POD_ID=$(echo "$POD_JSON" | jq -r '.id // empty')
[[ -n "$POD_ID" ]] || fail "Could not parse pod id from runpodctl output:\n$POD_JSON"
echo "  pod id: $POD_ID"

cleanup() {
    if [[ -n "${KEEP_POD:-}" ]]; then
        warn "KEEP_POD set — pod $POD_ID left running. Remove later with: runpodctl remove pod $POD_ID"
    else
        step "Tearing down pod $POD_ID"
        runpodctl remove pod "$POD_ID" || warn "Pod cleanup failed — remove manually: runpodctl remove pod $POD_ID"
    fi
}
trap cleanup EXIT

# ── 6. Wait for pod to come online ──────────────────────────────────
step "Waiting for pod to become reachable (up to 5 min)"
deadline=$(( $(date +%s) + 300 ))
until runpodctl ssh "$POD_ID" -- "echo ready" >/dev/null 2>&1; do
    [[ $(date +%s) -lt $deadline ]] || fail "Pod did not come online within 5 minutes."
    sleep 5
done
echo "  pod reachable."

# ── 7. Run pipeline on pod ──────────────────────────────────────────
step "Running training pipeline on pod"
runpodctl ssh "$POD_ID" -- bash -lc "
    set -e
    cd /workspace
    if [[ ! -d cl-macro-llm ]]; then
        git clone https://github.com/jborkowski/cl-macro-llm.git
    fi
    cd cl-macro-llm
    git pull --ff-only
    bash scripts/cloud/run.sh
"

step "Done — training run finished."
[[ -n "${HF_REPO:-}" ]] && echo "  adapter on HF: https://huggingface.co/$HF_REPO"
