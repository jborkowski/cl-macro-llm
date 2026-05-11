#!/bin/bash
# Boot an A100 80GB pod on RunPod, run the SFT pipeline end-to-end,
# tear the pod down on exit.
#
# Prerequisites:
#   - `runpodctl` 2.x (modern `runpodctl pod ...` syntax)
#   - `jq`, `ssh`, `curl` installed locally
#   - SSH key registered with RunPod (auto-managed key in
#     ~/.runpod/ssh/RunPod-Key-Go is used by default)
#   - `.env` in the repo root (copy from `.env.example`)
#
# Safeties:
#   1. Confirms authenticated user's email matches $RUNPOD_EXPECTED_EMAIL
#      before booting — protects against shipping work to a company account.
#   2. Refuses to boot if $HF_REPO is unset AND $KEEP_POD is unset —
#      otherwise the adapter dies with the pod.
#   3. EXIT trap deletes the pod unless $KEEP_POD=1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/RunPod-Key-Go}"

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

# runpodctl reads RUNPOD_API_KEY from env, no separate config step needed
export RUNPOD_API_KEY

# ── 2. Tool checks ──────────────────────────────────────────────────
command -v runpodctl >/dev/null || fail "runpodctl not installed."
command -v jq        >/dev/null || fail "jq not installed."
command -v ssh       >/dev/null || fail "ssh not installed."
command -v curl      >/dev/null || fail "curl not installed."
[[ -f "$SSH_KEY" ]]              || fail "SSH private key not found at $SSH_KEY (set RUNPOD_SSH_KEY to override)."

# ── 3. Verify RunPod account ────────────────────────────────────────
step "Verifying RunPod account"
CURRENT_EMAIL=$(curl -fsS \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query":"{ myself { email id } }"}' \
    https://api.runpod.io/graphql \
    | jq -r '.data.myself.email // empty')

[[ -n "$CURRENT_EMAIL" ]] || fail "Could not resolve current RunPod user. Bad API key?"
if [[ "$CURRENT_EMAIL" != "$RUNPOD_EXPECTED_EMAIL" ]]; then
    fail "Account mismatch — expected $RUNPOD_EXPECTED_EMAIL, got $CURRENT_EMAIL. Refusing to boot."
fi
echo "  authenticated as: $CURRENT_EMAIL"

# ── 4. Adapter-survival precondition ────────────────────────────────
if [[ -z "${HF_REPO:-}" && -z "${KEEP_POD:-}" ]]; then
    fail "HF_REPO is unset AND KEEP_POD is unset — adapter would be destroyed with the pod. Set HF_REPO in .env, or export KEEP_POD=1 to keep the pod alive."
fi

# ── 5. Boot pod ─────────────────────────────────────────────────────
POD_NAME="cl-macro-sft-$(date +%s)"
# Default to a small pytorch image (~5 GB, fast pull on community cloud).
# The official Unsloth template (pzr9tt3vvq, unsloth/unsloth:latest) avoids
# the pip --force-reinstall risk but is 13+ GB and takes 30-50 min to
# extract on cold community-cloud hosts. Override via env if you want it:
#     RUNPOD_TEMPLATE_ID=pzr9tt3vvq bash scripts/cloud/launch.sh
TEMPLATE_ID="${RUNPOD_TEMPLATE_ID:-}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
GPU_ID="${RUNPOD_GPU_ID:-NVIDIA A100 80GB PCIe}"
DISK_GB="${RUNPOD_DISK_GB:-100}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"

# Build the env JSON object — modern `runpodctl pod create --env` expects
# a single JSON object, not repeated --env KEY=VAL flags.
ENV_JSON=$(jq -n \
    --arg HF_TOKEN     "$HF_TOKEN" \
    --arg HF_REPO      "${HF_REPO:-}" \
    --arg WANDB        "${WANDB_API_KEY:-}" \
    --arg BASE_MODEL   "${BASE_MODEL:-}" \
    --arg DATASET      "${DATASET:-}" \
    --arg MAX_SEQ      "${MAX_SEQ_LENGTH:-}" \
    '{HF_TOKEN: $HF_TOKEN}
     + (if $HF_REPO     != "" then {HF_REPO: $HF_REPO} else {} end)
     + (if $WANDB       != "" then {WANDB_API_KEY: $WANDB} else {} end)
     + (if $BASE_MODEL  != "" then {BASE_MODEL: $BASE_MODEL} else {} end)
     + (if $DATASET     != "" then {DATASET: $DATASET} else {} end)
     + (if $MAX_SEQ     != "" then {MAX_SEQ_LENGTH: $MAX_SEQ} else {} end)')

if [[ -n "$TEMPLATE_ID" ]]; then
    step "Booting pod: $POD_NAME (template $TEMPLATE_ID, $GPU_ID, $DISK_GB GB disk, $CLOUD_TYPE cloud)"
    IMAGE_FLAG=(--template-id "$TEMPLATE_ID")
else
    step "Booting pod: $POD_NAME (image $IMAGE, $GPU_ID, $DISK_GB GB disk, $CLOUD_TYPE cloud)"
    IMAGE_FLAG=(--image "$IMAGE")
fi

CREATE_ARGS=(
    --name "$POD_NAME"
    "${IMAGE_FLAG[@]}"
    --gpu-id "$GPU_ID"
    --gpu-count 1
    --container-disk-in-gb "$DISK_GB"
    --ports "22/tcp"
    --cloud-type "$CLOUD_TYPE"
    --env "$ENV_JSON"
)
# Community cloud SSH requires a public IP; secure cloud routes via proxy.
[[ "$CLOUD_TYPE" == "COMMUNITY" ]] && CREATE_ARGS+=(--public-ip)

CREATE_OUT=$(runpodctl pod create "${CREATE_ARGS[@]}" 2>&1)
POD_ID=$(echo "$CREATE_OUT" | jq -r '.id // empty' 2>/dev/null)
if [[ -z "$POD_ID" ]]; then
    # Fallback parsing for non-pure-JSON output
    POD_ID=$(echo "$CREATE_OUT" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"([^"]+)"$/\1/')
fi
[[ -n "$POD_ID" ]] || fail "Could not parse pod id from runpodctl output:\n$CREATE_OUT"
echo "  pod id: $POD_ID"

cleanup() {
    if [[ -n "${KEEP_POD:-}" ]]; then
        warn "KEEP_POD set — pod $POD_ID left running. Remove later with: runpodctl pod delete $POD_ID"
    else
        step "Tearing down pod $POD_ID"
        runpodctl pod delete "$POD_ID" || warn "Pod cleanup failed — remove manually: runpodctl pod delete $POD_ID"
    fi
}
trap cleanup EXIT INT TERM HUP

# ── 6. Wait for SSH details ─────────────────────────────────────────
step "Waiting for pod SSH details (up to 5 min)"
deadline=$(( $(date +%s) + 300 ))
SSH_HOST=""
SSH_PORT=""
SSH_USER="root"
while [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; do
    [[ $(date +%s) -lt $deadline ]] || fail "Pod did not expose SSH within 5 minutes."
    sleep 8
    INFO=$(runpodctl ssh info "$POD_ID" 2>/dev/null || echo "{}")
    SSH_HOST=$(echo "$INFO" | jq -r '.ip // empty' 2>/dev/null)
    SSH_PORT=$(echo "$INFO" | jq -r '.port // empty' 2>/dev/null)
    if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
        CMD=$(echo "$INFO" | jq -r '.ssh_command // empty' 2>/dev/null)
        if [[ -n "$CMD" ]]; then
            SSH_PORT=$(echo "$CMD" | grep -oE -- '-p [0-9]+' | awk '{print $2}')
            SSH_HOST=$(echo "$CMD" | grep -oE '[a-z]+@[^ ]+' | tail -1 | cut -d@ -f2)
            SSH_USER=$(echo "$CMD" | grep -oE '[a-z]+@[^ ]+' | tail -1 | cut -d@ -f1)
        fi
    fi
done
echo "  ssh: $SSH_USER@$SSH_HOST -p $SSH_PORT"

# ── 7. Wait for sshd ────────────────────────────────────────────────
step "Waiting for sshd to accept connections"
deadline=$(( $(date +%s) + 180 ))
until ssh -i "$SSH_KEY" \
          -o IdentitiesOnly=yes \
          -o StrictHostKeyChecking=accept-new \
          -o ConnectTimeout=5 \
          -o BatchMode=yes \
          -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "echo ready" >/dev/null 2>&1; do
    [[ $(date +%s) -lt $deadline ]] || fail "sshd did not respond within 3 minutes."
    sleep 5
done
echo "  ssh ready."

# ── 8. Run pipeline on pod ──────────────────────────────────────────
# The container `--env` JSON we passed to `runpodctl pod create` sets env vars
# at PID 1, but RunPod's default sshd doesn't expose them to login shells.
# Pipe the script over stdin so we can `export` the secrets explicitly in the
# remote shell — also avoids the SSH-arg-quoting trap that breaks `bash -lc`.
# Build env-export block; skip vars whose local value is empty/unset, so the
# remote shell doesn't see e.g. `export MAX_SEQ_LENGTH=''` (which then takes
# precedence over Python-side defaults).
ENV_EXPORTS="export HF_TOKEN='$HF_TOKEN'"
for var in HF_REPO WANDB_API_KEY WANDB_ENTITY BASE_MODEL DATASET MAX_SEQ_LENGTH PHASE; do
    val="${!var:-}"
    [[ -n "$val" ]] && ENV_EXPORTS="$ENV_EXPORTS
export $var='$val'"
done

step "Running training pipeline on pod"
ssh -i "$SSH_KEY" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" bash -ls <<EOF
set -e
$ENV_EXPORTS
mkdir -p /workspace
cd /workspace
if [[ ! -d cl-macro-llm ]]; then
    git clone https://github.com/jborkowski/cl-macro-llm.git
fi
cd cl-macro-llm
git pull --ff-only || true
bash scripts/cloud/run.sh
EOF

step "Done — training run finished."
[[ -n "${HF_REPO:-}" ]] && echo "  adapter on HF: https://huggingface.co/$HF_REPO"
