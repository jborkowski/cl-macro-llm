#!/bin/bash
# Convert the trained+merged HF model to native Apple Silicon MLX 4-bit.
# Run this on your Mac (NOT the RunPod pod — mlx-lm is Apple Silicon only).
#
# Prerequisites:
#   - postprocess.py has already pushed <HF_REPO>-merged to HuggingFace
#   - HF_TOKEN exported (read access to the merged repo)
#   - mlx-lm installed: `pip install --upgrade mlx-lm`
#
# Output: a self-contained MLX model directory ready for
#   `python -m mlx_lm.generate --model <dir> --prompt "..."`

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

# Source .env if present (gives us HF_TOKEN, HF_REPO)
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }

: "${HF_REPO:?HF_REPO not set — point at the BASE name (script appends -merged)}"
: "${HF_TOKEN:?HF_TOKEN not set — needed for HF download}"

MERGED_REPO="${MERGED_REPO:-${HF_REPO}-merged}"
OUT_DIR="${OUT_DIR:-./models/cl-macro-27b-mlx-q4}"
QBITS="${QBITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-64}"

echo "==> Converting $MERGED_REPO → $OUT_DIR (q-bits=$QBITS group=$GROUP_SIZE)"
echo "    This pulls ~54 GB from HF and writes ~14 GB locally. First time only."

command -v python >/dev/null || { echo "python not on PATH" >&2; exit 1; }

# mlx-lm convert is the official path. -q enables quantization.
python -m mlx_lm.convert \
    --hf-path "$MERGED_REPO" \
    --mlx-path "$OUT_DIR" \
    --quantize \
    --q-bits "$QBITS" \
    --q-group-size "$GROUP_SIZE" \
    --hf-token "$HF_TOKEN"

echo ""
echo "==> Done."
echo "    Test generation:"
echo "      python -m mlx_lm.generate \\"
echo "          --model $OUT_DIR \\"
echo "          --prompt 'Write a Common Lisp macro when-let that binds a value and runs body only if non-nil.' \\"
echo "          --max-tokens 512 --temp 0.6"
echo ""
echo "    Or push it to HuggingFace as a third variant:"
echo "      python -m mlx_lm.convert --hf-path $MERGED_REPO --mlx-path $OUT_DIR \\"
echo "                                 --quantize --q-bits $QBITS \\"
echo "                                 --upload-repo ${HF_REPO}-mlx-q$QBITS"
