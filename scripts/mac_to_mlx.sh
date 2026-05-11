#!/bin/bash
# Mac-side: download base + LoRA adapter from HuggingFace, merge them on
# CPU, then quantize to native Apple Silicon MLX 4-bit. End-to-end
# ~1-2 h on a 128 GB Mac. Requires ~120 GB free disk (54 GB cached base +
# 54 GB merged + 14 GB MLX 4-bit output).
#
# Uses `uv run --with ...` so no global pip required. `uv` itself must
# be on PATH (already true on this Mac).
#
# Prereqs:
#   - uv (https://docs.astral.sh/uv/)
#   - .env with HF_TOKEN, HF_REPO

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }

: "${HF_TOKEN:?HF_TOKEN not set}"
: "${HF_REPO:?HF_REPO not set (e.g. j14i/cl-macro-27b-lora)}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B}"
MERGED_DIR="${MERGED_DIR:-$HOME/models/$(basename $HF_REPO)-merged}"
MLX_DIR="${MLX_DIR:-$HOME/models/$(basename $HF_REPO)-mlx-q4}"
QBITS="${QBITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-64}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
fail() { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || fail "uv not on PATH. install: curl -LsSf https://astral.sh/uv/install.sh | sh"

# Reusable uv-run launchers with all our deps. First call resolves + caches
# the dep set; subsequent calls reuse the cache.
MERGE_DEPS=(--with "transformers>=5" --with "peft>=0.13" --with "torch" --with "huggingface_hub" --with "safetensors")
MLX_DEPS=(--with "mlx-lm")

step "1/3  Checking uv + warming dep caches"
echo "  uv: $(uv --version)"
uv run "${MERGE_DEPS[@]}" python -c "
import transformers, peft, torch
print(f'  transformers: {transformers.__version__}')
print(f'  peft:         {peft.__version__}')
print(f'  torch:        {torch.__version__}')
" || fail "merge deps install failed"
uv run "${MLX_DEPS[@]}" python -c "
import mlx_lm
print(f'  mlx-lm:       {getattr(mlx_lm, \"__version__\", \"installed\")}')" || fail "mlx-lm install failed"

step "2/3  Merging adapter into base ($BASE_MODEL + $HF_REPO -> $MERGED_DIR)"
if [[ -d "$MERGED_DIR" && -f "$MERGED_DIR/config.json" ]]; then
    echo "  Merged dir already exists, skipping merge: $MERGED_DIR"
else
    uv run "${MERGE_DEPS[@]}" python - <<PY
import os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id    = os.environ.get("BASE_MODEL", "$BASE_MODEL")
adapter_id = os.environ.get("HF_REPO",    "$HF_REPO")
merged_dir = Path(os.environ.get("MERGED_DIR", "$MERGED_DIR"))
token      = os.environ["HF_TOKEN"]

print(f"  base:    {base_id}")
print(f"  adapter: {adapter_id}")
print(f"  output:  {merged_dir}")
print("  Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(adapter_id, token=token)
print("  Loading base in bf16 (~54 GB, ~5-15 min depending on cache)...")
base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, token=token, trust_remote_code=False,
)
print("  Attaching adapter...")
model = PeftModel.from_pretrained(base, adapter_id, token=token)
print("  Merging (~5-10 min)...")
model = model.merge_and_unload()
print(f"  Saving to {merged_dir} (~54 GB write, ~5-10 min)...")
merged_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(merged_dir))
tok.save_pretrained(str(merged_dir))
print("  Merge done.")
PY
fi

step "3/3  Quantizing to MLX 4-bit ($MERGED_DIR -> $MLX_DIR)"
if [[ -d "$MLX_DIR" && -f "$MLX_DIR/config.json" ]]; then
    echo "  MLX dir already exists, skipping convert: $MLX_DIR"
else
    uv run "${MLX_DEPS[@]}" python -m mlx_lm.convert \
        --hf-path  "$MERGED_DIR" \
        --mlx-path "$MLX_DIR" \
        --quantize \
        --q-bits "$QBITS" \
        --q-group-size "$GROUP_SIZE"
fi

step "Done"
echo "  merged HF:  $MERGED_DIR"
echo "  MLX 4-bit:  $MLX_DIR"
echo ""
echo "Test:"
echo "  uv run --with mlx-lm python -m mlx_lm.generate \\"
echo "      --model $MLX_DIR \\"
echo "      --prompt 'Write a Common Lisp macro \`when-let\` that binds a value and runs body only when non-nil.' \\"
echo "      --max-tokens 1024 --temp 0.6"
