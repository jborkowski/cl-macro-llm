#!/usr/bin/env python3
"""Post-training: merge LoRA adapter into the base model and ship deployment
artifacts to HuggingFace.

Produces (best-effort, partial success is fine):

  <HF_REPO>-merged   — full bf16 merged model
                       (input for MLX 4-bit conversion on Mac;
                       see scripts/to_mlx.sh)
  <HF_REPO>-gguf     — one or more GGUF quantizations
                       (Q4_K_M for size, Q5_K_M for quality, Q8_0 for headroom)
                       usable in llama.cpp, Ollama, LM Studio.

Qwen3.6 (model_type qwen3_5) is brand-new — llama.cpp's converter may not
support the hybrid Gated DeltaNet architecture yet. GGUF steps wrap in
try/except so 16-bit merge can succeed even if quant fails.

Env vars:
  HF_REPO        required — base name for variants
  HF_TOKEN       required — write access
  BASE_MODEL     default: Qwen/Qwen3.6-27B
  OUTPUT_DIR     default: ./output
  MAX_SEQ_LENGTH default: 4096
  GGUF_METHODS   comma-separated, default: q4_k_m,q5_k_m,q8_0
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from unsloth import FastLanguageModel


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def main() -> int:
    output_dir = Path(_env("OUTPUT_DIR", "./output"))
    adapter_dir = output_dir / "final_adapter"
    hf_repo = _env("HF_REPO")
    hf_token = _env("HF_TOKEN")
    max_seq = int(_env("MAX_SEQ_LENGTH", "4096"))
    gguf_methods = [m.strip() for m in _env(
        "GGUF_METHODS", "q4_k_m,q5_k_m,q8_0"
    ).split(",") if m.strip()]

    if not adapter_dir.exists():
        print(f"Adapter not found: {adapter_dir}", file=sys.stderr)
        return 1
    if not hf_repo or not hf_token:
        print("HF_REPO and HF_TOKEN are required to push artifacts.", file=sys.stderr)
        return 1

    print(f"Loading base + adapter from {adapter_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=hf_token,
    )

    # ─── Merged 16-bit ────────────────────────────────────────────────
    merged_repo = f"{hf_repo}-merged"
    print(f"\n== Pushing merged bf16 to {merged_repo} ==")
    try:
        model.push_to_hub_merged(
            merged_repo,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
        print(f"   https://huggingface.co/{merged_repo}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        # If merge fails everything downstream is doomed, but adapter still
        # lives on disk + as the primary HF_REPO upload from upload_to_hf.py.
        return 2

    # ─── GGUF variants ────────────────────────────────────────────────
    gguf_repo = f"{hf_repo}-gguf"
    successful = []
    failed = []
    for method in gguf_methods:
        print(f"\n== Pushing GGUF {method} to {gguf_repo} ==")
        try:
            model.push_to_hub_gguf(
                gguf_repo,
                tokenizer,
                quantization_method=method,
                token=hf_token,
            )
            print(f"   https://huggingface.co/{gguf_repo} ({method})")
            successful.append(method)
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")
            failed.append((method, str(e)))

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n=== Postprocess summary ===")
    print(f"  merged 16-bit: https://huggingface.co/{merged_repo}")
    if successful:
        print(f"  GGUF ok:       {', '.join(successful)} → https://huggingface.co/{gguf_repo}")
    if failed:
        print("  GGUF failed (likely llama.cpp lacks qwen3_5 support yet):")
        for method, err in failed:
            print(f"    - {method}: {err[:120]}")
    print("\nFor MLX 4-bit on Mac: bash scripts/to_mlx.sh")
    return 0 if successful or not failed else 0  # return 0 even with partial


if __name__ == "__main__":
    sys.exit(main())
