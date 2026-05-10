# Environment variables — `scripts/cloud/run.sh`

Set these on the RunPod pod before invoking `bash scripts/cloud/run.sh`.

## Required

| Var        | Purpose |
|------------|---------|
| `HF_TOKEN` | HuggingFace token. Needed to download the gated `Qwen/Qwen3.6-27B` base model and (if `HF_REPO` is set) to push the trained adapter. |

## Optional

| Var             | Default                          | Purpose |
|-----------------|----------------------------------|---------|
| `HF_REPO`       | *(unset — upload skipped)*       | HuggingFace repo to push the LoRA adapter to, e.g. `j14i/cl-macro-27b-lora`. The repo is created if it does not exist. |
| `BASE_MODEL`    | `Qwen/Qwen3.6-27B`               | Base model id to fine-tune. |
| `DATASET`       | `j14i/cl-macros`                 | HF dataset id. Must expose either a `messages` chat-format column or `instruction`/`output` columns (the trainer auto-converts the latter). |
| `OUTPUT_DIR`    | `./output`                       | Training output directory. Adapter lands in `$OUTPUT_DIR/final_adapter`. |
| `WANDB_API_KEY` | *(unset — W&B disabled)*         | Weights & Biases API key for live training curves. |
| `SMOKE_PROMPT`  | a `when-let` macro prompt        | Prompt used by the post-training generation smoke-test. |

## Quick start

```bash
export HF_TOKEN=hf_xxx                       # required
export HF_REPO=j14i/cl-macro-27b-lora        # optional, enables upload
export WANDB_API_KEY=xxx                     # optional, enables W&B
bash scripts/cloud/run.sh
```

## Notes

- `Qwen/Qwen3.6-27B` is gated. Visit the model page on HuggingFace and accept
  the licence with the same account that owns `HF_TOKEN` before running.
- A100 80GB fits 27B bf16 + rank-32 LoRA + Adam states + activations at
  `batch_size=2`, `grad_accum=4`, `max_seq_length=4096` only with gradient
  checkpointing on (which the trainer enables). If you OOM, drop
  `per_device_train_batch_size` to 1 or `max_seq_length` to 2048 in
  `train_sft.py`.
- The thinking-trace (`<think>...</think>`) workflow is local-only for now —
  `j14i/cl-macros` doesn't yet ship reasoning traces. The system prompt in
  `generate.py` is kept ready for when traces are uploaded.
