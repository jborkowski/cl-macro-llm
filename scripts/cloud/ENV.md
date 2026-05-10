# Environment variables — `scripts/cloud/run.sh`

Set these on the RunPod pod before invoking `bash scripts/cloud/run.sh`.

## Required

| Var        | Purpose |
|------------|---------|
| `HF_TOKEN` | HuggingFace token. Needed to upload the trained adapter (if `HF_REPO` is set) and to query model/dataset metadata reliably. `Qwen/Qwen3.6-27B` is public (Apache 2.0), but passing a token avoids HF rate limits and keeps the auth path consistent. |

## Optional

| Var               | Default                          | Purpose |
|-------------------|----------------------------------|---------|
| `HF_REPO`         | *(unset — upload skipped)*       | HuggingFace repo to push the LoRA adapter to, e.g. `j14i/cl-macro-27b-lora`. The repo is created if it does not exist. |
| `BASE_MODEL`      | `Qwen/Qwen3.6-27B`               | Base model id. The pipeline expects `model_type: qwen3_5` (Qwen3.5 / Qwen3.6 family) — step 2.5 of `run.sh` aborts if you point at a different family without adjusting `target_modules` in `train_sft.py`. |
| `DATASET`         | `j14i/cl-macros-thinking`        | HF dataset id. The default is the chat-format corpus with `<think>` reasoning traces (system + user + assistant turns) — uploaded via `scripts/upload_dataset_to_hf.py`. Override to `j14i/cl-macros` to train on the raw instruction/output triples (the trainer auto-converts to messages). |
| `MAX_SEQ_LENGTH`  | `4096`                           | Tokens per training sample. Drop to `2048` if you OOM — the trace-heavy assistant turns can push individual sequences long. |
| `OUTPUT_DIR`      | `./output`                       | Training output directory. Adapter lands in `$OUTPUT_DIR/final_adapter`. |
| `WANDB_API_KEY`   | *(unset — W&B disabled)*         | Weights & Biases API key for live training curves. |
| `SMOKE_PROMPT`    | a `when-let` macro prompt        | Prompt used by the post-training generation smoke-test. |

## Quick start

```bash
export HF_TOKEN=hf_xxx                       # required
export HF_REPO=j14i/cl-macro-27b-lora        # optional, enables upload
export WANDB_API_KEY=xxx                     # optional, enables W&B
bash scripts/cloud/run.sh
```

## Notes

- **Unsloth-managed stack.** `Qwen/Qwen3.6-27B` (`model_type: qwen3_5`) is a
  hybrid Gated DeltaNet + full-attention model that the standard
  `AutoModelForCausalLM` path can't load (the architecture class is
  `Qwen3_5ForConditionalGeneration`, multimodal). `run.sh` installs Unsloth,
  which provides the custom Mamba/DeltaNet kernels and pulls `transformers v5`
  automatically. **Do not pin `transformers` / `peft` / `trl` in
  `requirements.txt`** — let Unsloth resolve the versions.
- **No QLoRA / 4-bit.** Unsloth explicitly recommends against 4-bit training
  for Qwen3.5/3.6 models; `train_sft.py` uses `load_in_16bit=True` (native bf16).
- **VRAM budget.** Unsloth cites **27B bf16 LoRA: ~56 GB**. The defaults
  (`batch_size=1`, `grad_accum=8`, `max_seq_length=4096`, `optim="adamw_8bit"`,
  Unsloth gradient checkpointing) fit on A100 80 GB with headroom. If you OOM,
  drop `MAX_SEQ_LENGTH=2048` first; the effective batch of 8 already comes from
  accumulation, so don't lower batch further.
- **Step 2.5 compat check is cheap.** It downloads only the config JSON
  (~5 KB), so a `qwen3_5`-mismatch failure costs ~$0.01, not a 54 GB download.
- **Reasoning traces.** Generated locally by
  `scripts/generate_thinking_traces_full.py` (Qwen3.6-27B via MLX, ~hours on a
  Mac). To avoid re-running that step on the A100, push the enriched files to
  HuggingFace once with `scripts/upload_dataset_to_hf.py` and point
  `DATASET=j14i/cl-macros-thinking` at it on the pod (the default).
