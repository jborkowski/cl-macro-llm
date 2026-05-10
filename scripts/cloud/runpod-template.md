# RunPod training template

Fine-tune **Qwen3.6-27B** with **rank-32 LoRA in native bf16** on the Common
Lisp macro corpus, targeting a single A100 80GB. Driven by Unsloth (the
official `transformers` path can't load Qwen3.6 — its hybrid Gated DeltaNet +
full-attention architecture needs Unsloth's custom Mamba/Triton kernels).

The pipeline is fully driven by `scripts/cloud/run.sh`: install deps → verify
HuggingFace access → verify Unsloth + Qwen3.6 compatibility → train →
smoke-test generation → push the LoRA adapter back to HuggingFace → print a
summary. The training data is downloaded directly from
`j14i/cl-macros-thinking` (chat-format with `<think>` reasoning traces) —
no scp of local JSONL needed.

## Fast path: scripted launch

If you have `runpodctl` and `jq` installed and a populated `.env` (see
`.env.example`):

```bash
bash scripts/cloud/launch.sh
```

This boots an A100 80GB pod, verifies you're authenticated as
`RUNPOD_EXPECTED_EMAIL` (so you can't ship work to a company account by
accident), clones the repo on the pod, runs `bash scripts/cloud/run.sh`,
and tears the pod down on exit (`KEEP_POD=1` to keep it alive). Refuses to
boot if `HF_REPO` is unset — without it the adapter dies with the pod.

The rest of this doc covers the manual / dashboard flow.

## 1. Provision the pod

- Cloud: RunPod → Deploy → **A100 80GB** (Secure Cloud or Community Cloud).
- Template: **RunPod PyTorch 2.x** (any image with CUDA 12.x and Python 3.10+).
  CUDA 12.x — **do not pick 13.2 images** (Unsloth flags 13.2 as gibberish-
  output-inducing for Qwen3.x).
- Volume: 100GB+ workspace disk (the 27B base weights + checkpoints are large).
- Expose: SSH + Jupyter (or Web Terminal — both work).

## 2. Connect

Either:
- SSH: `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519`, or
- RunPod dashboard → **Connect** → **Web Terminal**.

## 3. Get the project onto the pod

```bash
git clone https://github.com/jborkowski/cl-macro-llm.git
cd cl-macro-llm
```

(If you've added the pod's SSH key to GitHub: `git clone git@github.com:jborkowski/cl-macro-llm.git`.)

## 4. Set environment variables

`scripts/cloud/ENV.md` is the source of truth. The minimum:

```bash
export HF_TOKEN=hf_xxx                       # required (upload + rate limits)
export HF_REPO=j14i/cl-macro-27b-lora        # optional — auto-push the adapter to HF
export WANDB_API_KEY=xxx                     # optional — live training curves
# export MAX_SEQ_LENGTH=2048                 # optional — drop to 2048 if you OOM
```

`Qwen/Qwen3.6-27B` is currently public (Apache 2.0, not gated). `HF_TOKEN` is
still required because the pipeline uses it to push the adapter and to keep
the auth path consistent across API calls.

## 5. Train

```bash
bash scripts/cloud/run.sh
```

This does, in order:

1. `pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo`
   then `pip install -r scripts/cloud/requirements.txt -q`. Unsloth pulls
   `transformers v5`, `peft`, `trl`, `torch`, and `accelerate` itself —
   they're intentionally not pinned in `requirements.txt`.
2. Verify HF access to the base model and the dataset.
2.5. Verify Unsloth recognises Qwen3.6's `model_type: qwen3_5` (downloads
     only the ~5 KB config — costs ~$0.01 if it fails).
3. `python scripts/cloud/train_sft.py` — 3 epochs, `batch_size=1`,
   `grad_accum=8`, `max_seq_length=4096`, bf16, `adamw_8bit`, Unsloth
   gradient checkpointing.
4. Smoke-test generation via `scripts/cloud/generate.py`.
5. Push the adapter to `$HF_REPO` if set (no-op if unset). The upload script
   re-queries the repo after push to verify `adapter_config.json` and
   `adapter_model.safetensors` actually landed.

Checkpoints land in `./output/checkpoint-*` and the final LoRA adapter in
`./output/final_adapter/`.

**VRAM budget** (Unsloth's own number for 27B bf16 LoRA: ~56 GB). With our
defaults you'll see ~60–63 GB peak — comfortable on 80 GB. If you OOM, set
`MAX_SEQ_LENGTH=2048` before re-running.

**Wall clock**: ~2–3.5 h end-to-end, including base-model download (~54 GB)
and adapter upload. Cost on A100 80GB: roughly **$4–9** at typical RunPod
rates.

## 6. Retrieve the adapter

If `HF_REPO` was set, it's already on HuggingFace — pull it from anywhere with
Unsloth:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="<HF_REPO>",      # adapter dir; resolves base via adapter_config.json
    max_seq_length=4096,
    load_in_16bit=True,
)
```

Otherwise, scp it off the pod:

```bash
# from your laptop
scp -P <port> -r \
    root@<pod-ip>:/workspace/cl-macro-llm/output/final_adapter ./
```

The adapter is small (a few hundred MB); the base weights stay on the pod.

## 7. Tear down

Stop or terminate the pod from the RunPod dashboard once the adapter is on
HuggingFace (or scp'd locally) — A100 80GB is billed per second.

## Optional: override the dataset

To train on the raw `j14i/cl-macros` (instruction/output, no reasoning
traces) instead of the default trace-enriched corpus:

```bash
export DATASET=j14i/cl-macros
bash scripts/cloud/run.sh
```

The trainer auto-converts instruction/output rows to chat-format messages.

## Optional: override the base model

`run.sh` step 2.5 enforces `model_type: qwen3_5` (Qwen3.5/3.6 family). To
train a different family (e.g. Qwen3, Llama, Mistral), unset that check and
adjust `target_modules` in `train_sft.py` to match the architecture's linear
layer names. Unsloth still handles most modern LLMs via the same
`FastLanguageModel` API.
