# Troubleshooting — RunPod training pipeline

Recurring failure modes hit during real runs of `scripts/cloud/run.sh` /
`scripts/cloud/launch.sh`, with the actual fixes.

## 1. Slow training: GPU at ~11%, CPU at 100%, first step ~190 s

**Symptom.** Pod telemetry shows the A100 idling at ~11% utilization and
power draw ~60 W (of 300 W limit) while CPU pegs 100%. `nvidia-smi` is quiet.
First training step takes ~190 s; projected ETA is ~36 hours for 687 steps.
Unsloth's startup log includes:

```
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation. To install follow
https://github.com/fla-org/flash-linear-attention#installation and
https://github.com/Dao-AILab/causal-conv1d
```

**Why.** Qwen3.6's hybrid Gated DeltaNet (≈¾ of all layers) only hits the
fast path when **both** `flash-linear-attention` and `causal-conv1d` import
successfully. Without them, the DeltaNet ops run on a CPU-bound PyTorch
reference implementation. `pip install causal-conv1d` fails on the
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image because:

- No prebuilt wheel exists for `torch 2.10.0+cu128` (the version Unsloth's
  `--force-reinstall` pulls).
- The image ships CUDA 12.4 nvcc at `/usr/local/cuda-12.4/bin/nvcc` but
  doesn't put it on `PATH`, so the source build can't find a compiler.
- `CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE` doesn't help — the package
  unconditionally `import causal_conv1d_cuda` at module load.

**Fix.** Build causal-conv1d from source with explicit CUDA env:

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST="8.0"      # A100 = Ampere = 8.0; H100 = 9.0
export MAX_JOBS=4
pip install flash-linear-attention
pip install --no-build-isolation --no-cache-dir causal-conv1d
python -c "from causal_conv1d import causal_conv1d_fn; print('OK')"
```

The minor-version mismatch (CUDA 12.4 nvcc against torch's CUDA 12.8 libs) is
tolerated. Wheel builds in ~5-10 min. After that, the "fast path not
available" warning disappears and per-step time drops from ~190 s to the
expected ~5-30 s.

`scripts/cloud/run.sh` step 1 attempts both installs as a best-effort. If
you change RunPod images, verify that block's `CUDA_HOME` / `PATH` still
match the image's actual nvcc location.

**Alternative.** Use the official Unsloth template `pzr9tt3vvq`
(`unsloth/unsloth:latest`) which ships both kernels prebuilt. Caveat: the
13 GB image takes ~30-50 min to extract on cold Community Cloud hosts
(~$0.50-1 burn). Secure Cloud is faster but ~2× the hourly rate.

---

## 2. `AttributeError: 'str' object has no attribute 'get'` from `tokenizer.apply_chat_template`

**Symptom.** Trainer init fails at `_prepare_dataset` with this attribute
error pointing to `apply_chat_template` iterating message bodies.

**Why.** Unsloth's `SFTTrainer` calls `formatting_func` with two shapes:

1. First, `formatting_func(next(iter(dataset)))` — a single example dict
   where `messages` is one conversation (list of role/content dicts).
2. Later, via `.map(batched=True)` — a batched dict where `messages` is a
   list of conversations.

A `formatting_func` that only handles one shape crashes on the other.

**Fix** (in `train_sft.py`, already applied):

```python
def formatting_func(examples: dict) -> list:
    msgs = examples["messages"]
    if msgs and isinstance(msgs[0], dict):
        # single example: msgs is one conversation
        return [tokenizer.apply_chat_template(msgs, tokenize=False)]
    # batched: msgs is a list of conversations
    return [tokenizer.apply_chat_template(m, tokenize=False) for m in msgs]
```

---

## 3. `Unsloth: You must specify a formatting_func`

**Symptom.** Trainer init fails immediately with this RuntimeError.

**Why.** Unsloth's wrapper requires an explicit `formatting_func=` for
messages-format datasets; the autodetect path that works with vanilla
`trl.SFTTrainer` doesn't apply.

**Fix.** Pass `formatting_func=` to `SFTTrainer(...)` — see issue 2.

---

## 4. `wandb.errors.errors.CommError: entity not specified` / `entity X not found during upsertBucket`

**Symptom.** Training reaches `trainer.train()` and crashes in the wandb
`on_train_begin` callback. Two flavors:

- `entity not specified, and viewer has no default entity` — `WANDB_ENTITY`
  isn't set and the wandb account has no default.
- `entity X not found during upsertBucket` — `WANDB_ENTITY=X` is set but X
  isn't a valid entity for this account (e.g. setting it to your **username**
  when the actual entity is an **org** name).

**How to discover your real entity.** Run locally:

```bash
WANDB_API_KEY=... python -c "
import wandb; wandb.login(); v = wandb.Api().viewer
print('username:', v.username)
print('entity:  ', v.entity)
"
```

The right value for `WANDB_ENTITY` is the **entity** field, which can differ
from the username (e.g. `j14i-justme-org`, not `j14i`).

**Fix.** Either:

- Put `WANDB_ENTITY=<your-entity>` in `.env`. `launch.sh` forwards it.
- Or remove it entirely so `_maybe_init_wandb()` in `train_sft.py`
  auto-discovers via `wandb.Api().viewer.entity`.
- Or set `WANDB_DISABLED=true` to skip wandb entirely; HF Trainer respects
  it and adapter still saves + uploads.

---

## 5. `ValueError: invalid literal for int() with base 10: ''`

**Symptom.** `train_sft.py` crashes at module load on `int(_env("MAX_SEQ_LENGTH", "4096"))`.

**Why.** `launch.sh` previously forwarded `KEY=''` for every optional var
that wasn't set locally — `os.environ.get(name, default)` then returned the
empty string instead of taking the default.

**Fix.** `train_sft.py`'s `_env()` helper treats empty == unset:

```python
def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default
```

`launch.sh` also skips empty exports in the heredoc loop.

---

## 6. SSH heredoc env vars not seen on the pod (`HF_TOKEN is not set`)

**Symptom.** `run.sh` step 2 dies with `HF_TOKEN is not set` even though
`.env` defines it and `launch.sh` ran cleanly.

**Why.** `runpodctl pod create --env '<JSON>'` sets env vars at the
container PID 1, but RunPod's default sshd on `runpod/pytorch:*` images
does **not** expose them to SSH login shells.

**Fix.** `launch.sh` writes each env var as an explicit `export` line into
the bash heredoc that runs the pipeline. The heredoc is unquoted so local
`$VARS` interpolate into single-quoted assignments on the pod side. Empty
values are skipped.

---

## 7. Launcher killed before EXIT trap fires → orphaned pod still billing

**Symptom.** The launcher process gets SIGHUP'd (e.g., parent shell exits)
and the `trap cleanup EXIT` doesn't run, so the pod stays alive draining
credits.

**Why / how to avoid.**

- Run `launch.sh` foreground in a Bash tool with `run_in_background=true`
  instead of wrapping with `& wait`. The harness manages a real detached
  process; in-script `&` produces grandchildren that lose the trap.
- Pre-emptive defence: set `KEEP_POD=1` so failures keep the pod alive on
  purpose — then SSH in and debug live without spending another $0.30 on a
  fresh boot.
- If a pod really does get orphaned: `runpodctl pod list` to find it,
  `runpodctl pod delete <id>` to kill it.

---

## 8. Detached pipeline can't find a script that was committed mid-run

**Symptom.** The pod's `full_pipeline.sh` runs train → upload → postprocess
sequentially. Training succeeds, upload succeeds, then:

```
python: can't open file '/workspace/cl-macro-llm/scripts/cloud/postprocess.py':
        [Errno 2] No such file or directory
postprocess.py rc=2
```

Watcher marks `POSTPROCESS_FAILED_BUT_ADAPTER_OK` and deletes the pod. The
adapter is on HF (good), but the merged / GGUF artifacts were never built
because their script wasn't on the pod yet.

**Why.** The detached pipeline did a single `git pull --ff-only` at the
moment it was launched on the pod. Any commits pushed to `origin/main` **after**
that initial pull (postprocess.py, to_mlx.sh, etc.) never landed on the pod —
the pipeline script just doesn't re-pull between phases. So a long-running
detached pipeline is effectively running a snapshot of the repo from when it
was first launched, even if you push fixes in parallel.

**Fix (workflow).** Don't commit code changes that the in-flight pipeline
will reach later. Push everything you need (training, upload, postprocess,
all helper scripts) **before** kicking off `setsid nohup full_pipeline.sh`.
Treat the launch as a release tag for that run.

**Fix (script).** When generating `full_pipeline.sh` on the pod, either:

```bash
# Option A: pull before each phase
cd /workspace/cl-macro-llm
git pull --ff-only || true
python scripts/cloud/upload_to_hf.py
git pull --ff-only || true
python scripts/cloud/postprocess.py
```

```bash
# Option B (safer): snapshot scripts once at pipeline start
cp -r /workspace/cl-macro-llm/scripts /workspace/scripts-snapshot
# ... then run from /workspace/scripts-snapshot/cloud/* throughout
```

Option B is more robust because mid-pipeline `git pull` could pull in a
breaking change too. Snapshot once at launch → run that snapshot to
completion → mid-run pushes only affect the NEXT pipeline launch.

**Recovery if it already happened.** The adapter is safe on HF; the merged
and GGUF variants are not. Two paths:

1. **Skip them.** Adapter is what most downstream uses need (PEFT load on
   top of base). MLX conversion can be done on a Mac via `mlx_lm.convert`
   pointing directly at the base model + adapter, no merged repo needed.
2. **Re-run postprocess on a fresh pod.** Cheaper to use a smaller GPU (the
   merge + push doesn't need an A100 — even an L4 would do). Just:
   `python scripts/cloud/postprocess.py` with the same env vars.

---

## 9. `ValueError: Model type qwen3_5_text not supported` from mlx-lm

**Symptom.** After merging the Qwen3.6-27B base + LoRA adapter on a Mac
with `PeftModel.merge_and_unload()` and trying to convert to MLX:

```
ValueError: Model type qwen3_5_text not supported.
```

**Why.** `Qwen/Qwen3.6-27B` is registered as the multimodal class
`Qwen3_5ForConditionalGeneration` with `model_type: qwen3_5`. When PEFT
merges a text-only LoRA into this and saves, transformers v5 strips down
to the text-only sub-config — model_type becomes `qwen3_5_text` and
architectures becomes `Qwen3_5ForCausalLM`. mlx-lm's model registry
only knows the umbrella `qwen3_5`, so the lookup fails.

The actual model weights and layer structure are unchanged — it's a
naming/routing issue, not an architecture incompatibility.

**Fix.** Patch the merged `config.json` to use `qwen3_5` before invoking
`mlx_lm.convert`:

```bash
sed -i.bak 's/"model_type": "qwen3_5_text"/"model_type": "qwen3_5"/' \
    "$MERGED_DIR/config.json"
```

`scripts/mac_to_mlx.sh` does this automatically as step 2.5 (with a `.bak`
backup so you can revert). Verified working: post-patch
`mlx_lm.convert --quantize --q-bits 4` produces a 14 GB model that
generates at ~33 tok/s on Apple Silicon, no quality loss vs unpatched
inference.

**Long-term fix.** mlx-lm needs a `qwen3_5_text` entry in its model
registry pointing at the same Qwen3.5 handler. File an issue / PR
upstream once stabilised.
