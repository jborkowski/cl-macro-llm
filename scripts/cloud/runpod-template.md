# RunPod training template

Fine-tune Qwen3.6-27B with LoRA (rank 32, NF4 4-bit base) on the Common Lisp
macro corpus. Targets a single A100 80GB.

## 1. Provision the pod

- Cloud: RunPod → Deploy → **A100 80GB** (Secure Cloud or Community Cloud).
- Template: **RunPod PyTorch 2.x** (any image with CUDA 12.x and Python 3.10+).
- Volume: 100GB+ workspace disk (the 27B base weights + checkpoints are large).
- Expose: SSH + Jupyter (or Web Terminal — both work).

## 2. Connect

Either:
- SSH: `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519`, or
- RunPod dashboard → **Connect** → **Web Terminal**.

## 3. Get the project onto the pod

Pick one:

```bash
# (a) clone from GitHub (HTTPS — no key needed on the pod)
git clone https://github.com/jborkowski/cl-macro-llm.git
cd cl-macro-llm

# (b) or, if you've added the pod's SSH key to GitHub:
#   git clone git@github.com:jborkowski/cl-macro-llm.git
#   cd cl-macro-llm

# (c) or scp the working tree from your laptop:
#   scp -P <port> -r cl-macro-llm root@<pod-ip>:/workspace/
```

Note: the JSONL training data lives under `data/processed/full/` and is
expected to be present in the repo. If it's gitignored, scp it in
separately:

```bash
scp -P <port> data/processed/full/*.jsonl \
    root@<pod-ip>:/workspace/cl-macro-llm/data/processed/full/
```

Make sure these files are present on the pod:
- `data/processed/full/train.jsonl`
- `data/processed/full/valid.jsonl`
- `scripts/cloud/`

Optional — log in to Weights & Biases for live curves:

```bash
export WANDB_API_KEY=...   # before run.sh, or `wandb login`
```

## 4. Train

```bash
bash scripts/cloud/run.sh
```

This installs deps and launches `train_sft.py` via `accelerate`. Expect ~3
epochs over the full dataset; checkpoints land in `./output/checkpoint-*` and
the final LoRA adapter in `./output/final_adapter/`.

## 5. Pull the adapter back

```bash
# from your laptop
scp -P <port> -r \
    root@<pod-ip>:/workspace/cl-macro-llm/output/final_adapter ./
```

The adapter is small (a few hundred MB); the 4-bit base stays on the pod.

## 6. Smoke-test on the pod (optional)

```bash
python scripts/cloud/generate.py --prompt "Write a Common Lisp macro that..."
```

## 7. Tear down

Stop or terminate the pod from the RunPod dashboard once the adapter is
downloaded — A100 80GB is billed per second.
