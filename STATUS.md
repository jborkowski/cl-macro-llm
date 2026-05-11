# Project Status — cl-macro-llm

Fine-tune a large language model to generate Common Lisp macros with
explicit `<think>...</think>` reasoning traces. Long-term plan: SFT (this
phase) → GRPO with executable-CL reward signal.

## Phase 1 — Supervised Fine-Tuning: **COMPLETE** ✅

### Artifacts produced

| Artifact | Where | Size | Status |
|---|---|---|---|
| LoRA adapter | [`j14i/cl-macro-27b-lora`](https://huggingface.co/j14i/cl-macro-27b-lora) | ~600 MB | On HuggingFace |
| Merged bf16 model | `~/models/cl-macro-27b-lora-merged` | ~54 GB | Local (Mac) |
| MLX 4-bit (Apple Silicon) | `~/models/cl-macro-27b-lora-mlx-q4` | ~14 GB | Local, **runs at 33 tok/s** |
| GGUF (llama.cpp/Ollama) | — | — | Not produced (postprocess script wasn't on the pod yet; recoverable) |
| Training data | [`j14i/cl-macros-thinking`](https://huggingface.co/datasets/j14i/cl-macros-thinking) | 1828 / 203 (train/val) | On HuggingFace |

### Training run

- **Base model:** `Qwen/Qwen3.6-27B` (hybrid Gated DeltaNet + full-attention, `model_type: qwen3_5`)
- **Hardware:** Single A100 80GB (RunPod Community Cloud)
- **Framework:** Unsloth 2026.5.2 + transformers v5 + trl 0.24
- **Method:** LoRA rank 32, alpha 64, dropout 0.05, target_modules q/k/v/o/gate/up/down
- **Schedule:** 3 epochs, batch 1, grad-accum 8 (effective 8), max_seq_length 4096, cosine LR 2e-5
- **Wall clock:** 2h 07min training + ~10 min for setup/eval; ~17 hours total project time including diagnostics
- **Cost:** ~$3 of A100 time

### Loss trajectory

| step | train | eval | gap |
|---:|---:|---:|---:|
| 50 | 0.551 | 0.556 | 1% |
| 100 | 0.464 | 0.476 | 3% |
| 200 | 0.400 | 0.439 | 10% |
| 300 | 0.355 | 0.407 | 15% |
| 400 | 0.313 | 0.383 | 18% |
| 500 | 0.291 | 0.372 | 22% |
| 687 (final) | 0.389 (avg) | **0.367** | — |

Eval loss decreased monotonically by 34%, no inflection up. Mild controlled
overfitting (gap stabilized around 15-18%) — `lora_dropout=0.05` did its job.
`load_best_model_at_end=True` kept the lowest-eval-loss checkpoint.

### Validated end-to-end

Ran `with-stopwatch` macro through the MLX 4-bit model on a Mac, then
verified the output in **SBCL 2.6.4**:

```lisp
(defmacro with-stopwatch (&body body)
  (let ((start-sym (gensym))
        (end-sym (gensym))
        (result-sym (gensym)))
    `(let ((,start-sym (get-internal-real-time)))
       (let ((,result-sym (progn ,@body)))
         (let ((,end-sym (get-internal-real-time)))
           (values (float (/ (- ,end-sym ,start-sym) internal-time-units-per-second))
                   ,result-sym))))))
```

**Result:** `(with-stopwatch (sleep 0.5) (+ 1 2))` → `0.505015 s, 3` ✓

What the model got right:
- Backquote/comma/comma-at syntax, `&body` handling
- Hygiene: three independent gensyms, no variable capture
- Idiomatic structure: `(values elapsed result)`, `float` coercion
- Correct constant: `internal-time-units-per-second`

What the model got wrong:
- Hallucinated function name `internal-real-time` (correct CL name is
  `get-internal-real-time`). Manual one-line fix, otherwise the macro was
  structurally correct.

## What this means for downstream use

- **Suitable for:** drafting macros that a human (or compiler) reviews.
  Format adherence is excellent; reasoning traces are coherent; mechanics
  are reliable.
- **Not suitable for:** auto-execution. The function-name hallucination is
  a class of error SFT alone cannot fix — the model has no signal
  distinguishing real CL functions from plausible-sounding ones.
- This is exactly the signal **Phase 2 (GRPO) is designed to fix**: a
  reward function that runs `sbcl --script` on the generated macro and
  rewards compilation success will eliminate function-name hallucinations
  much faster than scaling SFT data.

## Repository structure

```
scripts/
├── cloud/                    # RunPod pipeline (training-side)
│   ├── launch.sh             #   end-to-end launcher (account check + pod boot + ssh)
│   ├── run.sh                #   on-pod 5-step pipeline (install/HF-verify/compat/train/upload)
│   ├── train_sft.py          #   Unsloth FastLanguageModel + LoRA + SFTTrainer
│   ├── upload_to_hf.py       #   push adapter with post-upload verification
│   ├── postprocess.py        #   merge + push GGUF/merged variants (NOT executed yet — see §8)
│   ├── generate.py           #   inference smoke-test (Unsloth load)
│   ├── ENV.md                #   required/optional env vars
│   ├── TROUBLESHOOTING.md    #   9 documented failure modes + fixes
│   └── runpod-template.md    #   pod provisioning doc (manual + scripted)
├── mac_to_mlx.sh             # Mac-side: merge adapter -> MLX 4-bit, idempotent
├── to_mlx.sh                 # alt path that assumes -merged HF repo exists
├── upload_dataset_to_hf.py   # one-time: push local trace data to HF
├── prepare_data_full.py      # raw cl-macros -> chat-format JSONL
└── generate_thinking_traces_full.py  # Qwen3.6 MLX 4-bit fills <think> placeholders
.env.example                  # documented env var template
```

## Phase 2 — GRPO: **PLANNED**

What we now know from Phase 1 that informs the GRPO design:

1. **Reward signal must include SBCL compilation.** The function-name
   hallucination class is invisible to next-token-prediction loss. A
   pass/fail signal from `compile-file` (or `sb-ext:eval-when` + read +
   eval) catches it directly.
2. **`load_best_model_at_end` + eval at every save aligns naturally with
   PPO-style rollouts.** We already have the eval set wired.
3. **Network volume matters now.** A single GRPO iteration is 4-8 h vs
   2-3 h for SFT, and we iterate more. 150-200 GB RunPod volume in a
   region with reliable A100 supply.
4. **Reward harness must be sandboxed.** A generated macro can run
   arbitrary code in SBCL — use `:safe` package + timeout + process
   isolation. Don't `(load ...)` user input straight into the trainer's
   Lisp image.
5. **Keep `lora_dropout=0.05`.** GRPO benefits from exploration; some
   regularization keeps the policy from collapsing.

Reward signal ranking (strongest first):
1. **`macroexpand` correctness** — `(macroexpand '(your-macro ...))` returns
   sensible form without error. Cheapest sanity gate.
2. **Test pass rate** — if the prompt has expected expansion or behavior,
   compare against it.
3. **Linter / style** — symbol hygiene (gensym used where capture risk
   exists), naming conventions, `&body` vs `&rest` choice.
4. **LLM-as-judge** — fallback for open-ended prompts where the above
   don't apply.

## Recent commits (most recent first)

```
fa1f5c4 TROUBLESHOOTING §9: mlx-lm rejects qwen3_5_text after PEFT merge
1def9df mac_to_mlx.sh: auto-patch qwen3_5_text -> qwen3_5 in merged config
e2e4574 mac_to_mlx.sh: use uv run instead of bare python
9bab299 Add mac_to_mlx.sh — local Mac path from HF adapter to MLX 4-bit
2132eae TROUBLESHOOTING: pipeline snapshot vs mid-run git pull
74b886d Add post-training pipeline: merged HF + GGUF + MLX 4-bit
48fafa7 docs: add TROUBLESHOOTING.md + fix FLA/causal-conv1d install on pytorch image
71ecc01 run.sh: best-effort install of fast Qwen3.6 DeltaNet kernels
07ccd2f train_sft.py: wandb entity check + warmup_steps migration
7dcfce5 train_sft.py: formatting_func handles single-example and batch shapes
1e07548 Switch to Unsloth for Qwen3.6 fine-tuning
c0a5534 Switch RunPod training to A100 80GB full bf16 + j14i/cl-macros
```

Notable diagnostic episodes (now documented in `scripts/cloud/TROUBLESHOOTING.md`):
- Qwen3.6 needs Unsloth (vanilla transformers can't load multimodal model_type)
- FLA + causal-conv1d source build required for fast DeltaNet on pytorch image
- WandB needs explicit entity (username ≠ entity for some accounts)
- Detached pipeline snapshots scripts at launch — mid-run pushes don't land
- mlx-lm rejects `qwen3_5_text` model_type after PEFT merge — patch config.json
