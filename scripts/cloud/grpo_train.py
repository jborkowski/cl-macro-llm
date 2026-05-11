#!/usr/bin/env python3
"""GRPO training on cl-ds + creative-macros katas with macro-gym SBCL reward.

Phase 2 trainer. Starts from the Phase 1 SFT adapter (j14i/cl-macro-27b-lora)
and continues policy updates against the executable reward produced by
running each generated defmacro through macro-gym's SBCL harness.

Required env vars (typically forwarded from .env via launch.sh):
    HF_TOKEN           HuggingFace token (read base model, push final adapter)
    HF_REPO            target HF repo for the final adapter (e.g. j14i/cl-macro-27b-grpo)
    KATA_ROOT          dir containing kata subdirs from cl_ds_to_katas.py + creative authoring

Optional:
    SFT_ADAPTER        default j14i/cl-macro-27b-lora — Phase 1 LoRA, policy init
    BASE_MODEL         default Qwen/Qwen3.6-27B — only used if SFT_ADAPTER doesn't
                       carry the base reference in its config
    OUTPUT_DIR         default ./grpo-output
    MAX_SEQ_LENGTH     default 4096
    MAX_COMPLETION_LEN default 2048 — completions longer than this get truncated
    NUM_GENERATIONS    default 8 — rollouts per prompt for advantage estimation
    MAX_STEPS          default 500
    LEARNING_RATE      default 5e-7
    BETA               default 0.05 — KL penalty toward frozen reference (SFT)
    TEMPERATURE        default 0.9 — rollout sampling temperature
    WANDB_ENTITY       e.g. j14i-justme-org
    WANDB_PROJECT      default cl-macro-llm-grpo
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

# Unsloth long-context GRPO recommendation: keep vLLM weights swapped out
# while the trainer step runs, then page them back for rollouts. Must be
# set before unsloth is imported.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


# ─── env helpers ──────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


SFT_ADAPTER         = _env("SFT_ADAPTER", "j14i/cl-macro-27b-lora")
HF_REPO             = _env("HF_REPO")           # required for final upload
KATA_ROOT           = Path(_env("KATA_ROOT", "/workspace/katas"))
OUTPUT_DIR          = _env("OUTPUT_DIR", "./grpo-output")
MAX_SEQ_LENGTH      = int(_env("MAX_SEQ_LENGTH", "4096"))
MAX_COMPLETION_LEN  = int(_env("MAX_COMPLETION_LEN", "2048"))
NUM_GENERATIONS     = int(_env("NUM_GENERATIONS", "8"))
MAX_STEPS           = int(_env("MAX_STEPS", "500"))
LEARNING_RATE       = float(_env("LEARNING_RATE", "5e-7"))
BETA                = float(_env("BETA", "0.05"))
TEMPERATURE         = float(_env("TEMPERATURE", "0.9"))

# Unsloth long-context GRPO knobs (see
# https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/grpo-long-context
# and the RL guide root). loss_type ∈ {grpo, gspo, dr_grpo}; the asymmetric
# epsilon_high / delta trust region is the Unsloth-recommended default for
# stable RL on long completions.
LOSS_TYPE                       = _env("LOSS_TYPE", "grpo")
EPSILON                         = float(_env("EPSILON", "0.2"))
EPSILON_HIGH                    = float(_env("EPSILON_HIGH", "0.28"))
DELTA                           = float(_env("DELTA", "1.5"))
MASK_TRUNCATED_COMPLETIONS      = _env("MASK_TRUNCATED_COMPLETIONS", "1") != "0"
# Both default to "auto-tune by VRAM" if env is empty.
UNSLOTH_GRPO_MINI_BATCH         = _env("UNSLOTH_GRPO_MINI_BATCH", "")
UNSLOTH_LOGIT_CHUNK_MULTIPLIER  = _env("UNSLOTH_LOGIT_CHUNK_MULTIPLIER", "")

# Baseline-only: run the SFT policy over the eval split, compute mean
# reward, print baseline_mean=<float> and exit before constructing the
# trainer. Gate C of run_grpo.sh depends on this.
BASELINE_ONLY       = _env("BASELINE_ONLY", "") not in ("", "0", "false", "False")
EVAL_TEST_SIZE      = float(_env("EVAL_TEST_SIZE", "0.1"))


SYSTEM_PROMPT = (
    "You are an expert Common Lisp macro programmer. Think step by step "
    "before writing the macro. Always explain your reasoning in <think>...</think> "
    "tags, then provide the defmacro form."
)


# ─── wandb (non-fatal, mirrors train_sft.py pattern) ──────────────────

def _maybe_init_wandb() -> bool:
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="never", relogin=True)
    except Exception as e:
        print(f"wandb login failed ({type(e).__name__}: {e}); training without W&B.")
        return False
    if not os.environ.get("WANDB_ENTITY"):
        try:
            viewer = wandb.Api().viewer
            entity = getattr(viewer, "entity", None) or getattr(viewer, "username", None)
        except Exception:
            entity = None
        if not entity:
            print("wandb: no default entity; set WANDB_ENTITY in env. Skipping.")
            return False
        os.environ.setdefault("WANDB_ENTITY", entity)
    os.environ.setdefault("WANDB_PROJECT", "cl-macro-llm-grpo")
    return True


# ─── kata loading ─────────────────────────────────────────────────────

def load_katas(kata_root: Path) -> Dataset:
    """Walk kata directory tree (cl-ds/* + creative/*), emit prompt+meta rows."""
    rows = []
    candidates = []
    # Two-level walk: top-level kata dirs OR nested under cl-ds/, creative/
    for p in kata_root.iterdir():
        if not p.is_dir() or p.name.startswith("_"):
            continue
        if (p / "metadata.json").exists():
            candidates.append(p)
        else:
            for sub in p.iterdir():
                if sub.is_dir() and (sub / "metadata.json").exists():
                    candidates.append(sub)

    for kata_dir in sorted(candidates):
        meta = json.loads((kata_dir / "metadata.json").read_text())
        instruction = (meta.get("instruction") or "").strip()
        if not instruction:
            continue
        rows.append({
            "kata_id":       kata_dir.name,
            "kata_path":     str(kata_dir.resolve()),
            "prompt":        [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": instruction},
            ],
            "category":      meta.get("category") or "?",
            "complexity":    meta.get("complexity") or "?",
            "quality_score": float(meta.get("quality_score") or 1.0),
        })
    return Dataset.from_list(rows)


# ─── reward function ──────────────────────────────────────────────────

_DEFMACRO_HEAD_RE = re.compile(r"\(defmacro\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_defmacro(text: str) -> str | None:
    """Strip <think>...</think> and pull the first balanced (defmacro …)."""
    cleaned = _THINK_RE.sub("", text)
    m = _DEFMACRO_HEAD_RE.search(cleaned)
    if not m:
        return None
    start = m.start()
    depth = 0
    in_string = False
    in_escape = False
    i = start
    while i < len(cleaned):
        c = cleaned[i]
        if in_escape:
            in_escape = False
        elif c == "\\":
            in_escape = True
        elif in_string:
            if c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return cleaned[start:i + 1]
        i += 1
    return None  # truncated, never closed


# Lazy env cache. macro-gym persists an SBCL subprocess per env; we keep
# the most recently used ones alive but don't preload all 753+.
_env_cache: dict[str, "MacroEnv"] = {}
_ENV_CACHE_MAX = 64


def _get_env(kata_path: str):
    global _env_cache
    if kata_path in _env_cache:
        return _env_cache[kata_path]
    from macro_gym import MacroEnv  # imported lazily so script imports without it
    if len(_env_cache) >= _ENV_CACHE_MAX:
        # FIFO eviction (good enough — kata sampling is random anyway)
        oldest = next(iter(_env_cache))
        try:
            _env_cache[oldest].close()
        except Exception:
            pass
        del _env_cache[oldest]
    _env_cache[kata_path] = MacroEnv(kata_dir=kata_path)
    return _env_cache[kata_path]


def _completion_to_text(completion) -> str:
    """trl can hand us a string or a list-of-messages depending on prompt format."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # last assistant turn
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return completion[-1].get("content", "") if completion else ""
    return str(completion)


def macro_gym_reward(prompts, completions, **kwargs) -> list[float]:
    """GRPO reward callable. Per-completion:
        1. extract (defmacro …) from text
        2. submit to macro-gym MacroEnv.step
        3. clamp to [-0.5, 1.5] in case the env returns out-of-band values
    """
    kata_paths = kwargs.get("kata_path") or []
    if len(kata_paths) != len(completions):
        raise RuntimeError(
            f"reward fn: got {len(completions)} completions but "
            f"{len(kata_paths)} kata_paths"
        )

    rewards: list[float] = []
    for completion, kata_path in zip(completions, kata_paths):
        text = _completion_to_text(completion)
        defmacro = extract_defmacro(text)
        if defmacro is None:
            rewards.append(-0.2)   # didn't even produce a balanced defmacro
            continue
        try:
            env = _get_env(kata_path)
            _obs, r, _done, _trunc, _info = env.step(defmacro)
            rewards.append(max(-0.5, min(1.5, float(r))))
        except Exception as e:
            # Treat env errors as a mild penalty (likely model emitted
            # something the SBCL server couldn't even parse).
            print(f"  reward err: {Path(kata_path).name}: {type(e).__name__}: {e}", file=sys.stderr)
            rewards.append(-0.3)
    return rewards


# ─── baseline-only mode ───────────────────────────────────────────────

def _run_baseline(model, tokenizer, eval_ds) -> int:
    """Greedy-ish single rollout per eval kata; report mean reward,
    compile rate (>0), and full-credit rate (>=0.99). Gate C consumer
    parses the `baseline_mean=` line.
    """
    FastLanguageModel.for_inference(model)
    rewards: list[float] = []
    n = len(eval_ds)
    print(f"\n=== Baseline rollout over {n} eval katas ===")
    for i, row in enumerate(eval_ds):
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_COMPLETION_LEN,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        r = macro_gym_reward(
            prompts=[row["prompt"]],
            completions=[completion],
            kata_path=[row["kata_path"]],
        )[0]
        rewards.append(r)
        if (i + 1) % 10 == 0 or i + 1 == n:
            running = statistics.fmean(rewards)
            print(f"  [{i + 1:>4d}/{n}] running mean={running:+.4f}")

    if not rewards:
        print("baseline_mean=0.0")
        print("baseline_compile_rate=0.0")
        print("baseline_full_credit=0.0")
        return 0

    mean = statistics.fmean(rewards)
    compile_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    full_credit  = sum(1 for r in rewards if r >= 0.99) / len(rewards)
    print(f"\n=== Baseline summary (n={len(rewards)}) ===")
    print(f"baseline_mean={mean:.4f}")
    print(f"baseline_compile_rate={compile_rate:.4f}")
    print(f"baseline_full_credit={full_credit:.4f}")
    return 0


# ─── main ─────────────────────────────────────────────────────────────

def main() -> int:
    if not HF_REPO:
        print("WARN: HF_REPO not set — final adapter will only be saved locally", file=sys.stderr)
    if not KATA_ROOT.exists():
        print(f"ERROR: KATA_ROOT does not exist: {KATA_ROOT}", file=sys.stderr)
        print(f"  Generate katas first: python scripts/cloud/cl_ds_to_katas.py "
              f"--dataset j14i/cl-ds --output-dir {KATA_ROOT} --validate", file=sys.stderr)
        return 1

    # Verify macro-gym is importable before doing anything expensive
    try:
        import macro_gym  # noqa: F401
    except ImportError as e:
        print(f"ERROR: macro-gym not installed: {e}", file=sys.stderr)
        print("  pip install git+https://github.com/jborkowski/macro-gym", file=sys.stderr)
        return 1

    wandb_ok = _maybe_init_wandb()

    print(f"Loading policy from {SFT_ADAPTER} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_ADAPTER,    # adapter dir; base model resolved via adapter_config.json
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=os.environ.get("HF_TOKEN"),
    )
    # GRPO updates the existing PEFT adapter in place. Reference for KL
    # is computed by disabling the adapter (built into trl when model is PEFT).

    print(f"Loading katas from {KATA_ROOT} ...")
    full_ds = load_katas(KATA_ROOT)
    if len(full_ds) == 0:
        print(f"ERROR: no katas found under {KATA_ROOT}", file=sys.stderr)
        return 1
    print(f"  {len(full_ds)} katas total")

    by_cat: dict[str, int] = {}
    for cat in full_ds["category"]:
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("  by category: " + ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))

    splits = full_ds.train_test_split(test_size=EVAL_TEST_SIZE, seed=42)
    train_ds, eval_ds = splits["train"], splits["test"]
    print(f"  split: {len(train_ds)} train / {len(eval_ds)} eval")

    if BASELINE_ONLY:
        return _run_baseline(model, tokenizer, eval_ds)

    grpo_kwargs: dict = dict(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        num_generations=NUM_GENERATIONS,
        temperature=TEMPERATURE,
        max_completion_length=MAX_COMPLETION_LEN,
        max_prompt_length=1024,
        beta=BETA,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        seed=42,
        report_to=["wandb"] if wandb_ok else "none",
        # Rollouts: vllm doesn't support qwen3_5 yet, so rollouts via HF generate.
        use_vllm=False,
        # Unsloth long-context GRPO defaults (see RL guide):
        loss_type=LOSS_TYPE,
        epsilon=EPSILON,
        epsilon_high=EPSILON_HIGH,
        delta=DELTA,
        mask_truncated_completions=MASK_TRUNCATED_COMPLETIONS,
    )
    if UNSLOTH_GRPO_MINI_BATCH:
        grpo_kwargs["unsloth_grpo_mini_batch"] = int(UNSLOTH_GRPO_MINI_BATCH)
    if UNSLOTH_LOGIT_CHUNK_MULTIPLIER:
        grpo_kwargs["unsloth_logit_chunk_multiplier"] = int(UNSLOTH_LOGIT_CHUNK_MULTIPLIER)
    config = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[macro_gym_reward],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("\n=== Starting GRPO training ===")
    train_result = trainer.train()

    # Save final adapter to disk
    final_dir = Path(OUTPUT_DIR) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Tidy up env subprocesses
    for env in _env_cache.values():
        try:
            env.close()
        except Exception:
            pass
    _env_cache.clear()

    print("\n=== Training summary ===")
    print(f"Final adapter: {final_dir}")
    print(f"Train loss:    {train_result.training_loss:.4f}")
    print(f"Total steps:   {train_result.global_step}")
    print(f"\nNext: push to HF with `python scripts/cloud/upload_to_hf.py` "
          f"(set HF_REPO={HF_REPO or '<your-grpo-repo>'} first).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
