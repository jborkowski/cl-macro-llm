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
    WANDB_ENTITY       e.g. j14i-n
    WANDB_PROJECT      default cl-macro-llm

Runtime (SBCL parallelism is handled by macro-gym v0.3's MacroGrader):
    MACRO_GYM_POOL_SIZE    default 6    — SBCL workers in the shared grader pool
    MACRO_GYM_HEAP_MB      default 384  — per-worker dynamic heap (SBCL)
    LOG_SAMPLES_EVERY      default 25   — steps between sample dumps
    SAMPLES_PER_DUMP       default 4    — rollouts written per dump
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Unsloth long-context GRPO recommendation: keep vLLM weights swapped out
# while the trainer step runs, then page them back for rollouts. Must be
# set before unsloth is imported.
#
# But: we use use_vllm=False (qwen3_5 unsupported by vLLM), so STANDBY does
# nothing useful and actively blocks PYTORCH_ALLOC_CONF=expandable_segments
# (which Unsloth strips out when STANDBY is on). With OOM at the wire
# (~1.5 GB short on the chunked-logsoftmax scratch), expandable_segments
# is what closes the gap.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback


# ─── env helpers ──────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


SFT_ADAPTER         = _env("SFT_ADAPTER", "j14i/cl-macro-27b-lora")
HF_REPO             = _env("HF_REPO")           # required for final upload
KATA_ROOT           = Path(_env("KATA_ROOT", "/workspace/katas"))
OUTPUT_DIR          = _env("OUTPUT_DIR", "./grpo-output")
MAX_SEQ_LENGTH      = int(_env("MAX_SEQ_LENGTH", "4096"))
MAX_COMPLETION_LEN  = int(_env("MAX_COMPLETION_LEN", "1024"))
NUM_GENERATIONS     = int(_env("NUM_GENERATIONS", "2"))
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

EVAL_TEST_SIZE      = float(_env("EVAL_TEST_SIZE", "0.1"))


SYSTEM_PROMPT = (
    "You are an expert Common Lisp macro programmer. Think step by step "
    "before writing the macro. Always explain your reasoning in <think>...</think> "
    "tags, then provide the defmacro form.\n\n"
    "Your answer will be graded by SBCL: the reference call form is run "
    "through `(macroexpand-1 ...)` against your defmacro, and the result "
    "is compared structurally to a reference expansion. So write a macro "
    "whose expansion matches what `(macroexpand-1 <input>)` would actually "
    "produce — not what you'd want a human to read."
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
    os.environ.setdefault("WANDB_PROJECT", "cl-macro-llm")
    return True


# ─── kata loading ─────────────────────────────────────────────────────

def load_katas(kata_root: Path) -> Dataset:
    """Walk kata directory tree (cl-ds/* + creative/*), emit prompt+meta rows.

    Side effect: symlinks each kata into macro-gym's `katas/` directory so
    `MacroGrader.grade(kata_id, ...)` can resolve them. v0.3's Lisp server
    hardcodes ``*kata-root*`` to ``"katas/"`` relative to the macro-gym
    package, so we shim our generated katas into that path rather than
    monkey-patching the Lisp global.
    """
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

    # Symlink katas into macro-gym's katas/ dir. Idempotent — re-runs just
    # refresh the links. Skips dirs missing setup.lisp + tests.lisp to
    # avoid grader errors later. macro-gym v0.3 still uses
    # `macro_gym.env.KATAS_DIR` as the canonical path; the new grader
    # resolves through the same Lisp `*kata-root*`.
    try:
        import macro_gym.env as _mg
        _mg.KATAS_DIR.mkdir(parents=True, exist_ok=True)
        n_linked = 0
        for kata_dir in candidates:
            if not (kata_dir / "setup.lisp").exists() or \
               not (kata_dir / "tests.lisp").exists():
                continue
            link = _mg.KATAS_DIR / kata_dir.name
            if link.is_symlink():
                if link.resolve() == kata_dir.resolve():
                    continue
                link.unlink()
            elif link.exists():
                continue  # don't clobber a real dir (e.g., bundled samples)
            link.symlink_to(kata_dir.resolve())
            n_linked += 1
        print(f"  symlinked {n_linked} katas into {_mg.KATAS_DIR}")
    except Exception as e:
        print(f"  warning: macro-gym kata symlink setup failed: {type(e).__name__}: {e}",
              file=sys.stderr)

    for kata_dir in sorted(candidates):
        meta = json.loads((kata_dir / "metadata.json").read_text())
        instruction = (meta.get("instruction") or "").strip()
        if not instruction:
            continue
        rows.append({
            # macro-gym v0.3's reward_fn expects `kata_ids` (plural) — keep
            # the dataset column name aligned with the public API.
            "kata_ids":      kata_dir.name,
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


# Reward backend: macro-gym v0.3's MacroGrader (a shared SBCL worker pool
# with per-worker kata-setup caching). The grader serialises macroexpand
# requests across MACRO_GYM_POOL_SIZE workers — no per-kata bag, no env
# checkout, no reset/step state. Trainer's reward fn is just an
# extract-defmacro shim around grader.grade_batch().
import random
from macro_gym import get_grader, shutdown_grader


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


# Mutable runtime state — updated by callbacks, read by reward fn for
# sample dumping. See RuntimeControlCallback / SampleDumperCallback.
_runtime: dict = {
    "step":              0,
    "log_samples_every": int(_env("LOG_SAMPLES_EVERY", "25")),
    "sample_now":        False,
    "n_samples_per_dump": int(_env("SAMPLES_PER_DUMP", "4")),
    "output_dir":        Path(OUTPUT_DIR),
}


def _maybe_dump_samples(prompts, completions, kata_ids, rewards) -> None:
    step  = _runtime["step"]
    every = _runtime["log_samples_every"]
    if not (_runtime["sample_now"] or (every > 0 and step > 0 and step % every == 0)):
        return
    _runtime["sample_now"] = False

    n = min(_runtime["n_samples_per_dump"], len(completions))
    if n <= 0:
        return
    idxs = random.sample(range(len(completions)), n)
    out = _runtime["output_dir"] / f"samples-step-{step:05d}.jsonl"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for i in idxs:
                text = _completion_to_text(completions[i])
                f.write(json.dumps({
                    "step":               step,
                    "kata_id":            kata_ids[i] if kata_ids else None,
                    "prompt":             prompts[i] if prompts else None,
                    "completion":         text,
                    "reward":             rewards[i],
                    "defmacro_extracted": extract_defmacro(text),
                }, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  sample dump failed: {type(e).__name__}: {e}", file=sys.stderr)


def macro_gym_reward(prompts, completions, **kwargs) -> list[float]:
    """GRPO reward callable backed by macro-gym v0.3's MacroGrader.

    The grader pools MACRO_GYM_POOL_SIZE SBCL workers (default 6), each
    caching kata setup on first hit, with TTL-based recycling to bound
    memory drift across thousands of calls. Parallelism is fully
    delegated — the trainer just extracts the defmacro from each
    chat-format completion and hands a batch to the grader.

    `kata_ids` is required as a kwarg, supplied by the dataset (rename
    of the old `kata_id` column for v0.3 API alignment). It is
    authoritative — TRL's `prompts` arg is intentionally ignored for
    routing so the model can't game which kata grades its output.

    Also writes a sample to OUTPUT_DIR/samples-step-NNN.jsonl every
    LOG_SAMPLES_EVERY training steps (or when control.json sets sample_now).
    """
    kata_ids = kwargs.get("kata_ids") or []
    if len(kata_ids) != len(completions):
        raise RuntimeError(
            f"reward fn: got {len(completions)} completions but "
            f"{len(kata_ids)} kata_ids"
        )
    if not completions:
        return []

    # Chat-format completions need their defmacro extracted; the grader
    # treats `macro_src` as raw Lisp source. Empty string for completions
    # with no parseable defmacro — grader returns -0.1 (syntax error).
    macros = []
    for c in completions:
        text = _completion_to_text(c)
        m = extract_defmacro(text)
        macros.append(m if m is not None else "")

    grader = get_grader()
    verdicts = grader.grade_batch(list(zip(kata_ids, macros)))
    rewards = [float(v.get("reward", -0.1)) for v in verdicts]

    _maybe_dump_samples(prompts, completions, kata_ids, rewards)
    return rewards


# ─── runtime control plane ────────────────────────────────────────────
#
# Three files under OUTPUT_DIR let me tune the run from a different shell
# without holding an SSH session open:
#
#   metrics.jsonl              — appended every log step (loss/reward/kl/…)
#   samples-step-NNN.jsonl     — 4 rollouts every LOG_SAMPLES_EVERY steps
#                                (see _maybe_dump_samples in the reward fn)
#   control.json               — polled at on_step_begin; recognised keys:
#                                  stop:      true  → checkpoint + clean exit
#                                  save_now:  true  → force a checkpoint write
#                                  sample_now:true  → dump samples on next batch
#                                  temperature: f   → swap rollout temperature
#                                  log_samples_every: int → change cadence

class MetricsLoggerCallback(TrainerCallback):
    """Append every set of logged metrics to OUTPUT_DIR/metrics.jsonl."""

    def __init__(self, metrics_path: Path):
        self.path = metrics_path

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")
        except Exception as e:
            print(f"  metrics log failed: {type(e).__name__}: {e}", file=sys.stderr)


class RuntimeControlCallback(TrainerCallback):
    """Poll control.json each step; honour stop/save_now/sample_now/etc.

    Re-reads only when the file's mtime changes — cheap and chatter-free.
    """

    def __init__(self, ctl_path: Path, trainer_args):
        self.ctl_path = ctl_path
        self.trainer_args = trainer_args
        self._last_mtime: float = 0.0

    def _read(self) -> dict | None:
        try:
            if not self.ctl_path.exists():
                return None
            m = self.ctl_path.stat().st_mtime
            if m == self._last_mtime:
                return None
            self._last_mtime = m
            return json.loads(self.ctl_path.read_text())
        except Exception as e:
            print(f"  control.json read failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return None

    def on_step_begin(self, args, state, control, **kw):
        # Keep _runtime["step"] fresh so the reward fn can name sample files.
        _runtime["step"] = state.global_step

        ctl = self._read()
        if ctl is None:
            return control

        if ctl.get("stop"):
            print(f"[control] stop requested at step {state.global_step}; "
                  f"checkpointing and exiting cleanly", file=sys.stderr)
            control.should_save = True
            control.should_training_stop = True
        if ctl.get("save_now"):
            print(f"[control] save_now requested at step {state.global_step}",
                  file=sys.stderr)
            control.should_save = True
        if ctl.get("sample_now"):
            print(f"[control] sample_now requested at step {state.global_step}",
                  file=sys.stderr)
            _runtime["sample_now"] = True
        if "log_samples_every" in ctl:
            try:
                _runtime["log_samples_every"] = int(ctl["log_samples_every"])
                print(f"[control] log_samples_every = "
                      f"{_runtime['log_samples_every']}", file=sys.stderr)
            except (TypeError, ValueError):
                pass
        if "temperature" in ctl:
            try:
                t = float(ctl["temperature"])
                # GRPOConfig.temperature is read at rollout time on most TRL
                # versions; mutating args.temperature is best-effort.
                self.trainer_args.temperature = t
                print(f"[control] temperature = {t}", file=sys.stderr)
            except (TypeError, ValueError):
                pass
        return control


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

    grpo_kwargs: dict = dict(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        num_generations=NUM_GENERATIONS,
        temperature=TEMPERATURE,
        max_completion_length=MAX_COMPLETION_LEN,
        max_prompt_length=512,
        gradient_checkpointing=True,
        beta=BETA,
        logging_steps=5,
        # trl GRPOConfig.__post_init__ requires eval_batch_size * world_size
        # to be divisible by num_generations. With world_size=1 and
        # num_generations=8, pin per_device_eval_batch_size=NUM_GENERATIONS.
        # Eval here is mostly monitoring — actual RL signal is rollout reward.
        eval_strategy="no",
        per_device_eval_batch_size=NUM_GENERATIONS,
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
    # Durable checkpoints via HF Hub — survives pod death without a network
    # volume. Trainer pushes a `last-checkpoint` branch on every save_steps
    # (~600 MB LoRA × 20-30s upload — negligible vs the GRPO step time).
    if HF_REPO and os.environ.get("HF_TOKEN"):
        grpo_kwargs.update(
            push_to_hub=True,
            hub_model_id=HF_REPO,
            hub_strategy="checkpoint",
            hub_token=os.environ["HF_TOKEN"],
            hub_private_repo=False,
        )
        print(f"  HF Hub checkpointing → {HF_REPO} (last-checkpoint branch every {grpo_kwargs['save_steps']} steps)")
    if UNSLOTH_GRPO_MINI_BATCH:
        grpo_kwargs["unsloth_grpo_mini_batch"] = int(UNSLOTH_GRPO_MINI_BATCH)
    if UNSLOTH_LOGIT_CHUNK_MULTIPLIER:
        grpo_kwargs["unsloth_logit_chunk_multiplier"] = int(UNSLOTH_LOGIT_CHUNK_MULTIPLIER)
    config = GRPOConfig(**grpo_kwargs)

    # Make sure the reward fn writes sample dumps next to the actual run.
    _runtime["output_dir"] = Path(OUTPUT_DIR)

    metrics_path = Path(OUTPUT_DIR) / "metrics.jsonl"
    control_path = Path(OUTPUT_DIR) / "control.json"
    callbacks = [
        MetricsLoggerCallback(metrics_path),
        RuntimeControlCallback(control_path, config),
    ]
    print(f"  metrics → {metrics_path}")
    print(f"  control ← {control_path}  (edit this file to tune at runtime)")
    print(f"  samples → {OUTPUT_DIR}/samples-step-NNN.jsonl  "
          f"(every {_runtime['log_samples_every']} steps)")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[macro_gym_reward],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )

    # Resume from the latest checkpoint if one exists — survives OOM /
    # pod restart without losing prior progress.
    resume = False
    out_path = Path(OUTPUT_DIR)
    if out_path.exists() and any(out_path.glob("checkpoint-*")):
        resume = True
        print(f"  resuming from latest checkpoint under {OUTPUT_DIR}")

    print("\n=== Starting GRPO training ===")
    train_result = trainer.train(resume_from_checkpoint=resume)

    # Save final adapter to disk
    final_dir = Path(OUTPUT_DIR) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Tidy up the shared SBCL worker pool (v0.3 grader's singleton).
    shutdown_grader()

    print("\n=== Training summary ===")
    print(f"Final adapter: {final_dir}")
    print(f"Train loss:    {train_result.training_loss:.4f}")
    print(f"Total steps:   {train_result.global_step}")
    print(f"\nNext: push to HF with `python scripts/cloud/upload_to_hf.py` "
          f"(set HF_REPO={HF_REPO or '<your-grpo-repo>'} first).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
