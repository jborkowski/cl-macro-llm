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

TED-blended reward shaping (conservative; macro-gym Result.semantic_eq_score):
    --ted-blend FLOAT  / TED_BLEND  default 0.3   — max bonus when base reward ≈ 0
    --ted-band  FLOAT  / TED_BAND   default 0.05  — |base reward| ≤ band → blend fires

    Blending rule: when the verdict's `semantic_eq_score` is populated AND
    the base reward sits in the [-band, +band] "compiled but 0/1 tests pass"
    bucket, replace the reward with `blend * sim` (so max bonus is `blend`,
    bounded to 0.3 by default). Everything else — sim is None, an error
    was returned, a syntax-error -0.1, or a full-pass 1.0 — is untouched.

    No-op guarantee: when macro-gym returns `semantic_eq_score=None`
    (today's reality on every verdict), rewards are byte-identical to the
    pre-TED path. Set `--ted-blend 0.0` (or TED_BLEND=0) for an explicit
    ablation control run that disables blending even after TED ships.

    Self-test: `python grpo_train.py --self-test` exercises the blend
    arithmetic against three hand-built verdicts (error / band-hit /
    full-pass) and prints expected-vs-computed without loading the model.
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

# Early-out for `--self-test`: arithmetic-only smoke test for the TED
# blend rule. Skip the heavy ML imports so the test can run on a laptop
# (or anywhere unsloth/trl aren't installed).
_SELF_TEST_MODE = "--self-test" in sys.argv

if not _SELF_TEST_MODE:
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
else:
    # Lightweight stubs so class-level `TrainerCallback` references parse.
    # These code paths are unreachable in --self-test mode.
    class TrainerCallback:  # type: ignore[no-redef]
        pass
    class GRPOTrainer:  # type: ignore[no-redef]
        pass
    Dataset = None  # type: ignore[assignment]
    FastLanguageModel = None  # type: ignore[assignment]
    GRPOConfig = None  # type: ignore[assignment]


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

# TED-blended reward shaping (see top-of-file docstring). Defaults are
# conservative: blend in TED only when the base reward sits in the
# "compiled but 0/1 tests pass" band, and cap the bonus at `TED_BLEND`.
# CLI overrides (--ted-blend / --ted-band) win over env vars; both
# default to 0.3 / 0.05 if unset.
TED_BLEND           = float(_env("TED_BLEND", "0.3"))
TED_BAND            = float(_env("TED_BAND",  "0.05"))


def _parse_cli_overrides(argv: list[str]) -> tuple[list[str], bool]:
    """Strip TED-related flags from argv, mutate module globals.

    Returns (remaining_argv, self_test_requested). Kept as a tiny ad-hoc
    parser rather than introducing argparse because the rest of the file
    is env-var driven — adding a full argparse here would be a bigger
    refactor than the feature warrants.
    """
    global TED_BLEND, TED_BAND
    self_test = False
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--self-test":
            self_test = True
            i += 1
        elif a == "--ted-blend" and i + 1 < len(argv):
            TED_BLEND = float(argv[i + 1])
            i += 2
        elif a.startswith("--ted-blend="):
            TED_BLEND = float(a.split("=", 1)[1])
            i += 1
        elif a == "--ted-band" and i + 1 < len(argv):
            TED_BAND = float(argv[i + 1])
            i += 2
        elif a.startswith("--ted-band="):
            TED_BAND = float(a.split("=", 1)[1])
            i += 1
        else:
            remaining.append(a)
            i += 1
    return remaining, self_test


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
        complexity = meta.get("complexity") or "?"
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
            "complexity":    complexity,
            "quality_score": float(meta.get("quality_score") or 1.0),
        })
        _runtime["kata_complexity"][kata_dir.name] = complexity
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
if not _SELF_TEST_MODE:
    from macro_gym import get_grader, shutdown_grader
else:
    def get_grader(): raise RuntimeError("self-test: grader unavailable")
    def shutdown_grader(): pass


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
    "reward_by_tier":    {"basic": [], "intermediate": [], "complex": [], "advanced": []},
    "reward_window":     int(_env("REWARD_WINDOW", "100")),
    "kata_complexity":   {},
}


# ─── curriculum learning ──────────────────────────────────────────────

class CurriculumSampler:
    """Step-aware index sampler for curriculum-learning kata GRPO.

    At step S, restrict to katas whose tier index is <= unlock_tier(S).
    Within the allowed set, sample uniformly with replacement.

    Reads current step from module-level `_runtime["step"]` (the same
    mailbox that `RuntimeControlCallback` already updates each step) so
    we do not need a second callback. See /autoplan audit trail #2.

    NOTE: __iter__ returns exactly len(complexities) indices per call
    (one "virtual epoch"). TRL re-calls __iter__ on each epoch and
    re-reads the current step at that time, advancing the curriculum.
    See audit trail #3.
    """
    TIERS = ("basic", "intermediate", "complex", "advanced")

    def __init__(self, complexities, max_steps, schedule=(0.0, 0.25, 0.5, 0.75)):
        if len(schedule) != len(self.TIERS):
            raise ValueError(f"schedule must have {len(self.TIERS)} entries, got {len(schedule)}")
        if schedule[0] != 0.0:
            raise ValueError("schedule[0] must be 0.0 (basic tier must be unlocked at step 0)")
        if any(a > b for a, b in zip(schedule, schedule[1:])):
            raise ValueError(f"schedule must be monotonic non-decreasing, got {schedule}")

        self.indices_by_tier = [[] for _ in self.TIERS]
        unknown_count = 0
        for i, c in enumerate(complexities):
            if c in self.TIERS:
                self.indices_by_tier[self.TIERS.index(c)].append(i)
            else:
                self.indices_by_tier[-1].append(i)
                unknown_count += 1

        if not self.indices_by_tier[0]:
            raise RuntimeError(
                "CurriculumSampler: 0 'basic' katas in dataset — "
                "curriculum would silently disable for first phase. "
                "Set CURRICULUM_DISABLED=1 or fix the data."
            )

        for i in range(1, len(self.TIERS) - 1):
            if not self.indices_by_tier[i]:
                print(
                    f"WARN: CurriculumSampler: 0 '{self.TIERS[i]}' katas — "
                    f"unlock at progress={schedule[i]:.2f} will add no new samples",
                    file=sys.stderr,
                )

        if unknown_count:
            print(
                f"WARN: CurriculumSampler: {unknown_count} katas with unknown "
                f"complexity → bucketed as 'advanced' (worst case)",
                file=sys.stderr,
            )

        counts = ", ".join(
            f"{name}={len(idx)}"
            for name, idx in zip(self.TIERS, self.indices_by_tier)
        )
        print(
            f"CurriculumSampler: {counts}, max_steps={max_steps}, "
            f"schedule={schedule}",
            file=sys.stderr,
        )

        self.max_steps = max_steps
        self.schedule = schedule
        self._total_len = sum(len(b) for b in self.indices_by_tier)

    def unlock_tier(self, step):
        """Return the highest tier index unlocked at this step."""
        progress = step / max(self.max_steps, 1)
        for tier_idx in range(len(self.TIERS) - 1, -1, -1):
            if progress >= self.schedule[tier_idx]:
                return tier_idx
        return 0

    def allowed_indices(self, step):
        tier = self.unlock_tier(step)
        return [i for t in range(tier + 1) for i in self.indices_by_tier[t]]

    def __iter__(self):
        # Finite one-epoch sampler. Reads step from _runtime mailbox.
        # Granularity: tier is fixed for the duration of one __iter__ call
        # (one epoch's worth of samples). Acceptable for the 4-tier
        # schedule; see audit trail #9.
        import random as _random
        step = _runtime.get("step", 0)
        allowed = self.allowed_indices(step)
        return iter([_random.choice(allowed) for _ in range(self._total_len)])

    def __len__(self):
        return self._total_len


class CurriculumGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that swaps in a CurriculumSampler when provided.

    Overriding `_get_train_sampler` (private TRL API) instead of
    monkey-patching at the instance level: subclassing is durable across
    minor TRL version skews and fails LOUD on signature changes.
    See /autoplan audit trail #1.
    """
    def __init__(self, *args, curriculum_sampler=None, **kwargs):
        self._curriculum_sampler = curriculum_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, *args, **kwargs):
        if self._curriculum_sampler is not None:
            return self._curriculum_sampler
        return super()._get_train_sampler(*args, **kwargs)


class CurriculumSanityCallback(TrainerCallback):
    """Fail loud at training start if the sampler isn't actually wired up.

    Verifies that on the first batch, at step 0, all sampled indices belong
    to tier-0 (basic). If TRL silently re-wraps the sampler (e.g., in a
    SequentialSampler for DDP), this catches it before we waste compute.
    See /autoplan audit trail #1.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self._checked = False

    def on_train_begin(self, args, state, control, **kw):
        if self._checked or self.sampler is None:
            return control
        self._checked = True
        if state.global_step != 0:
            return control
        tier_0 = set(self.sampler.indices_by_tier[0])
        if not tier_0:
            return control
        sample = list(self.sampler)[:8]
        in_tier_0 = sum(1 for s in sample if s in tier_0)
        if in_tier_0 < 8:
            raise RuntimeError(
                f"CurriculumSanityCallback: at step 0, expected all 8 "
                f"sampled indices in tier-0 (basic), got {in_tier_0}/8. "
                f"TRL may have re-wrapped the sampler — check trainer wiring."
            )
        print(
            f"CurriculumSanityCallback: tier-0 assertion passed "
            f"(8/8 samples in 'basic' at step 0)",
            file=sys.stderr,
        )
        return control


def _maybe_dump_samples(
    prompts,
    completions,
    kata_ids,
    rewards,
    verdict_rewards=None,
    semantic_eq_scores=None,
) -> None:
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
                # `reward` is the final (possibly TED-blended) reward; the
                # raw grader output is recorded as `verdict_reward` so we
                # can reconstruct the un-blended distribution post hoc.
                f.write(json.dumps({
                    "step":               step,
                    "kata_id":            kata_ids[i] if kata_ids else None,
                    "prompt":             prompts[i] if prompts else None,
                    "completion":         text,
                    "reward":             rewards[i],
                    "verdict_reward":     verdict_rewards[i] if verdict_rewards is not None else rewards[i],
                    "semantic_eq_score":  semantic_eq_scores[i] if semantic_eq_scores is not None else None,
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

    # Conservative TED-blended reward shaping. The blend ONLY fires when
    # macro-gym populates `semantic_eq_score` (None today, populated
    # post-TED-shipment) AND the base reward sits in the [-band, +band]
    # "compiled but 0/1 tests pass" bucket. Outside that band — syntax
    # errors (-0.1) and full-pass (1.0) — the reward is untouched. Max
    # blended reward is `TED_BLEND * 1.0` (default 0.3), bounding any
    # reward-hacking surface from TED-only optimisation.
    blend_weight = TED_BLEND
    blend_band   = TED_BAND
    verdict_rewards: list[float] = []
    semantic_eq_scores: list[float | None] = []
    rewards: list[float] = []
    for v in verdicts:
        base = float(v.get("reward", -0.1))
        sim = v.get("semantic_eq_score")
        verdict_rewards.append(base)
        semantic_eq_scores.append(sim)
        if blend_weight == 0.0 or sim is None or v.get("error") is not None:
            rewards.append(base)                  # fallback / error / no-TED: untouched
        elif -blend_band <= base <= blend_band:
            rewards.append(blend_weight * float(sim))  # band hit: bounded by blend_weight
        else:
            rewards.append(base)                  # syntax error or full pass: untouched

    # Record per-tier reward for curriculum-aware metrics (audit trail #8)
    kc = _runtime.get("kata_complexity", {})
    rbt = _runtime.get("reward_by_tier", {})
    win = _runtime.get("reward_window", 100)
    for kid, r in zip(kata_ids, rewards):
        tier = kc.get(kid, "advanced")
        if tier not in rbt:
            tier = "advanced"
        rbt[tier].append(r)
        if len(rbt[tier]) > win:
            del rbt[tier][:-win]

    _maybe_dump_samples(
        prompts, completions, kata_ids, rewards,
        verdict_rewards=verdict_rewards,
        semantic_eq_scores=semantic_eq_scores,
    )
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
        enriched = dict(logs)
        rbt = _runtime.get("reward_by_tier", {})
        for tier_name, samples in rbt.items():
            if samples:
                enriched[f"train/reward_{tier_name}"] = sum(samples) / len(samples)
                enriched[f"train/n_{tier_name}"] = len(samples)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, **enriched}) + "\n")
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

    # Stratified eval split by complexity so eval reward stays comparable
    # across the curriculum (otherwise eval at step 100 would be inflated).
    # See /autoplan audit trail #13.
    import collections as _collections
    _rng = random.Random(42)
    by_complexity: dict = _collections.defaultdict(list)
    for i in range(len(full_ds)):
        by_complexity[full_ds[i]["complexity"]].append(i)
    eval_idx, train_idx = [], []
    for _c, _idxs in by_complexity.items():
        _idxs = list(_idxs)
        _rng.shuffle(_idxs)
        _n_eval = max(1, round(len(_idxs) * EVAL_TEST_SIZE)) if len(_idxs) >= 2 else 0
        eval_idx.extend(_idxs[:_n_eval])
        train_idx.extend(_idxs[_n_eval:])
    _rng.shuffle(train_idx)
    _rng.shuffle(eval_idx)
    train_ds = full_ds.select(train_idx)
    eval_ds  = full_ds.select(eval_idx)
    print(f"  stratified split: {len(train_ds)} train / {len(eval_ds)} eval")
    _eval_by_cx = _collections.Counter(eval_ds["complexity"])
    print(f"  eval by complexity: {dict(_eval_by_cx)}")

    # Curriculum learning configuration (see thoughts/2026-05-13-curriculum-learning-plan.md)
    curriculum_disabled = _env("CURRICULUM_DISABLED", "0") != "0"
    curriculum_schedule_str = _env("CURRICULUM_SCHEDULE", "0,0.25,0.5,0.75")
    try:
        curriculum_schedule = tuple(float(x.strip()) for x in curriculum_schedule_str.split(","))
    except ValueError as e:
        print(f"ERROR: bad CURRICULUM_SCHEDULE='{curriculum_schedule_str}': {e}", file=sys.stderr)
        return 1

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
        # use_reentrant=False routes activation checkpoints through the newer
        # PyTorch path that frees scratch buffers between chunks instead of
        # holding them for the full step. Combined with `paged_adamw_8bit`
        # (Adam state paged to host RAM via bitsandbytes), this is the
        # bf16-fidelity CPU-offload route — no quality compromise vs 4-bit
        # base, paid for in step time.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
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

    # Resume from the latest checkpoint if one exists — survives OOM /
    # pod restart without losing prior progress.
    resume = False
    resume_step = 0
    out_path = Path(OUTPUT_DIR)
    if out_path.exists() and any(out_path.glob("checkpoint-*")):
        resume = True
        # Find latest checkpoint and pre-seed _runtime["step"] so the
        # curriculum sampler picks the right tier on the first batch.
        # See /autoplan audit trail #5.
        try:
            latest_ckpt = max(out_path.glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[-1]))
            state_path = latest_ckpt / "trainer_state.json"
            if state_path.exists():
                _state = json.loads(state_path.read_text())
                resume_step = int(_state.get("global_step", 0))
                _runtime["step"] = resume_step
                print(f"  resume: pre-seeded step={resume_step} from {state_path}")
        except Exception as e:
            print(f"  WARN: failed to pre-seed resume step: {type(e).__name__}: {e}",
                  file=sys.stderr)
        print(f"  resuming from latest checkpoint under {OUTPUT_DIR}")

    # Curriculum wiring (audit trail #1, #2)
    sampler = None
    if not curriculum_disabled:
        sampler = CurriculumSampler(
            complexities=list(train_ds["complexity"]),
            max_steps=MAX_STEPS,
            schedule=curriculum_schedule,
        )
        callbacks.append(CurriculumSanityCallback(sampler))
    else:
        print("  CURRICULUM_DISABLED=1 — uniform sampling")

    trainer = CurriculumGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[macro_gym_reward],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
        curriculum_sampler=sampler,
    )

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


def _run_self_test() -> int:
    """Smoke-test the TED-blend arithmetic against three hand-built verdicts.

    Not a pytest — just a sanity gate. Exits 0 on agreement, 1 otherwise.
    Does NOT load the model or touch macro-gym; pure arithmetic on the
    blending rule. Run with `python grpo_train.py --self-test`.
    """
    print(f"[self-test] TED_BLEND={TED_BLEND}  TED_BAND={TED_BAND}")
    blend_weight = TED_BLEND
    blend_band   = TED_BAND

    # Three hand-built verdicts exercising the three branches of the blend.
    verdicts = [
        # (label, verdict, expected)
        ("error-path",
         {"reward": 0.0, "semantic_eq_score": 0.85,
          "error": {"type": "TimeoutError", "message": "x"}},
         0.0),
        ("band-hit (base=0.0, sim=0.9)",
         {"reward": 0.0, "semantic_eq_score": 0.9, "error": None},
         blend_weight * 0.9),
        ("full-pass (base=1.0, sim=0.95)",
         {"reward": 1.0, "semantic_eq_score": 0.95, "error": None},
         1.0),
        # The no-op guarantee: sim=None must yield base byte-identically.
        ("no-TED-yet (sim=None, base=0.0)",
         {"reward": 0.0, "semantic_eq_score": None, "error": None},
         0.0),
        ("no-TED-yet (sim=None, base=-0.1)",
         {"reward": -0.1, "semantic_eq_score": None, "error": None},
         -0.1),
    ]
    ok = True
    for label, v, expected in verdicts:
        base = float(v.get("reward", -0.1))
        sim = v.get("semantic_eq_score")
        if blend_weight == 0.0 or sim is None or v.get("error") is not None:
            got = base
        elif -blend_band <= base <= blend_band:
            got = blend_weight * float(sim)
        else:
            got = base
        agree = abs(got - expected) < 1e-9
        ok = ok and agree
        flag = "OK " if agree else "FAIL"
        print(f"  [{flag}] {label:42s} expected={expected:+.4f}  got={got:+.4f}")

    print("[self-test] CurriculumSampler schedule arithmetic")
    _complexities = (
        ["basic"] * 4 + ["intermediate"] * 3 + ["complex"] * 2 + ["advanced"] * 1
    )
    _s = CurriculumSampler(_complexities, max_steps=100, schedule=(0.0, 0.25, 0.5, 0.75))
    schedule_checks = [
        (0,  0),
        (24, 0),
        (25, 1),
        (49, 1),
        (50, 2),
        (74, 2),
        (75, 3),
        (99, 3),
    ]
    for step, expected in schedule_checks:
        got = _s.unlock_tier(step)
        agree = (got == expected)
        ok = ok and agree
        flag = "OK " if agree else "FAIL"
        print(f"  [{flag}] step={step:3d}  expected_tier={expected}  got_tier={got}")

    print(f"[self-test] {'all passed' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    _argv, _self_test = _parse_cli_overrides(sys.argv[1:])
    sys.argv = [sys.argv[0]] + _argv  # leave the rest untouched for downstream tools
    if _self_test:
        sys.exit(_run_self_test())
    sys.exit(main())
