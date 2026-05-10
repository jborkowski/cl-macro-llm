#!/usr/bin/env python3
"""SFT LoRA fine-tuning of Qwen3.6-27B via Unsloth on a single A100 80GB.

Qwen3.6-27B (model_type: qwen3_5) uses a hybrid Gated DeltaNet + full-attention
architecture and is registered in `transformers` as Qwen3_5ForConditionalGeneration
(a multimodal class). Vanilla `AutoModelForCausalLM` + PEFT can't load it, and
even if it could, the Mamba-style linear-attention blocks need custom kernels
that Unsloth ships and `transformers` does not. This script uses Unsloth's
FastLanguageModel, which auto-handles model loading, the DeltaNet kernels,
gradient checkpointing, and PEFT integration. See:

    https://unsloth.ai/docs/models/qwen3.5/fine-tune
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


def _env(name: str, default: str) -> str:
    """os.environ.get but treats empty string the same as unset.

    The launcher exports optional overrides as `KEY=''` for any var that
    isn't set locally, so plain `os.environ.get(name, default)` would
    return the empty string and skip the default.
    """
    return os.environ.get(name) or default


MODEL_NAME = _env("BASE_MODEL", "Qwen/Qwen3.6-27B")
OUTPUT_DIR = _env("OUTPUT_DIR", "./output")
DATASET_NAME = _env("DATASET", "j14i/cl-macros-thinking")
MAX_SEQ_LENGTH = int(_env("MAX_SEQ_LENGTH", "4096"))
FINAL_ADAPTER_DIR = Path(OUTPUT_DIR) / "final_adapter"


def to_messages(example: dict) -> dict:
    """Convert a {instruction, input, output, ...} row into chat messages.

    The `instruction` text already embeds the macro call form, so we don't
    duplicate `input`. `output` is the defmacro definition we want the model
    to learn to produce.
    """
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def _maybe_init_wandb() -> bool:
    """Try to log in to Weights & Biases and confirm an entity exists.
    Returns True only when SFTConfig can safely set report_to=["wandb"].

    Login alone is not enough: HF Trainer's wandb callback later calls
    wandb.init() which raises `entity not specified, and viewer has no
    default entity` for accounts that have never set up a team or default
    username. Pre-check the viewer's entity here and disable wandb
    gracefully if it's missing — training shouldn't die for telemetry.
    Override by exporting WANDB_ENTITY=<team-or-username>.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="never", relogin=True)
    except Exception as e:
        print(f"wandb login failed ({type(e).__name__}: {e}); training without W&B reporting.")
        return False
    if os.environ.get("WANDB_ENTITY"):
        return True
    try:
        viewer = wandb.Api().viewer
        entity = getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    except Exception:
        entity = None
    if not entity:
        print(
            "wandb account has no default entity; set WANDB_ENTITY in .env "
            "(e.g. your wandb username or team). Skipping W&B for this run."
        )
        return False
    os.environ.setdefault("WANDB_ENTITY", entity)
    return True


def main() -> None:
    wandb_ok = _maybe_init_wandb()

    # load_in_16bit=True → native bf16; QLoRA (load_in_4bit) is explicitly
    # discouraged by Unsloth for Qwen3.5/3.6 models.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=os.environ.get("HF_TOKEN"),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    raw = load_dataset(DATASET_NAME, token=os.environ.get("HF_TOKEN"))
    if "validation" not in raw:
        if "test" in raw:
            raw["validation"] = raw["test"]
        else:
            split = raw["train"].train_test_split(test_size=0.1, seed=42)
            raw = {"train": split["train"], "validation": split["test"]}

    train_cols = raw["train"].column_names
    if "messages" not in train_cols:
        required = {"instruction", "output"}
        missing = required - set(train_cols)
        if missing:
            raise RuntimeError(
                f"Dataset {DATASET_NAME!r} has columns {train_cols} — "
                f"expected either a 'messages' field or {sorted(required)}."
            )
        raw["train"] = raw["train"].map(to_messages, remove_columns=train_cols)
        raw["validation"] = raw["validation"].map(
            to_messages, remove_columns=raw["validation"].column_names
        )

    print(
        f"Loaded {DATASET_NAME}: "
        f"{len(raw['train'])} train / {len(raw['validation'])} validation"
    )

    # warmup_ratio is deprecated in trl 5.2; compute equivalent warmup_steps.
    # 5% of total optimizer steps (effective batch 8, 3 epochs, ~1828 rows).
    _effective_batch = 1 * 8
    _total_steps = (len(raw["train"]) // _effective_batch + 1) * 3
    _warmup_steps = max(10, int(_total_steps * 0.05))

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps,
        bf16=True,
        optim="adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        seed=42,
        report_to=["wandb"] if wandb_ok else "none",
    )

    def formatting_func(examples: dict) -> list:
        """Render chat-format rows. Unsloth probes this twice with different
        shapes: first with a single example (`messages` is one conversation
        — a list of role/content dicts), then later with a batch (`messages`
        is a list of conversations). Return a list in both cases.
        """
        msgs = examples["messages"]
        if msgs and isinstance(msgs[0], dict):
            return [tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )]
        return [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False,
            )
            for m in msgs
        ]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=raw["train"],
        eval_dataset=raw["validation"],
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))

    print("\n=== Training summary ===")
    print(f"Train loss:      {train_result.training_loss:.4f}")
    print(f"Eval loss:       {eval_metrics.get('eval_loss', float('nan')):.4f}")
    print(f"Trainable params: {trainable:,} / {total:,}")
    print(f"Adapter saved to: {FINAL_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
