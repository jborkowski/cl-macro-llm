#!/usr/bin/env python3
"""SFT LoRA fine-tuning of Qwen3.6-27B on the j14i/cl-macros dataset.

Loads the base model in native bf16 (no quantization — targeted at A100 80GB),
attaches a rank-32 LoRA adapter to all attention/MLP projections, and trains
with TRL's SFTTrainer on chat-format messages produced from the public
HuggingFace dataset `j14i/cl-macros`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_NAME = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-27B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATASET_NAME = os.environ.get("DATASET", "j14i/cl-macros-thinking")
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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

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

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=4096,
        packing=False,
        seed=42,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=raw["train"],
        eval_dataset=raw["validation"],
        processing_class=tokenizer,
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
