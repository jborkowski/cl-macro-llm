#!/usr/bin/env python3
"""SFT LoRA fine-tuning of Qwen3.6-27B on Common Lisp macro traces.

Loads the base model in 4-bit (NF4 + bf16 compute), attaches a rank-32 LoRA
adapter to all attention/MLP projections, and trains with TRL's SFTTrainer
on the chat-format JSONL produced by scripts/prepare_data_full.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "full" / "train.jsonl"
VALID_FILE = PROJECT_ROOT / "data" / "processed" / "full" / "valid.jsonl"

MODEL_NAME = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-27B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
FINAL_ADAPTER_DIR = Path(OUTPUT_DIR) / "final_adapter"


def main() -> None:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
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

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALID_FILE),
        },
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
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
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
