#!/usr/bin/env python3
"""Inference smoke-test: load Qwen3.6-27B + trained LoRA adapter and generate."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


SYSTEM_PROMPT = (
    "You are an expert Common Lisp macro programmer. Think step by step "
    "before writing the macro. Always explain your reasoning in <think>...</think> "
    "tags, then provide the defmacro form."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-27B"),
        help="HF model id of the base model.",
    )
    p.add_argument(
        "--adapter",
        default="./output/final_adapter",
        help="Path to the trained LoRA adapter directory.",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="User prompt. If omitted, read from stdin.",
    )
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.prompt is None:
        if sys.stdin.isatty():
            print("Enter prompt (Ctrl-D to end):", file=sys.stderr)
        prompt = sys.stdin.read().strip()
    else:
        prompt = args.prompt
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        sys.exit(2)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
