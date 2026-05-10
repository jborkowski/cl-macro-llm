#!/usr/bin/env python3
"""Inference smoke-test: load Qwen3.6-27B + trained LoRA adapter and generate.

Uses Unsloth's FastLanguageModel for the same reason train_sft.py does —
Qwen3.6 (model_type: qwen3_5) needs the custom Mamba/DeltaNet kernels and
isn't loadable via plain AutoModelForCausalLM. Pointing `model_name` at the
adapter directory (which contains an adapter_config.json with the base
model id) makes Unsloth load both the base weights and the adapter in one
call — the documented inference pattern.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are an expert Common Lisp macro programmer. Think step by step "
    "before writing the macro. Always explain your reasoning in <think>...</think> "
    "tags, then provide the defmacro form."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapter",
        default="./output/final_adapter",
        help="Path to the trained LoRA adapter directory.",
    )
    p.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-27B"),
        help="HF model id of the base. Ignored when --adapter is present "
        "(the adapter's adapter_config.json holds the base reference).",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="User prompt. If omitted, read from stdin.",
    )
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-seq-length", type=int, default=4096)
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

    model_name = args.adapter if os.path.isdir(args.adapter) else args.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=os.environ.get("HF_TOKEN"),
    )
    FastLanguageModel.for_inference(model)

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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
