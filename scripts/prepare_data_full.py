#!/usr/bin/env python3
"""Emit the *full* cl-macros dataset in chat format with minimal filtering.

Sibling of prepare_data.py — same record schema and placeholder thinking trace,
but no complexity rebalancing or category undersampling. Use this for the
baseline training run that should see every available macro example."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "cl-macros" / "train.jsonl"
OUT_DIR = ROOT / "data" / "processed"
TRAIN_OUT = OUT_DIR / "train_chat_full.jsonl"
VAL_OUT = OUT_DIR / "val_chat_full.jsonl"

MIN_OUTPUT_LEN = 30
VAL_FRACTION = 0.10

SYSTEM_PROMPT = (
    "You are an expert Common Lisp macro programmer. "
    "Think step by step before writing the macro. "
    "Always explain your reasoning in <think>...</think> tags, "
    "then provide the defmacro form."
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def strategy_for(category: str | None, complexity: str | None) -> str:
    cat = (category or "").lower()
    if cat == "anaphoric":
        return "Introduce an implicit binding (e.g. `it`) captured by the macro body."
    if cat == "control-flow":
        return "Expand into conditional/iteration primitives that wrap the supplied forms."
    if cat == "dsl":
        return "Translate the surface syntax into the underlying Lisp forms of the DSL."
    if cat == "capture-management":
        return "Use gensyms to avoid variable capture while binding intermediate values."
    if cat == "binding":
        return "Establish lexical bindings around the body forms."
    if cat == "iteration":
        return "Generate a loop construct that walks the supplied sequence/forms."
    if (complexity or "").lower() == "advanced":
        return "Plan the expansion shape, then assemble it with quasiquotation and gensyms."
    return "Map each input fragment to its place in the expansion via quasiquotation."


def make_thinking_trace(record: dict) -> str:
    snippet = (record.get("input") or "").strip().splitlines()
    snippet_line = snippet[0] if snippet else ""
    if len(snippet_line) > 160:
        snippet_line = snippet_line[:157] + "..."
    category = record.get("category") or "unknown"
    complexity = record.get("complexity") or "unknown"
    return (
        "<think>\n"
        f"Analyzing the macro call pattern: {snippet_line}\n"
        f"Category: {category}\n"
        f"Complexity: {complexity}\n"
        f"Approach: {strategy_for(category, complexity)}\n"
        "</think>"
    )


def to_chat(record: dict) -> dict:
    trace = make_thinking_trace(record)
    assistant = f"{trace}\n{record.get('output', '')}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.get("instruction", "")},
            {"role": "assistant", "content": assistant},
        ]
    }


def passes_filter(record: dict) -> bool:
    out = record.get("output") or ""
    return len(out) >= MIN_OUTPUT_LEN and "defmacro" in out


def stratified_split(
    pairs: list[tuple[dict, dict]], val_fraction: float
) -> tuple[list[tuple[dict, dict]], list[tuple[dict, dict]]]:
    by_complexity: dict[str, list] = defaultdict(list)
    for pair in pairs:
        by_complexity[pair[1].get("complexity", "unknown")].append(pair)
    train, val = [], []
    for bucket in by_complexity.values():
        shuffled = bucket[:]
        random.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def summarize(label: str, rows: list[dict]) -> None:
    by_complexity = Counter(r.get("complexity", "unknown") for r in rows)
    by_category = Counter(r.get("category", "unknown") for r in rows)
    print(f"\n[{label}] total={len(rows)}")
    print("  by complexity:")
    for k, v in sorted(by_complexity.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<14} {v}")
    print("  by category:")
    for k, v in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22} {v}")


def main() -> None:
    raw = load_jsonl(RAW_PATH)
    print(f"loaded {len(raw)} raw records from {RAW_PATH}")

    kept = [r for r in raw if passes_filter(r)]
    print(
        f"after minimal filter (output >= {MIN_OUTPUT_LEN} chars and contains "
        f"'defmacro'): {len(kept)}"
    )

    pairs = [(to_chat(r), r) for r in kept]
    train_pairs, val_pairs = stratified_split(pairs, VAL_FRACTION)

    train_chat = [chat for chat, _ in train_pairs]
    val_chat = [chat for chat, _ in val_pairs]
    train_meta = [meta for _, meta in train_pairs]
    val_meta = [meta for _, meta in val_pairs]

    write_jsonl(TRAIN_OUT, train_chat)
    write_jsonl(VAL_OUT, val_chat)

    summarize("train_full", train_meta)
    summarize("val_full", val_meta)
    print(f"\nwrote {len(train_chat)} -> {TRAIN_OUT}")
    print(f"wrote {len(val_chat)} -> {VAL_OUT}")


if __name__ == "__main__":
    main()
