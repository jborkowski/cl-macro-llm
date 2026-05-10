#!/usr/bin/env python3
"""Balance the cl-macros dataset and emit chat-format train/val splits."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "cl-macros" / "train.jsonl"
OUT_DIR = ROOT / "data" / "processed"
TRAIN_OUT = OUT_DIR / "train_chat.jsonl"
VAL_OUT = OUT_DIR / "val_chat.jsonl"

BASIC_MIN_LEN = 30
INTERMEDIATE_MIN_LEN = 50
BASIC_TARGET = 400
CONTROL_FLOW_TARGET = 400
MINORITY_TARGET = 200
MINORITY_CATEGORIES = {"anaphoric", "dsl", "capture-management"}
VAL_FRACTION = 0.10

SYSTEM_PROMPT = (
    "You are an expert Common Lisp macro programmer. "
    "Think step by step before writing the macro. "
    "Always explain your reasoning in <think>...</think> tags, "
    "then provide the defmacro form."
)

PROMPT_VARIATIONS = [
    "You are a Common Lisp macro expert. Write the macro expansion.",
    "Generate a defmacro form for the following code pattern.",
    "Write a Common Lisp macro that handles this transformation.",
    "Define a defmacro that transforms the following code.",
]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def strategy_for(category, complexity):
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


def make_thinking_trace(record):
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


def to_chat(record, system_prompt=SYSTEM_PROMPT):
    trace = make_thinking_trace(record)
    assistant = f"{trace}\n{record.get('output', '')}"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record.get("instruction", "")},
            {"role": "assistant", "content": assistant},
        ]
    }


def summarize(label, rows):
    by_complexity = Counter(r.get("complexity", "unknown") for r in rows)
    by_category = Counter(r.get("category", "unknown") for r in rows)
    print(f"\n[{label}] total={len(rows)}")
    print("  by complexity:")
    for k, v in sorted(by_complexity.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<14} {v}")
    print("  by category:")
    for k, v in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22} {v}")


def passes_length_filter(record):
    output = record.get("output") or ""
    complexity = (record.get("complexity") or "").lower()
    if complexity == "basic":
        return len(output) >= BASIC_MIN_LEN and "defmacro" in output
    if complexity == "intermediate":
        return len(output) >= INTERMEDIATE_MIN_LEN
    return True


def main():
    raw = load_jsonl(RAW_PATH)
    print(f"loaded {len(raw)} raw records from {RAW_PATH}")

    filtered = [r for r in raw if passes_length_filter(r)]
    print(
        f"after per-complexity length filter: {len(filtered)} "
        f"(basic >= {BASIC_MIN_LEN} + defmacro, intermediate >= {INTERMEDIATE_MIN_LEN}, advanced unfiltered)"
    )

    by_complexity = defaultdict(list)
    for rec in filtered:
        by_complexity[rec.get("complexity", "unknown")].append(rec)

    basic = by_complexity.get("basic", [])
    if len(basic) > BASIC_TARGET:
        basic = random.sample(basic, BASIC_TARGET)
    intermediate = by_complexity.get("intermediate", [])
    advanced = by_complexity.get("advanced", [])
    other_complexity = [
        rec
        for comp, bucket in by_complexity.items()
        if comp not in {"basic", "intermediate", "advanced"}
        for rec in bucket
    ]

    complexity_balanced = basic + intermediate + advanced + other_complexity

    by_category = defaultdict(list)
    for rec in complexity_balanced:
        by_category[rec.get("category", "unknown")].append(rec)

    control_flow = by_category.pop("control-flow", [])
    if len(control_flow) > CONTROL_FLOW_TARGET:
        control_flow = random.sample(control_flow, CONTROL_FLOW_TARGET)

    rebalanced = list(control_flow)
    for cat, bucket in by_category.items():
        rebalanced.extend(bucket)

    final_records = []
    final_meta = []
    for rec in rebalanced:
        final_records.append(to_chat(rec))
        final_meta.append(rec)

    by_category_after = defaultdict(list)
    for rec in rebalanced:
        by_category_after[rec.get("category", "unknown")].append(rec)

    oversample_added = Counter()
    for cat in MINORITY_CATEGORIES:
        bucket = by_category_after.get(cat, [])
        if not bucket:
            continue
        needed = MINORITY_TARGET - len(bucket)
        if needed <= 0:
            continue
        for i in range(needed):
            base = bucket[i % len(bucket)]
            variant = PROMPT_VARIATIONS[i % len(PROMPT_VARIATIONS)]
            chat = to_chat(base, system_prompt=variant)
            final_records.append(chat)
            final_meta.append(base)
            oversample_added[cat] += 1

    print("\nbalanced complexity buckets:")
    print(f"  basic        {len(basic)}")
    print(f"  intermediate {len(intermediate)}")
    print(f"  advanced     {len(advanced)}")
    if other_complexity:
        print(f"  other        {len(other_complexity)}")
    print(f"control-flow after undersample: {len(control_flow)}")
    if oversample_added:
        print("oversampled minority categories:")
        for cat, n in oversample_added.items():
            print(f"  {cat:<22} +{n}")

    paired = list(zip(final_records, final_meta))
    train_pairs, val_pairs = stratified_split_pairs(paired, VAL_FRACTION)

    train_chat = [chat for chat, _ in train_pairs]
    val_chat = [chat for chat, _ in val_pairs]
    train_meta = [meta for _, meta in train_pairs]
    val_meta = [meta for _, meta in val_pairs]

    write_jsonl(TRAIN_OUT, train_chat)
    write_jsonl(VAL_OUT, val_chat)

    summarize("train", train_meta)
    summarize("val", val_meta)
    print(f"\nwrote {len(train_chat)} -> {TRAIN_OUT}")
    print(f"wrote {len(val_chat)} -> {VAL_OUT}")


def stratified_split_pairs(pairs, val_fraction):
    by_complexity = defaultdict(list)
    for pair in pairs:
        complexity = pair[1].get("complexity", "unknown")
        by_complexity[complexity].append(pair)
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


if __name__ == "__main__":
    main()
