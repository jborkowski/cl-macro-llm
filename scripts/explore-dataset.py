#!/usr/bin/env python3
"""Explore j14i/cl-macros dataset structure and distribution."""

import json
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "cl-macros"

def explore():
    # --- File counts ---
    print("=== FILE COUNTS ===")
    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
    for f in jsonl_files:
        lines = sum(1 for _ in open(f))
        print(f"  {f.name}: {lines} records")

    # --- First record structure ---
    print("\n=== TRAIN CHAT: FIRST RECORD ===")
    with open(DATA_DIR / "train_chat.jsonl") as f:
        d = json.loads(f.readline())
        print(f"  Keys: {list(d.keys())}")
        for k, v in d.items():
            if k != "messages":
                print(f"  {k}: {repr(v)}")
            else:
                print(f"  messages: {len(v)} turns, roles: {[m['role'] for m in v]}")
                for m in v:
                    print(f"    [{m['role']}]: {m['content'][:120]}...")

    # --- Check all keys across records ---
    print("\n=== ALL KEYS IN DATASET ===")
    all_keys = Counter()
    for fname in ["train_chat.jsonl", "val_chat.jsonl", "test_chat.jsonl"]:
        with open(DATA_DIR / fname) as f:
            for line in f:
                d = json.loads(line)
                for k in d.keys():
                    all_keys[k] += 1
    for k, c in all_keys.most_common():
        print(f"  {k}: {c} records")

    # --- Check if difficulty/category exist in records ---
    print("\n=== METADATA CHECK ===")
    with open(DATA_DIR / "train_chat.jsonl") as f:
        first = json.loads(f.readline())
    meta_keys = [k for k in first if k != "messages"]
    if meta_keys:
        print(f"  Extra keys per record: {meta_keys}")
        for k in meta_keys:
            vals = Counter()
            with open(DATA_DIR / "train_chat.jsonl") as f:
                for line in f:
                    d = json.loads(line)
                    vals[str(d.get(k, "missing"))] += 1
            print(f"\n  Distribution of '{k}':")
            for v, c in vals.most_common():
                print(f"    {v}: {c}")
    else:
        print("  No metadata keys -- only 'messages' in each record")
        print("  Checking if metadata is embedded in messages...")
        # Check user message patterns
        patterns = Counter()
        with open(DATA_DIR / "train_chat.jsonl") as f:
            for line in f:
                d = json.loads(line)
                user_msg = d["messages"][0]["content"]
                if "difficulty" in user_msg.lower():
                    patterns["has_difficulty_hint"] += 1
                if "category" in user_msg.lower():
                    patterns["has_category_hint"] += 1
                if "Pattern" in user_msg:
                    patterns["has_pattern_tag"] += 1
        for p, c in patterns.most_common():
            print(f"    {p}: {c}")

    # --- Message length distribution ---
    print("\n=== RESPONSE LENGTH DISTRIBUTION ===")
    lens = []
    with open(DATA_DIR / "train_chat.jsonl") as f:
        for line in f:
            d = json.loads(line)
            assistant_msg = [m for m in d["messages"] if m["role"] == "assistant"][-1]["content"]
            lens.append(len(assistant_msg))
    lens.sort()
    print(f"  min: {min(lens)} chars")
    print(f"  median: {lens[len(lens)//2]} chars")
    print(f"  max: {max(lens)} chars")
    print(f"  mean: {sum(lens)//len(lens)} chars")

    # --- example_scores ---
    if (DATA_DIR / "example_scores.jsonl").exists():
        print("\n=== EXAMPLE SCORES ===")
        with open(DATA_DIR / "example_scores.jsonl") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                d = json.loads(line)
                print(f"  {json.dumps(d, indent=2)[:200]}")

    print("\nDone. This info drives the data balancing + format decision.")

if __name__ == "__main__":
    explore()
