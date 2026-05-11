#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Rank katas by GRPO performance from the trainer's sample dumps.

Reads `samples-step-*.jsonl` files written by the reward function and
groups by kata_path. For each kata, computes mean reward, sample count,
and shows one representative completion. Sorted ascending so the worst
katas float to the top — those are the ones the model is failing on
repeatedly and worth manual inspection.

Usage:
    uv run scripts/cloud/triage_katas.py \\
        --samples-dir grpo-output/full \\
        --top 10

The expected JSONL line shape (one per rollout):
    {"step":int, "kata_path":str, "completion":str,
     "reward":float, "defmacro_extracted":str|null, ...}

This is a read-only post-hoc tool — safe to run while training is live.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _iter_samples(samples_dir: Path, since_step: int | None = None):
    files = sorted(samples_dir.glob("samples-step-*.jsonl"))
    for f in files:
        try:
            step = int(f.stem.split("-")[-1])
        except ValueError:
            step = None
        if since_step is not None and step is not None and step < since_step:
            continue
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", type=Path, required=True,
                    help="Directory containing samples-step-NNN.jsonl files")
    ap.add_argument("--top",         type=int, default=10,
                    help="How many worst-reward katas to print")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="Ignore katas with fewer than this many dumped rollouts")
    ap.add_argument("--since-step",  type=int, default=None,
                    help="Only count dumps from this step onward")
    ap.add_argument("--show-completion", action="store_true",
                    help="Print the worst completion in full for each kata")
    args = ap.parse_args()

    if not args.samples_dir.exists():
        print(f"ERROR: --samples-dir {args.samples_dir} not found", file=sys.stderr)
        return 1

    by_kata: dict[str, list[dict]] = defaultdict(list)
    for s in _iter_samples(args.samples_dir, since_step=args.since_step):
        kp = s.get("kata_path") or "<unknown>"
        by_kata[kp].append(s)

    if not by_kata:
        print(f"  no sample dumps found under {args.samples_dir}/samples-step-*.jsonl")
        return 0

    stats = []
    for kp, samples in by_kata.items():
        rewards = [float(s.get("reward", 0.0)) for s in samples]
        if len(rewards) < args.min_samples:
            continue
        mean = statistics.fmean(rewards)
        worst = min(samples, key=lambda s: float(s.get("reward", 0.0)))
        stats.append({
            "kata":      Path(kp).name,
            "kata_path": kp,
            "n":         len(rewards),
            "mean":      mean,
            "min":       min(rewards),
            "max":       max(rewards),
            "frac_neg":  sum(1 for r in rewards if r < 0) / len(rewards),
            "worst":     worst,
        })

    stats.sort(key=lambda r: r["mean"])

    n_files = len(list(args.samples_dir.glob("samples-step-*.jsonl")))
    print(f"\n=== triage: {len(stats)} katas with >= {args.min_samples} "
          f"samples across {n_files} dump files ===\n")

    for i, s in enumerate(stats[:args.top]):
        bar_w = 24
        # map mean reward in [-0.5, 1.5] to a bar in [0, bar_w]
        pos = max(0, min(bar_w, int(round((s["mean"] + 0.5) / 2.0 * bar_w))))
        bar = "█" * pos + "·" * (bar_w - pos)
        print(f"  {i + 1:>2d}. {s['kata']:<32s}  "
              f"mean={s['mean']:+.3f}  n={s['n']:>3d}  "
              f"min={s['min']:+.2f}  neg={s['frac_neg']:.0%}  [{bar}]")
        if args.show_completion:
            w = s["worst"]
            ext = (w.get("defmacro_extracted") or "")[:240]
            full = (w.get("completion") or "")[:240].replace("\n", " ")
            print(f"        kata_path: {s['kata_path']}")
            print(f"        step:      {w.get('step')}")
            print(f"        extracted: {ext}")
            print(f"        raw:       {full}")
            print()

    if not args.show_completion and stats[:args.top]:
        print("\n  (rerun with --show-completion to see what the model produced)")

    print()
    print(f"  overall mean reward (across {sum(s['n'] for s in stats)} samples): "
          f"{statistics.fmean([r for s in stats for r in [s['mean']] * s['n']]):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
