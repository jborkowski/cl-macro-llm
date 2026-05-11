#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Live kata-difficulty triage during training.

Watches the trainer's sample dumps (samples-step-*.jsonl) as they're
written and continuously re-classifies each kata into three buckets:

  • TOO_EASY      — mean reward ≥ 0.90 over ≥ 3 samples
                    → model already solves it; no learning signal here
  • UNREACHABLE   — mean reward ≤ -0.30 over ≥ 3 samples
                    → model can't get traction; kata may be broken
                       or beyond current ability
  • FRONTIER      — everything else with non-trivial variance
                    → these are doing the actual GRPO learning work

Every tick (default 60s) it writes a dashboard to OUTPUT_DIR/live_triage.md
AND emits one summary line per tick to stdout so the supervisor's
Monitor can catch it. It also emits one-line alerts when a kata FLIPS
into TOO_EASY or UNREACHABLE so the operator can react (e.g., inject
harder/easier katas).

Read-only — never modifies the trainer or its dataset. Safe to run for
the whole duration of a GRPO run.

Usage on the pod (drop into a tmux pane or nohup):
    uv run scripts/cloud/live_triage.py
    INTERVAL=30 EASY_THRESHOLD=0.85 uv run scripts/cloud/live_triage.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ─── env knobs (mirrors grpo_train.py style) ──────────────────────────

def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


DEFAULT_OUTPUT_DIR = Path(_env("OUTPUT_DIR", "grpo-output/full"))
DEFAULT_INTERVAL   = int(_env("INTERVAL", "60"))
EASY_THRESHOLD     = float(_env("EASY_THRESHOLD", "0.90"))
HARD_THRESHOLD     = float(_env("HARD_THRESHOLD", "-0.30"))
MIN_SAMPLES        = int(_env("MIN_SAMPLES", "3"))
TOP_PER_BUCKET     = int(_env("TOP_PER_BUCKET", "10"))


# ─── ingest ──────────────────────────────────────────────────────────

def _iter_samples(samples_dir: Path):
    """Yield parsed JSON lines from every samples-step-*.jsonl in dir."""
    for f in sorted(samples_dir.glob("samples-step-*.jsonl")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def _kata_id_of(sample: dict) -> str | None:
    kid = sample.get("kata_id")
    if kid:
        return str(kid)
    # backward-compat: older dumps had kata_path
    kp = sample.get("kata_path")
    if kp:
        return Path(str(kp)).name
    return None


def _aggregate(samples_dir: Path) -> dict[str, dict]:
    """Build per-kata stats from every sample on disk."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in _iter_samples(samples_dir):
        kid = _kata_id_of(s)
        if kid is None:
            continue
        buckets[kid].append(s)

    stats: dict[str, dict] = {}
    for kid, rows in buckets.items():
        rewards = [float(r.get("reward", 0.0)) for r in rows]
        if len(rewards) < MIN_SAMPLES:
            continue
        try:
            mean = statistics.fmean(rewards)
        except statistics.StatisticsError:
            continue
        stdev = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        # Last sample we saw → diagnostic for what the model just produced
        last = rows[-1]
        stats[kid] = {
            "kata_id":   kid,
            "n":         len(rewards),
            "mean":      mean,
            "stdev":     stdev,
            "min":       min(rewards),
            "max":       max(rewards),
            "frac_pos":  sum(1 for r in rewards if r > 0) / len(rewards),
            "frac_full": sum(1 for r in rewards if r >= 0.99) / len(rewards),
            "frac_neg":  sum(1 for r in rewards if r < 0) / len(rewards),
            "last_step": last.get("step"),
            "last_reward": float(last.get("reward", 0.0)),
            "last_ext":  (last.get("defmacro_extracted") or "")[:200],
            "category":  last.get("category") or "?",
        }
    return stats


# ─── classify ────────────────────────────────────────────────────────

def _classify(s: dict) -> str:
    if s["mean"] >= EASY_THRESHOLD:
        return "TOO_EASY"
    if s["mean"] <= HARD_THRESHOLD:
        return "UNREACHABLE"
    return "FRONTIER"


# ─── dashboard render ────────────────────────────────────────────────

def _bar(mean: float, width: int = 20) -> str:
    """Reward in [-0.5, 1.5] → bar [0, width]."""
    pos = max(0, min(width, int(round((mean + 0.5) / 2.0 * width))))
    return "█" * pos + "·" * (width - pos)


def _render_dashboard(stats: dict[str, dict], n_dumps: int) -> str:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for s in stats.values():
        by_bucket[_classify(s)].append(s)

    n_easy  = len(by_bucket["TOO_EASY"])
    n_hard  = len(by_bucket["UNREACHABLE"])
    n_front = len(by_bucket["FRONTIER"])
    n_total = sum(s["n"] for s in stats.values())
    overall_mean = (statistics.fmean([s["mean"] for s in stats.values()])
                    if stats else 0.0)

    lines = []
    lines.append(f"# live_triage — {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append(f"- katas observed: **{len(stats)}** "
                 f"({n_easy} TOO_EASY · {n_front} FRONTIER · {n_hard} UNREACHABLE)")
    lines.append(f"- total rollouts: {n_total}")
    lines.append(f"- sample dumps:   {n_dumps}")
    lines.append(f"- overall mean:   {overall_mean:+.4f}")
    lines.append(f"- thresholds:     easy ≥ {EASY_THRESHOLD:+.2f} · "
                 f"hard ≤ {HARD_THRESHOLD:+.2f} · min_samples = {MIN_SAMPLES}")
    lines.append("")

    def _section(title: str, items: list[dict], sort_key, *, suggest: str):
        lines.append(f"## {title}  ({len(items)})")
        if not items:
            lines.append("_(none)_")
            lines.append("")
            return
        lines.append(suggest)
        lines.append("")
        lines.append("| kata_id | n | mean | stdev | last step | bar |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for s in sorted(items, key=sort_key)[:TOP_PER_BUCKET]:
            lines.append(f"| `{s['kata_id']}` | {s['n']} | "
                         f"{s['mean']:+.3f} | {s['stdev']:.3f} | "
                         f"{s['last_step']} | `{_bar(s['mean'])}` |")
        if len(items) > TOP_PER_BUCKET:
            lines.append(f"_…and {len(items) - TOP_PER_BUCKET} more_")
        lines.append("")

    _section(
        "🎯 FRONTIER — where learning is actually happening",
        by_bucket["FRONTIER"],
        sort_key=lambda s: -s["stdev"],  # highest variance first
        suggest="_Keep these — they're doing the work. High-stdev rows are at "
                "the active learning edge._",
    )
    _section(
        "✅ TOO_EASY — model already solves, drag on training mix",
        by_bucket["TOO_EASY"],
        sort_key=lambda s: -s["mean"],
        suggest="_Consider replacing with harder variants or dropping for the "
                "next run._",
    )
    _section(
        "❌ UNREACHABLE — model can't get traction",
        by_bucket["UNREACHABLE"],
        sort_key=lambda s: s["mean"],
        suggest="_Check kata correctness, or inject scaffolded/easier "
                "versions._",
    )
    return "\n".join(lines)


# ─── main loop ───────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"trainer output dir (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--interval",   type=int, default=DEFAULT_INTERVAL,
                    help=f"seconds between ticks (default: {DEFAULT_INTERVAL})")
    ap.add_argument("--once", action="store_true",
                    help="one tick + exit")
    args = ap.parse_args()

    out  = Path(args.output_dir)
    dash = out / "live_triage.md"

    out.mkdir(parents=True, exist_ok=True)

    print(f"# live_triage watching {out} every {args.interval}s")
    print(f"# dashboard: {dash}")

    # Track previously-seen classifications to emit transition alerts
    last_class: dict[str, str] = {}

    stop = {"v": False}
    def _sig(_signum, _frame):
        stop["v"] = True
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    while not stop["v"]:
        try:
            stats = _aggregate(out)
            n_dumps = len(list(out.glob("samples-step-*.jsonl")))

            # Emit transition alerts (one line each)
            for kid, s in stats.items():
                cls = _classify(s)
                prev = last_class.get(kid)
                if prev != cls:
                    if prev is not None:
                        # ALERT lines are easy to grep for in supervisor logs
                        print(f"[{datetime.now():%H:%M:%S}] ALERT "
                              f"{kid}: {prev} -> {cls} "
                              f"(mean={s['mean']:+.3f} n={s['n']})",
                              flush=True)
                    last_class[kid] = cls

            # Tick summary
            n_easy = sum(1 for s in stats.values()
                         if _classify(s) == "TOO_EASY")
            n_hard = sum(1 for s in stats.values()
                         if _classify(s) == "UNREACHABLE")
            n_fr   = sum(1 for s in stats.values()
                         if _classify(s) == "FRONTIER")
            overall = (statistics.fmean([s["mean"] for s in stats.values()])
                       if stats else 0.0)
            print(f"[{datetime.now():%H:%M:%S}] triage "
                  f"katas={len(stats)} "
                  f"frontier={n_fr} easy={n_easy} unreachable={n_hard} "
                  f"overall_mean={overall:+.4f} dumps={n_dumps}",
                  flush=True)

            # Write markdown dashboard atomically
            tmp = dash.with_suffix(".md.tmp")
            tmp.write_text(_render_dashboard(stats, n_dumps), encoding="utf-8")
            tmp.replace(dash)

        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] tick error: "
                  f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)

        if args.once:
            break

        # Short-sleep loop so SIGINT fires promptly
        slept = 0
        while slept < args.interval and not stop["v"]:
            time.sleep(min(1, args.interval - slept))
            slept += 1

    print(f"# live_triage exiting cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
