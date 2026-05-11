#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Liveness watcher for the GRPO trainer. Run in a tmux pane alongside it.

Each tick appends one line to OUTPUT_DIR/alive.log AND prints it, so the
tmux pane shows the live status and you can also `grep` history later.

What it checks each tick:
  • is grpo_train.py still running? (pgrep)
  • metrics.jsonl mtime — older than STALL_AFTER seconds = STALLED
  • last logged step + loss + reward + kl from metrics.jsonl
  • how many checkpoint-* dirs landed
  • how many sample-step-*.jsonl files written
  • GPU utilization + memory (nvidia-smi)

Exits cleanly on Ctrl-C or SIGTERM. Read-only — never touches training
state. Safe to start/stop any time.

Usage:
    uv run scripts/cloud/alive_loop.py                       # defaults
    INTERVAL=30 STALL_AFTER=600 \\
      uv run scripts/cloud/alive_loop.py --output-dir grpo-output/full
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ─── helpers ──────────────────────────────────────────────────────────

def _running_trainer() -> int | None:
    """Return PID of grpo_train.py if running, else None."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "grpo_train.py"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.split()[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def _gpu_snapshot() -> str:
    """One-line GPU summary or '?' if nvidia-smi missing."""
    if not shutil.which("nvidia-smi"):
        return "no-nvidia-smi"
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        # one card → "12, 34567, 81920"
        rows = [r.strip() for r in out.stdout.splitlines() if r.strip()]
        if not rows:
            return "no-rows"
        parts = []
        for i, r in enumerate(rows):
            util, mem_u, mem_t = [x.strip() for x in r.split(",")]
            parts.append(f"gpu{i}={util}%/{mem_u}MB/{mem_t}MB")
        return " ".join(parts)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return "err"


def _last_metrics_line(path: Path) -> dict | None:
    """Read the last JSON line from metrics.jsonl (tail), parse it."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    # Cheap tail: read last ~4 KB.
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        f.seek(max(0, end - 4096))
        tail = f.read().decode("utf-8", errors="replace")
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _fmt_metric(d: dict | None, key: str, fmt: str = "{:+.4f}") -> str:
    if not d or key not in d:
        return "-"
    try:
        return fmt.format(float(d[key]))
    except (TypeError, ValueError):
        return str(d[key])


def _human_age(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# ─── tick ─────────────────────────────────────────────────────────────

def tick(args, log_f) -> None:
    ts  = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.output_dir)
    mp  = out / "metrics.jsonl"

    pid     = _running_trainer()
    proc    = f"PID={pid}" if pid else "DEAD"

    last    = _last_metrics_line(mp)
    step    = last.get("step", "-") if last else "-"
    loss    = _fmt_metric(last, "loss")
    reward  = _fmt_metric(last, "reward")
    kl      = _fmt_metric(last, "kl")
    if reward == "-":
        reward = _fmt_metric(last, "rewards/macro_gym_reward")

    metrics_age_s: float | None = None
    if mp.exists():
        metrics_age_s = time.time() - mp.stat().st_mtime
    age_str = _human_age(metrics_age_s) if metrics_age_s is not None else "no-metrics"

    n_ckpts   = len(list(out.glob("checkpoint-*"))) if out.exists() else 0
    n_samples = len(list(out.glob("samples-step-*.jsonl"))) if out.exists() else 0

    gpu = _gpu_snapshot()

    # Status verdict
    if pid is None:
        status = "DEAD"
    elif metrics_age_s is not None and metrics_age_s > args.stall_after:
        status = f"STALLED({age_str})"
    elif metrics_age_s is None:
        status = "STARTING"   # trainer up, no metrics yet
    else:
        status = "OK"

    line = (
        f"[{ts}] {status:<14}  {proc:<10}  step={step:<5}  "
        f"loss={loss}  reward={reward}  kl={kl}  "
        f"ckpts={n_ckpts}  samples={n_samples}  "
        f"metrics_age={age_str}  {gpu}"
    )
    print(line, flush=True)
    if log_f is not None:
        log_f.write(line + "\n")
        log_f.flush()


# ─── main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path,
                    default=Path(os.environ.get("OUTPUT_DIR", "grpo-output/full")))
    ap.add_argument("--interval",   type=int,
                    default=int(os.environ.get("INTERVAL", "30")),
                    help="seconds between ticks")
    ap.add_argument("--stall-after", type=int,
                    default=int(os.environ.get("STALL_AFTER", "600")),
                    help="metrics.jsonl mtime older than this = STALLED")
    ap.add_argument("--once", action="store_true",
                    help="One tick + exit (useful for cron / single check)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "alive.log"
    log_f = log_path.open("a", encoding="utf-8")

    print(f"# alive_loop watching {args.output_dir} every {args.interval}s "
          f"(stall threshold {args.stall_after}s); log: {log_path}",
          flush=True)

    stop = {"v": False}
    def _sig(_signum, _frame):
        stop["v"] = True
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        while not stop["v"]:
            tick(args, log_f)
            if args.once:
                break
            # Sleep in small chunks so signals fire promptly.
            slept = 0
            while slept < args.interval and not stop["v"]:
                time.sleep(min(1, args.interval - slept))
                slept += 1
    finally:
        log_f.close()
    print("# alive_loop exiting cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
