#!/bin/bash
# Attach an alive_loop watcher to the running GRPO tmux session.
#
# Handles three cases automatically:
#   1. You are already attached to the 'grpo' tmux session
#      → split the current window vertically; alive_loop runs in the new pane.
#   2. You are in a plain SSH shell, the 'grpo' session exists detached
#      → add a pane to that session without forcing focus; reattach when ready.
#   3. tmux missing or session missing
#      → fall back to nohup background; tail alive.log to watch.
#
# After it runs, you can do:
#     tmux attach -t grpo        # see both panes
#     tail  -f grpo-output/full/alive.log
#
# Usage:
#     bash scripts/cloud/attach_alive.sh
#
# Env overrides:
#     SESSION         default 'grpo'
#     OUTPUT_DIR      default 'grpo-output/full'

set -euo pipefail

SESSION="${SESSION:-grpo}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-grpo-output/full}"

CMD="cd '$REPO_ROOT' && uv run scripts/cloud/alive_loop.py --output-dir '$OUTPUT_DIR'; echo; echo '[alive_loop exited — press enter to close]'; read"

# ─── tmux not available → background fallback ─────────────────────────
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not on PATH; starting alive_loop in the background instead"
    cd "$REPO_ROOT"
    nohup uv run scripts/cloud/alive_loop.py --output-dir "$OUTPUT_DIR" \
        > /dev/null 2>&1 &
    echo "  PID: $!"
    echo "  tail -f $REPO_ROOT/$OUTPUT_DIR/alive.log"
    exit 0
fi

# ─── case 1: we ARE attached to *some* tmux session ───────────────────
if [[ -n "${TMUX:-}" ]]; then
    echo "inside a tmux session — splitting current window"
    tmux split-window -v "$CMD"
    exit 0
fi

# ─── outside tmux — target the 'grpo' session if it exists ───────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "found tmux session '$SESSION'; adding a watcher pane"
    tmux split-window -t "$SESSION" -v -d "$CMD"
    echo
    echo "  attach with:   tmux attach -t $SESSION"
    echo "  or just watch: tail -f $REPO_ROOT/$OUTPUT_DIR/alive.log"
    exit 0
fi

# ─── session doesn't exist — create it just for the watcher ──────────
echo "no '$SESSION' session yet — starting one with alive_loop inside"
tmux new-session -d -s "$SESSION" "$CMD"
echo "  attach with:   tmux attach -t $SESSION"
