#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["tqdm>=4.65"]
# ///
"""Generate creative CL macros by subprocessing into `claude -p` (Claude
Code headless mode). Validates each proposal in SBCL, keeps the survivors.

Why this instead of `_via_api.py`: you're already paying for Claude Code.
No separate ANTHROPIC_API_KEY, no separate billing line, same quality
output. The headless flag (`-p` / `--print`) makes a single non-interactive
call: prompt in, completion out, exit.

Strategy:
1. Load seed JSONL as in-context examples.
2. Loop:
   a. Pick a `focus_area` (resource-management, DSL, etc.) — rotates each
      batch so the corpus stays diverse beyond ~30 batches.
   b. `claude -p "<assembled prompt>" --output-format text` returns JSONL.
   c. Parse; validate each entry via `scripts/validate_creative_macros.py`
      (or inline SBCL).
   d. Append survivors to keepers JSONL; track names for dedup.
3. Stop when keepers ≥ target or proposal budget exhausted.

Resume: re-running with the same --output-file picks up where it left off,
skipping macro names already on disk.

Usage:
    uv run scripts/generate_creative_macros.py \\
        --seed-file scripts/seed_creative_macros.jsonl \\
        --output-file data/creative-macros-keepers.jsonl \\
        --rejects-file data/creative-macros-rejects.jsonl \\
        --target-keepers 1000 \\
        --batch-size 15 \\
        --max-proposals 4000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm


# ─── prompt construction ──────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are generating training data for fine-tuning a \
language model on Common Lisp macro generation. You produce high-quality, \
varied, SBCL-compileable macro examples in a strict JSON schema.

Hard requirements for EVERY entry you generate:
1. Output ONE JSON object per line (JSONL). No markdown fences, no prose, \
no explanations before/between/after objects.
2. Schema fields (all required):
   - "instruction": natural-language prompt describing what the macro does.
   - "input": a single sample call form, e.g. (my-macro arg ...)
   - "output": the reference (defmacro ...) source. Must compile in SBCL. \
     Use ONLY standard Common Lisp — no Alexandria, no Quicklisp, no \
     SB-EXT internals, no implementation-specific features.
   - "macroexpand": the EXACT expansion of `input` under the defmacro. \
     This is what (macroexpand-1 'input) returns after defining the macro. \
     Match SBCL's printing: uppercase symbols, gensyms as #:G123 (the \
     specific number is normalized by the validator), no extra whitespace.
   - "category": one of control-flow, anaphoric, capture-management, dsl, \
     efficiency, resource-management, validation, debugging.
   - "technique": one of gensym, let-binding, recursive-expansion, \
     once-only, closure-capture, anaphora, nested-backquote, dispatch.
   - "complexity": basic | intermediate | advanced.
   - "quality_score": always 1.0.
3. The macroexpand value MUST be correct. Trace the expansion mentally \
before writing. If you're unsure, write a SIMPLER macro you can verify.
4. Diversity matters more than cleverness. AVOID near-duplicates of seeds.
5. Use gensym for any binding the macro introduces that the user could \
shadow. Use once-only for any argument evaluated more than once.
6. Common pitfall: `internal-real-time` is NOT a function — it's \
`get-internal-real-time`. Don't invent function names.

You'll be asked for batches with a `focus_area` per batch. Stay within \
the focus area but exercise different patterns within it.
"""


FOCUS_AREAS = [
    ("resource-management",
     "macros that acquire and release resources safely: with-temp-file, "
     "with-mutex-held, with-pinned-buffer, with-rollback, with-redirected-output, "
     "with-temporary-directory, with-cwd, with-timer-thread, etc."),
    ("control-flow",
     "control-flow macros beyond the obvious if/when: case-eql, "
     "ecase-pattern, do-permutations, do-combinations, "
     "while/until/repeat, multiple-value-prog2, when-bind-all, "
     "unless-bind, eswhen, etc."),
    ("anaphoric",
     "anaphoric macros that bind a test result to `it`: aand, acond, "
     "alambda, aprog1, awhile, anaphoric variants of standard forms."),
    ("capture-management",
     "macros that demonstrate hygiene tricks: once-only patterns, "
     "with-gensyms abstractions, macros that introduce multiple bindings "
     "that the user must never see."),
    ("dsl",
     "tiny DSL building blocks: defrule, defstate, defpipeline, "
     "defparser, defview, defmessage, defroute, defcommand, "
     "define-validator, define-binary-type, defenum."),
    ("efficiency",
     "macros that improve runtime: with-cache, defmemo, declare-fast, "
     "inline-expand, with-inlined, with-array-views, "
     "specialize-on, with-stack-allocated."),
    ("validation",
     "macros that add invariants: ensure-type, assert-positive, "
     "with-guards, validate-args, defun-typed, with-preconditions, "
     "with-postconditions, define-contract."),
    ("debugging",
     "macros that aid debugging: with-trace, with-instrumentation, "
     "inline-test, dbg, log-call, with-stopwatch (already covered — "
     "do variants), explain-form, with-pretty-error."),
    ("iteration",
     "macros that extend iteration: dohash, doplist, do-tree, "
     "do-paired, dolist-collecting, dotimes-summing, "
     "dorange, do-with-window, do-batches."),
    ("binding",
     "macros that extend binding: let-when, let-while, let-typed, "
     "rebind, define-bindings, fluid-let, with-shadowed-symbols, "
     "lexical-rebind, with-defaults, destructure-as."),
    ("functional",
     "function-construction macros: defun/curried, lambda-pipe, "
     "compose-into, case-lambda, defun-with-fallback, "
     "with-memoized-fn, define-applicative."),
    ("metaprogramming",
     "macros that operate on or produce code: with-collected-forms, "
     "define-rewriter, with-shadowed-fn, walk-and-replace, "
     "define-syntax-extension, defmacro/curried."),
]


def build_prompt(seed_entries: list[dict], batch_size: int,
                 already_seen_names: set[str],
                 focus: tuple[str, str]) -> str:
    seed_block = "\n".join(json.dumps(e, ensure_ascii=False) for e in seed_entries)
    avoid = sorted(already_seen_names)
    avoid_block = ""
    if avoid:
        avoid_block = (
            "\n\nALREADY DONE (don't regenerate these names or trivial variants):\n"
            + ", ".join(avoid[:100])
        )
        if len(avoid) > 100:
            avoid_block += f"  (... +{len(avoid) - 100} more)"

    focus_name, focus_desc = focus
    return (
        SYSTEM_INSTRUCTION
        + "\n\n=== Reference examples (correct schema) ===\n"
        + seed_block
        + avoid_block
        + f"\n\n=== This batch's focus: {focus_name} ===\n"
        + focus_desc
        + f"\n\nGenerate {batch_size} new entries. JSONL only — one JSON object "
          "per line. Nothing else."
    )


# ─── SBCL validation (mirrors validate_creative_macros.py) ─────────────

VALIDATE_SCRIPT_TEMPLATE = """(in-package :cl-user)
(handler-case
    (progn
      {reference_macro}
      (let* ((input '{input_form})
             (actual (macroexpand-1 input)))
        (let ((*print-pretty* nil) (*print-readably* nil))
          (format t "~&__ACTUAL__~S~%" actual))))
  (error (c)
    (format t "~&__ERROR__~A~%" c)))
"""


def _gensym_normalize(s: str) -> str:
    s = re.sub(r"#:[A-Z][A-Z0-9-]*?\d+", "#:GENSYM", s)
    s = re.sub(r"\bG\d{2,}\b", "#:GENSYM", s)
    return re.sub(r"\s+", " ", s).strip()


def validate_entry(entry: dict, sbcl: str = "sbcl",
                   timeout: float = 10.0) -> tuple[bool, str]:
    script = VALIDATE_SCRIPT_TEMPLATE.format(
        reference_macro=entry["output"],
        input_form=entry["input"],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".lisp",
                                     delete=False, encoding="utf-8") as tf:
        tf.write(script)
        path = tf.name
    try:
        proc = subprocess.run([sbcl, "--script", path],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    stdout = proc.stdout or ""
    if "__ERROR__" in stdout:
        return False, "SBCL_ERROR: " + stdout.split("__ERROR__", 1)[1].splitlines()[0]
    if "__ACTUAL__" not in stdout:
        return False, "NO_OUTPUT"
    actual = stdout.split("__ACTUAL__", 1)[1].splitlines()[0].strip()
    expected = entry["macroexpand"].strip()
    if _gensym_normalize(actual) == _gensym_normalize(expected):
        return True, ""
    return False, (f"MISMATCH\n  expected: {expected[:240]}\n"
                   f"  actual:   {actual[:240]}")


# ─── claude -p invocation ─────────────────────────────────────────────

def call_claude_headless(prompt: str, timeout: float = 300.0,
                         claude_bin: str = "claude") -> tuple[str, dict]:
    """Invoke `claude -p` headless with --output-format json.

    Returns (response_text, metadata_dict). Metadata may include
    `total_cost_usd`, `usage` (token counts), `session_id`, etc., depending
    on the installed Claude Code version.
    """
    proc = subprocess.run(
        [claude_bin, "-p", prompt, "--output-format", "json"],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p exited {proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}"
        )

    # Claude Code's JSON envelope (current shape):
    #   { "type": "result", "subtype": "success", "result": "<text>",
    #     "is_error": false, "total_cost_usd": 0.0123, "usage": {...},
    #     "session_id": "...", "num_turns": 1 }
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        # Fall back to raw stdout if the wrapper format changed
        return proc.stdout, {"warning": f"JSON parse failed: {e.msg}"}

    if env.get("is_error"):
        raise RuntimeError(f"claude error: {env.get('result') or env}")

    text = env.get("result") or env.get("content") or ""
    meta = {k: v for k, v in env.items() if k != "result"}
    return text, meta


# ─── parsing ──────────────────────────────────────────────────────────

def parse_jsonl_response(text: str) -> tuple[list[dict], list[str]]:
    text = text.strip()
    # Strip code fences if claude wrapped output
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    entries: list[dict] = []
    errors: list[str] = []
    for ln_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {ln_no}: {e.msg} :: {line[:80]}")
            continue
        required = {"instruction", "input", "output", "macroexpand"}
        if not required.issubset(obj.keys()):
            errors.append(f"line {ln_no}: missing fields")
            continue
        entries.append(obj)
    return entries, errors


def macro_name_from_output(output: str) -> str | None:
    m = re.search(r"\(defmacro\s+([^\s(]+)", output, re.IGNORECASE)
    return m.group(1).lower() if m else None


# ─── main loop ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-file",      type=Path, required=True)
    ap.add_argument("--output-file",    type=Path, required=True)
    ap.add_argument("--rejects-file",   type=Path)
    ap.add_argument("--target-keepers", type=int, default=1000)
    ap.add_argument("--batch-size",     type=int, default=15)
    ap.add_argument("--max-proposals",  type=int, default=4000)
    ap.add_argument("--seeds-in-prompt", type=int, default=8)
    ap.add_argument("--claude-bin",     default="claude")
    ap.add_argument("--sbcl",           default="sbcl")
    ap.add_argument("--call-timeout",   type=float, default=300.0)
    args = ap.parse_args()

    if subprocess.run([args.claude_bin, "--version"], capture_output=True
                      ).returncode != 0:
        print(f"ERROR: `{args.claude_bin} --version` failed. Install Claude Code.",
              file=sys.stderr)
        return 1

    seeds = [json.loads(ln) for ln in args.seed_file.read_text().splitlines()
             if ln.strip()]
    print(f"Loaded {len(seeds)} seed entries")
    seen_names: set[str] = set()
    for e in seeds:
        n = macro_name_from_output(e["output"])
        if n:
            seen_names.add(n)
    base_seen = set(seen_names)  # snapshot — seeds, not progress

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing keepers
    if args.output_file.exists() and args.output_file.stat().st_size > 0:
        for line in args.output_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                n = macro_name_from_output(e["output"])
                if n:
                    seen_names.add(n)
            except json.JSONDecodeError:
                pass
        print(f"Resuming: {len(seen_names) - len(base_seen)} keepers on disk")

    out_f = args.output_file.open("a", encoding="utf-8")
    rej_f = args.rejects_file.open("a", encoding="utf-8") if args.rejects_file else None

    keepers_count = len(seen_names) - len(base_seen)
    proposals_count = 0
    in_context_seeds = seeds[: args.seeds_in_prompt]
    batch_idx = 0

    pbar = tqdm(total=args.target_keepers, initial=keepers_count, desc="keepers")
    while keepers_count < args.target_keepers and proposals_count < args.max_proposals:
        focus = FOCUS_AREAS[batch_idx % len(FOCUS_AREAS)]
        prompt = build_prompt(in_context_seeds, args.batch_size, seen_names, focus)
        batch_idx += 1

        try:
            response, meta = call_claude_headless(
                prompt, timeout=args.call_timeout, claude_bin=args.claude_bin,
            )
        except Exception as e:
            print(f"\nclaude -p error ({type(e).__name__}: {e}); sleeping 10s",
                  file=sys.stderr)
            time.sleep(10)
            continue

        proposals, parse_errors = parse_jsonl_response(response)
        proposals_count += len(proposals) + len(parse_errors)

        # Light cost/usage tracking if Claude Code reports it
        cost = meta.get("total_cost_usd")
        if cost is not None:
            running_cost = (getattr(main, "_running_cost", 0.0) or 0.0) + float(cost)
            setattr(main, "_running_cost", running_cost)

        for err in parse_errors:
            if rej_f:
                rej_f.write(json.dumps({"phase": "parse",
                                        "focus": focus[0],
                                        "error": err}) + "\n")

        for entry in proposals:
            name = macro_name_from_output(entry["output"])
            if name and name in seen_names:
                if rej_f:
                    rej_f.write(json.dumps({"phase": "dedup",
                                            "name": name,
                                            "entry": entry}) + "\n")
                continue
            ok, why = validate_entry(entry, sbcl=args.sbcl)
            if ok:
                entry.setdefault("source", f"claude-code:{focus[0]}")
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                if name:
                    seen_names.add(name)
                keepers_count += 1
                pbar.update(1)
                if keepers_count >= args.target_keepers:
                    break
            else:
                if rej_f:
                    rej_f.write(json.dumps({"phase": "validate",
                                            "focus": focus[0],
                                            "reason": why,
                                            "entry": entry}) + "\n")
                    rej_f.flush()

        postfix = {
            "proposals": proposals_count,
            "yield_pct": f"{100.0 * keepers_count / max(1, proposals_count):.1f}",
            "focus": focus[0][:12],
        }
        rc = getattr(main, "_running_cost", None)
        if rc is not None:
            postfix["$"] = f"{rc:.2f}"
        pbar.set_postfix(postfix)

    pbar.close()
    out_f.close()
    if rej_f:
        rej_f.close()

    print(f"\nDone. {keepers_count} keepers in {args.output_file}")
    print(f"      {proposals_count} total proposals "
          f"({100.0 * keepers_count / max(1, proposals_count):.1f}% yield)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
