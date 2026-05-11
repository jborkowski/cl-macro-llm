#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic>=0.40",
#     "tqdm>=4.65",
# ]
# ///
"""Generate creative CL macros via Claude API, validate each through SBCL,
keep only those that pass.

Strategy:
1. Load the hand-curated seed JSONL (~10 high-quality examples)
2. Ask Claude to produce N more in the same schema, with style/diversity
   constraints in the prompt. Rotate a `focus_area` per batch (resource
   management, DSLs, anaphoric, etc.) to keep the corpus diverse across
   1000+ proposals — without rotation the model converges on its
   favorite patterns by ~batch 30.
3. For each proposal:
   - Parse JSON (drop malformed entries)
   - Run the defmacro + (macroexpand-1 input) in SBCL
   - Compare the actual expansion to the proposed `macroexpand` field
   - Dedup by macro name (track names across the whole session)
4. Survivors → keepers JSONL. Failures → rejects JSONL with diagnostics.

Iterate until we hit the target keeper count or burn the proposal budget.
Resume support: re-running with the same --output-file appends, skips
any macro names already on disk.

Env:
    ANTHROPIC_API_KEY    required

Usage:
    uv run scripts/generate_creative_macros.py \\
        --seed-file scripts/seed_creative_macros.jsonl \\
        --output-file data/creative-macros-keepers.jsonl \\
        --rejects-file data/creative-macros-rejects.jsonl \\
        --target-keepers 1000 \\
        --batch-size 12 \\
        --max-proposals 4000 \\
        --model claude-sonnet-4-6

Cost estimate (per 1000 keepers, ~30% yield → ~3500 proposals → ~300 API calls):
    claude-opus-4-7    ~$120  (best quality CL, smartest macros)
    claude-sonnet-4-6  ~$25   (sweet spot for this task — recommended)
    claude-haiku-4-5   ~$5    (cheapest, more validation failures, more dups)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from anthropic import Anthropic
from tqdm import tqdm


# ─── prompt construction ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are generating training data for fine-tuning a language \
model on Common Lisp macro generation. You produce high-quality, varied, \
SBCL-compileable macro examples in a strict JSON schema.

Hard requirements for EVERY entry you generate:
1. Output ONE JSON object per line (JSONL). No markdown fences, no prose, \
no explanations between objects.
2. Schema fields:
   - "instruction": natural-language description of what the macro does. \
Write it as a prompt an agent would receive.
   - "input": a single sample call form, e.g. (my-macro arg ...)
   - "output": the reference (defmacro ...) source. Must compile in SBCL. \
Use ONLY standard Common Lisp — no Alexandria, no Quicklisp, no SB-EXT \
internals, no implementation-specific features.
   - "macroexpand": the EXACT expansion of `input` under the defmacro. \
This is what (macroexpand-1 'input) returns after the defmacro is \
defined. Match SBCL's printing conventions: uppercase symbols, gensyms \
as #:G123 (the specific number doesn't have to match; the validator \
normalizes), no extra whitespace inside expressions.
   - "category": one of control-flow, anaphoric, capture-management, \
dsl, efficiency, resource-management, validation, debugging.
   - "technique": one of gensym, let-binding, recursive-expansion, \
once-only, closure-capture, anaphora, nested-backquote, dispatch.
   - "complexity": basic | intermediate | advanced.
   - "quality_score": always 1.0 (these are hand-curated quality).
3. The macroexpand value MUST be correct. If unsure, write a simpler macro.
4. Diversity matters more than cleverness. AVOID near-duplicates of the \
seed examples — invent fresh patterns.
5. Use gensym for any binding the macro introduces that the user might \
shadow. Use once-only semantics for any argument that gets evaluated \
more than once in the expansion.
6. Don't reference functions that don't exist in standard CL. Common \
pitfall: `internal-real-time` is NOT a function — it's `get-internal-real-time`.

Examples you should NOT generate (already covered): when-let, awhen, aif, \
unless-let, dohash, ->, ->>, with-stopwatch, with-collected, \
dotimes-collecting, defmemo, with-retry, with-temp-file, with-mutex-held.

Aim for: less-common but useful macros. Examples of GOOD targets: \
with-output-to-string variants, do-permutations, with-pinned-buffer, \
defun/curried, define-validator, with-restored, await-all, \
case-lambda, with-instrumentation, define-binary-type, \
multiple-value-prog2, with-shadowed-symbols, defcacheable, etc.

You'll be asked to generate batches. Each batch should contain DISTINCT \
ideas (no duplicates within a batch). Across batches the user tracks \
duplicates and asks you for more, so don't fixate on a single style."""


def build_user_prompt(seed_entries: list[dict], batch_size: int,
                      already_seen_names: set[str]) -> str:
    """Construct the user prompt with seeds as in-context examples."""
    seed_block = "\n".join(json.dumps(e, ensure_ascii=False) for e in seed_entries)
    avoid = sorted(already_seen_names)
    avoid_block = ""
    if avoid:
        avoid_block = (
            "\n\nMacros ALREADY in the corpus — do NOT generate these names or "
            "trivial variants of them:\n" + ", ".join(avoid[:80])
        )
        if len(avoid) > 80:
            avoid_block += f"  (… and {len(avoid) - 80} more)"

    return (
        "Here are example entries in the required schema:\n\n"
        + seed_block
        + avoid_block
        + f"\n\nNow generate {batch_size} NEW entries. JSONL only, one object "
          f"per line, no other output."
    )


# ─── SBCL validation ──────────────────────────────────────────────────

VALIDATE_SCRIPT_TEMPLATE = """(in-package :cl-user)
(handler-case
    (progn
      {reference_macro}
      (let* ((input '{input_form})
             (actual (macroexpand-1 input)))
        ;; Print as a single line; the runner parses this.
        (let ((*print-pretty* nil)
              (*print-readably* nil))
          (format t "~&__ACTUAL__~S~%" actual))))
  (error (c)
    (format t "~&__ERROR__~A~%" c)))
"""


def _gensym_normalize(s: str) -> str:
    """Replace gensym-looking symbols (#:G123, #:FOO456) and bare uninterned
    symbols like G123 with a canonical token so structural comparison
    works regardless of which counter SBCL is on."""
    import re
    s = re.sub(r"#:[A-Z][A-Z0-9-]*?(\d+)", "#:GENSYM", s)
    s = re.sub(r"\bG\d+\b", "#:GENSYM", s)
    return s


def validate_entry(entry: dict, sbcl: str = "sbcl", timeout: float = 10.0
                   ) -> tuple[bool, str]:
    """Run the entry's defmacro in SBCL, macroexpand-1 its input, compare to
    the proposed `macroexpand` field."""
    script = VALIDATE_SCRIPT_TEMPLATE.format(
        reference_macro=entry["output"],
        input_form=entry["input"],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".lisp", delete=False,
                                     encoding="utf-8") as tf:
        tf.write(script)
        script_path = tf.name
    try:
        proc = subprocess.run(
            [sbcl, "--script", script_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if "__ERROR__" in stdout:
        err = stdout.split("__ERROR__", 1)[1].splitlines()[0]
        return False, f"SBCL_ERROR: {err}"

    if "__ACTUAL__" not in stdout:
        return False, f"NO_OUTPUT\nstdout:{stdout[:200]}\nstderr:{stderr[:200]}"

    actual = stdout.split("__ACTUAL__", 1)[1].splitlines()[0].strip()
    expected = entry["macroexpand"].strip()
    if _gensym_normalize(actual) == _gensym_normalize(expected):
        return True, ""
    return False, f"MISMATCH\n  expected: {expected[:200]}\n  actual:   {actual[:200]}"


# ─── parsing LLM output ───────────────────────────────────────────────

def parse_jsonl_response(text: str) -> tuple[list[dict], list[str]]:
    """Lenient JSONL parser. Strip code fences if present. Return
    (entries, errors_per_line)."""
    text = text.strip()
    # Drop a leading/trailing ```...``` block if the model wrapped output
    if text.startswith("```"):
        # remove first line if it's a fence, last line if it's a fence
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
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {ln_no}: {e.msg} :: {line[:80]}")
            continue
        # Quick schema sanity
        required = {"instruction", "input", "output", "macroexpand"}
        missing = required - set(obj.keys())
        if missing:
            errors.append(f"line {ln_no}: missing fields {sorted(missing)}")
            continue
        entries.append(obj)
    return entries, errors


def macro_name_from_output(output: str) -> str | None:
    import re
    m = re.search(r"\(defmacro\s+([^\s(]+)", output, re.IGNORECASE)
    return m.group(1).lower() if m else None


# ─── main loop ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-file",     type=Path, required=True)
    ap.add_argument("--output-file",   type=Path, required=True)
    ap.add_argument("--rejects-file",  type=Path)
    ap.add_argument("--target-keepers", type=int, default=150)
    ap.add_argument("--batch-size",    type=int, default=10)
    ap.add_argument("--max-proposals", type=int, default=600)
    ap.add_argument("--model",         default="claude-opus-4-7")
    ap.add_argument("--max-tokens",    type=int, default=8000)
    ap.add_argument("--seeds-in-prompt", type=int, default=8,
                    help="How many seed examples to include as in-context references")
    ap.add_argument("--sbcl",          default="sbcl")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    seeds = [json.loads(ln) for ln in args.seed_file.read_text().splitlines() if ln.strip()]
    print(f"Loaded {len(seeds)} seed entries")
    seen_names: set[str] = set()
    for e in seeds:
        name = macro_name_from_output(e["output"])
        if name:
            seen_names.add(name)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output_file.open("a", encoding="utf-8")
    rej_f = args.rejects_file.open("a", encoding="utf-8") if args.rejects_file else None

    # Resume support: read existing output, accumulate names
    if args.output_file.exists() and args.output_file.stat().st_size > 0:
        for line in args.output_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                name = macro_name_from_output(e["output"])
                if name:
                    seen_names.add(name)
            except json.JSONDecodeError:
                pass
        print(f"Resuming: {len(seen_names) - len(seeds)} keepers already on disk")

    client = Anthropic(api_key=api_key)

    keepers_count = max(0, len(seen_names) - len(seeds))
    proposals_count = 0
    in_context_seeds = seeds[: args.seeds_in_prompt]

    pbar = tqdm(total=args.target_keepers, initial=keepers_count, desc="keepers")
    while keepers_count < args.target_keepers and proposals_count < args.max_proposals:
        prompt = build_user_prompt(in_context_seeds, args.batch_size, seen_names)
        try:
            msg = client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"\nAPI error: {type(e).__name__}: {e}; sleeping 30s and retrying")
            time.sleep(30)
            continue

        text = "".join(block.text for block in msg.content if block.type == "text")
        proposals, parse_errors = parse_jsonl_response(text)
        proposals_count += len(proposals) + len(parse_errors)

        for err in parse_errors:
            if rej_f:
                rej_f.write(json.dumps({"phase": "parse", "error": err}) + "\n")

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
                                            "reason": why,
                                            "entry": entry}) + "\n")
                    rej_f.flush()

        # Light token accounting
        pbar.set_postfix(
            proposals=proposals_count,
            yield_pct=f"{100.0 * keepers_count / max(1, proposals_count):.1f}",
        )

    pbar.close()
    out_f.close()
    if rej_f:
        rej_f.close()

    print(f"\nDone. {keepers_count} keepers in {args.output_file}")
    print(f"      {proposals_count} total proposals "
          f"({100.0 * keepers_count / max(1, proposals_count):.1f}% yield)")
    if args.rejects_file:
        print(f"      rejects in {args.rejects_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
