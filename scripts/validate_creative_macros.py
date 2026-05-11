#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["tqdm>=4.65"]
# ///
"""Validate a JSONL of proposed Common Lisp macros against SBCL.

For each entry:
    1. Define the proposed `output` (a defmacro form) in a fresh SBCL.
    2. macroexpand-1 the `input` form.
    3. Compare actual expansion to the proposed `macroexpand` (gensym-normalized).

Survivors → keepers JSONL. Failures → rejects JSONL with diagnostics.

This is the validator-only complement to `generate_creative_macros.py`.
When macros are produced directly by Claude Code (or hand-written),
just pipe them through here.

Usage:
    uv run scripts/validate_creative_macros.py \\
        --input  data/creative-macros-proposals.jsonl \\
        --output data/creative-macros-keepers.jsonl \\
        --rejects data/creative-macros-rejects.jsonl \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm


VALIDATE_SCRIPT_TEMPLATE = """(in-package :cl-user)
(handler-case
    (progn
      {reference_macro}
      (let* ((input '{input_form})
             (actual (macroexpand-1 input)))
        (let ((*print-pretty* nil)
              (*print-readably* nil))
          (format t "~&__ACTUAL__~S~%" actual))))
  (error (c)
    (format t "~&__ERROR__~A~%" c)))
"""


def _gensym_normalize(s: str) -> str:
    """Canonicalize gensym-looking tokens so structural compare works
    regardless of SBCL's internal counter."""
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
        proc = subprocess.run(
            [sbcl, "--script", path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    stdout = proc.stdout or ""
    if "__ERROR__" in stdout:
        err = stdout.split("__ERROR__", 1)[1].splitlines()[0]
        return False, f"SBCL_ERROR: {err}"
    if "__ACTUAL__" not in stdout:
        return False, f"NO_OUTPUT (stderr: {(proc.stderr or '')[:200]})"
    actual = stdout.split("__ACTUAL__", 1)[1].splitlines()[0].strip()
    expected = entry["macroexpand"].strip()
    if _gensym_normalize(actual) == _gensym_normalize(expected):
        return True, ""
    return False, (f"MISMATCH\n  expected: {expected[:240]}\n"
                   f"  actual:   {actual[:240]}")


def macro_name_from_output(output: str) -> str | None:
    m = re.search(r"\(defmacro\s+([^\s(]+)", output, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _worker(args):
    entry, idx, sbcl = args
    ok, why = validate_entry(entry, sbcl=sbcl)
    return idx, ok, why, entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",    type=Path, required=True,
                    help="Proposals JSONL (one macro entry per line)")
    ap.add_argument("--output",   type=Path, required=True,
                    help="Keepers JSONL — appended to if exists (dedup by macro name)")
    ap.add_argument("--rejects",  type=Path,
                    help="Rejects JSONL (with diagnostics)")
    ap.add_argument("--workers",  type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--sbcl",     default="sbcl")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: --input {args.input} not found", file=sys.stderr)
        return 1

    # Load proposals
    proposals: list[dict] = []
    parse_errors: list[str] = []
    for ln_no, line in enumerate(args.input.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            parse_errors.append(f"line {ln_no}: {e.msg}")
            continue
        required = {"instruction", "input", "output", "macroexpand"}
        missing = required - set(obj.keys())
        if missing:
            parse_errors.append(f"line {ln_no}: missing {sorted(missing)}")
            continue
        proposals.append(obj)
    print(f"Loaded {len(proposals)} proposals "
          f"({len(parse_errors)} malformed lines skipped)")

    # Dedup against existing keepers
    seen_names: set[str] = set()
    if args.output.exists() and args.output.stat().st_size > 0:
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                n = macro_name_from_output(e["output"])
                if n:
                    seen_names.add(n)
            except json.JSONDecodeError:
                pass
        print(f"  {len(seen_names)} existing keepers — dedup'ing against them")

    proposals_unique = []
    dups = 0
    for p in proposals:
        name = macro_name_from_output(p["output"])
        if name and name in seen_names:
            dups += 1
            continue
        proposals_unique.append(p)
        if name:
            seen_names.add(name)
    print(f"  {dups} duplicates removed; validating {len(proposals_unique)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("a", encoding="utf-8")
    rej_f = args.rejects.open("a", encoding="utf-8") if args.rejects else None

    if rej_f and parse_errors:
        for err in parse_errors:
            rej_f.write(json.dumps({"phase": "parse", "error": err}) + "\n")

    worker_args = [(e, i, args.sbcl) for i, e in enumerate(proposals_unique)]
    n_ok = n_bad = 0
    with mp.Pool(args.workers) as pool:
        for _idx, ok, why, entry in tqdm(
            pool.imap_unordered(_worker, worker_args),
            total=len(worker_args),
            desc="SBCL",
        ):
            if ok:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                if rej_f:
                    rej_f.write(json.dumps(
                        {"phase": "validate", "reason": why, "entry": entry}
                    ) + "\n")
                n_bad += 1

    out_f.close()
    if rej_f:
        rej_f.close()

    print(f"\nValidation complete:")
    print(f"  keepers:  {n_ok:>5d}")
    print(f"  rejected: {n_bad:>5d}")
    print(f"  yield:    {100.0 * n_ok / max(1, n_ok + n_bad):.1f}%")
    print(f"  total keepers on disk now: {len(seen_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
