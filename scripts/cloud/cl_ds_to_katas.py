#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=2.20",
#     "huggingface_hub>=0.24",
#     "tqdm>=4.65",
# ]
# ///
"""Convert j14i/cl-ds rows into macro-gym kata directories.

cl-ds schema (verified against the live dataset):
    input        the macro CALL form, e.g. (make-name-transformer name "-ZORK")
    output       the reference DEFMACRO source (what an agent should produce)
    macroexpand  the expected MACROEXPAND-1 result of input under that defmacro
    category, technique, complexity, quality_score — metadata

Each cl-ds row becomes one kata at <output_dir>/cl-ds-{idx}/ containing:

    setup.lisp     -- minimal package context (CL-USER) plus any imports
                      the reference macro happens to need
    tests.lisp     -- ((input-form . expected-expansion) ...) — one pair per
                      kata; can be augmented later
    metadata.json  -- carry cl-ds metadata so a GRPO sampler can
                      curriculum-sort or quality-filter

After writing each kata, the reference defmacro (cl-ds.output) is loaded
into SBCL and the input form is macroexpand-1'd. If the result equals
cl-ds.macroexpand, the kata is kept; otherwise it's quarantined under
<output_dir>/_rejected/ with the SBCL stderr for diagnosis.

Run on the laptop or any host with SBCL on PATH — this script is CPU-bound,
not GPU-bound. Validation step is the only slow part (~1 s per kata) and
runs in a worker pool.

Usage:
    uv run scripts/cloud/cl_ds_to_katas.py \\
        --dataset j14i/cl-ds \\
        --split train \\
        --output-dir ~/projects/macro-gym/katas/cl-ds \\
        --validate \\
        --workers 8

Env: HF_TOKEN if the dataset is private (it isn't, but consistency with
the rest of the pipeline).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ─── kata file templates ──────────────────────────────────────────────

SETUP_TEMPLATE = """;;; setup.lisp — generated from j14i/cl-ds row {idx}
;;; Sandbox: agent's macro definition runs in CL-USER.
;;;
;;; External packages referenced by the reference defmacro / expected
;;; expansion are stubbed below — empty packages that export the
;;; symbols we saw mentioned. This is enough to let SBCL READ the
;;; forms; semantic validity is checked separately via macroexpand-1.

{package_stubs}
(in-package :cl-user)
{requires}
"""

TESTS_TEMPLATE = """;;; tests.lisp — generated from j14i/cl-ds row {idx}
;;; instruction: {instruction_short}
;;; category: {category} | technique: {technique} | complexity: {complexity}
;;; quality_score: {quality_score}

(({input_form} . {expected_expansion}))
"""

VALIDATE_SCRIPT_TEMPLATE = """;;; Validation: define the reference macro, macroexpand-1 the test input,
;;; compare to expected. Writes "MATCH" or "MISMATCH" + diagnostics to stdout.

{package_stubs}
(in-package :cl-user)
(declaim (optimize (speed 0) (safety 3) (debug 3)))

(handler-case
    (progn
      {requires}
      {reference_macro}
      (let* ((input    '{input_form})
             (expected '{expected_expansion})
             (actual   (macroexpand-1 input)))
        (if (equal actual expected)
            (format t "~&MATCH~%")
            (progn
              (format t "~&MISMATCH~%")
              (format t "~&EXPECTED: ~S~%" expected)
              (format t "~&ACTUAL:   ~S~%" actual)))))
  (error (c)
    (format t "~&ERROR: ~A~%" c)))
"""


# ─── package-requires extraction ──────────────────────────────────────

# Pull qualified-symbol references (PKG:SYM or PKG::SYM) out of a CL
# source string. Used both for (a) building stub `defpackage` forms so
# the reader can find the package, and (b) listing systems to require
# via ASDF where available.

_QUALIFIED_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9.+_-]*)::?([A-Za-z!?<>=*_+/-][A-Za-z0-9!?<>=*_+/-]*)"
)

_STD_PACKAGES = {
    "cl", "common-lisp", "common-lisp-user", "cl-user",
    "sb-ext", "sb-int", "sb-impl", "sb-c", "sb-pcl", "sb-mop", "sb-kernel",
    "sb-sys", "sb-thread", "sb-debug", "sb-bsd-sockets", "sb-md5",
    "keyword", "system", "ext",
}

_ASDF_OVERRIDE = {
    "alex":             "alexandria",
    "alexandria.0.dev": "alexandria",
    "ax":               "alexandria",
    "alexandria.2":     "alexandria",
}


def detect_qualified_symbols(source: str) -> dict[str, set[str]]:
    """Return {pkg_name: set_of_symbol_names} for every PKG:SYM seen."""
    found: dict[str, set[str]] = {}
    for m in _QUALIFIED_RE.finditer(source):
        pkg = m.group(1).lower()
        sym = m.group(2).lower()
        if pkg in _STD_PACKAGES or len(pkg) <= 1:
            continue
        found.setdefault(pkg, set()).add(sym)
    return found


def detect_required_packages(source: str) -> list[str]:
    """Names of likely-ASDF systems we should attempt to require()."""
    seen = sorted(detect_qualified_symbols(source).keys())
    return [_ASDF_OVERRIDE.get(p, p) for p in seen]


def package_stubs_block(source: str) -> str:
    """Emit `defpackage` forms that create+export every symbol the reader
    will encounter. Empty packages — semantic validity is irrelevant here
    since we only `macroexpand-1`, never call the symbols' bound values.

    Wrapped in `eval-when` so the package exists at READ time for the
    *next* form in the file, not just at load time.
    """
    pkgs = detect_qualified_symbols(source)
    if not pkgs:
        return ""
    parts = []
    for pkg, syms in sorted(pkgs.items()):
        sym_list = " ".join(f'#:|{s}|' for s in sorted(syms))
        parts.append(
            f"(eval-when (:compile-toplevel :load-toplevel :execute)\n"
            f"  (unless (find-package :{pkg})\n"
            f"    (make-package :{pkg} :use nil))\n"
            f"  (let ((p (find-package :{pkg})))\n"
            f"    (dolist (s '({sym_list}))\n"
            f"      (let ((sym (intern (string-upcase (string s)) p)))\n"
            f"        (export sym p)))))"
        )
    return "\n".join(parts)


def requires_block(pkgs: list[str]) -> str:
    """Emit `(require :foo)` lines wrapped in ignore-errors so unknown
    systems don't crash the kata load. Belt-and-suspenders alongside the
    package-stubs block above."""
    if not pkgs:
        return ""
    lines = [f'(ignore-errors (require :{p}))' for p in pkgs]
    return "\n".join(lines)


# ─── kata writer ──────────────────────────────────────────────────────

@dataclass
class Kata:
    idx: int
    dir: Path
    row: dict
    requires: list[str]
    pkg_stubs: str


def write_kata(out_root: Path, idx: int, row: dict) -> Kata:
    kata_dir = out_root / f"cl-ds-{idx:05d}"
    kata_dir.mkdir(parents=True, exist_ok=True)

    # Scan reference defmacro + expected expansion + input for any qualified
    # symbol references. Build both stub packages (for the reader) and a
    # requires list (for ASDF, best-effort).
    all_source = " ".join([
        row.get("output") or "",
        row.get("macroexpand") or "",
        row.get("input") or "",
    ])
    pkg_stubs = package_stubs_block(all_source)
    requires = detect_required_packages(all_source)

    (kata_dir / "setup.lisp").write_text(
        SETUP_TEMPLATE.format(
            idx=idx,
            package_stubs=pkg_stubs,
            requires=requires_block(requires),
        )
    )

    instr = (row.get("instruction") or "").splitlines()[0][:120]
    (kata_dir / "tests.lisp").write_text(
        TESTS_TEMPLATE.format(
            idx=idx,
            instruction_short=instr,
            category=row.get("category") or "",
            technique=row.get("technique") or "",
            complexity=row.get("complexity") or "",
            quality_score=row.get("quality_score") or "",
            input_form=row["input"],
            expected_expansion=row["macroexpand"],
        )
    )

    meta = {
        "idx": idx,
        "instruction": row.get("instruction"),
        "reference_defmacro": row.get("output"),
        "category": row.get("category"),
        "technique": row.get("technique"),
        "complexity": row.get("complexity"),
        "quality_score": row.get("quality_score"),
        "requires_detected": requires,
    }
    (kata_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return Kata(idx=idx, dir=kata_dir, row=row, requires=requires, pkg_stubs=pkg_stubs)


# ─── validation ───────────────────────────────────────────────────────

def validate_kata(kata: Kata, sbcl: str = "sbcl", timeout: float = 10.0) -> tuple[bool, str]:
    """Drive SBCL: define reference macro, macroexpand-1 input, compare to
    cl-ds.macroexpand. Returns (ok, stderr_diagnostics).

    Uses a tempfile rather than /dev/stdin because sbcl's --script flag
    consumes the next argv as the script path, and we can't get
    --disable-debugger to take effect at the same time without a real
    file path.
    """
    script = VALIDATE_SCRIPT_TEMPLATE.format(
        package_stubs=kata.pkg_stubs,
        requires=requires_block(kata.requires),
        reference_macro=kata.row["output"],          # defmacro source
        input_form=kata.row["input"],
        expected_expansion=kata.row["macroexpand"],  # expected expansion
    )
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lisp", delete=False, encoding="utf-8",
    ) as tf:
        tf.write(script)
        script_path = tf.name
    try:
        # --script implies --non-interactive --disable-debugger --no-sysinit
        # --no-userinit --quit. Passing those extra flags explicitly silently
        # disables script execution — SBCL just prints its banner and exits.
        proc = subprocess.run(
            [sbcl, "--script", script_path],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        os.unlink(script_path)
        return False, "TIMEOUT"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    combined = (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or "")
    if "MATCH" in proc.stdout and "MISMATCH" not in proc.stdout and "ERROR:" not in proc.stdout:
        return True, ""
    return False, combined


def _validate_worker(args):
    kata_dict, sbcl, timeout = args
    # Rebuild a thin shim — dataclass not pickleable across all envs
    class _K:
        pass
    k = _K()
    k.idx = kata_dict["idx"]
    k.dir = Path(kata_dict["dir"])
    k.row = kata_dict["row"]
    k.requires = kata_dict["requires"]
    k.pkg_stubs = kata_dict["pkg_stubs"]
    ok, why = validate_kata(k, sbcl=sbcl, timeout=timeout)
    return kata_dict["idx"], ok, why


# ─── main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",    default="j14i/cl-ds")
    ap.add_argument("--split",      default="train", choices=["train", "validation", "test"])
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Destination root, typically macro-gym/katas/cl-ds/")
    ap.add_argument("--validate",   action="store_true",
                    help="Run SBCL to confirm each kata's reference produces the expected expansion")
    ap.add_argument("--workers",    type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit",      type=int, default=None,
                    help="Process only the first N rows (smoke testing)")
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="Skip rows with quality_score < this (default 0.0 = keep all)")
    ap.add_argument("--sbcl",       default="sbcl", help="Path to sbcl binary")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = args.output_dir / "_rejected"
    rejected_dir.mkdir(exist_ok=True)

    print(f"Loading {args.dataset} split={args.split}...")
    token = os.environ.get("HF_TOKEN")
    # cl-ds doesn't ship a YAML configs block — load_dataset(name) fails to
    # autodetect splits. Try the easy path first, fall back to explicit
    # data_files mapping (which works for any JSONL-only dataset).
    try:
        ds = load_dataset(args.dataset, split=args.split, token=token)
    except Exception as e:
        print(f"  load_dataset(name) failed ({type(e).__name__}); retrying with explicit data_files")
        # cl-ds layout: train.jsonl, val.jsonl, test.jsonl
        split_to_file = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}
        ds = load_dataset(
            args.dataset,
            data_files={args.split: split_to_file[args.split]},
            split=args.split,
            token=token,
        )
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  {len(ds)} rows")

    skipped_quality = 0
    katas: list[Kata] = []
    print("Writing katas...")
    for idx, row in enumerate(tqdm(ds)):
        if (row.get("quality_score") or 0.0) < args.min_quality:
            skipped_quality += 1
            continue
        katas.append(write_kata(args.output_dir, idx, row))

    print(f"  wrote {len(katas)} katas, skipped {skipped_quality} below quality threshold")

    if not args.validate:
        print("Skipping validation (pass --validate to run SBCL self-check)")
        return 0

    print(f"Validating {len(katas)} katas via SBCL (workers={args.workers})...")
    worker_args = [
        ({"idx": k.idx, "dir": str(k.dir), "row": k.row,
          "requires": k.requires, "pkg_stubs": k.pkg_stubs},
         args.sbcl, 10.0)
        for k in katas
    ]
    n_ok = 0
    n_bad = 0
    with mp.Pool(args.workers) as pool:
        for idx, ok, why in tqdm(
            pool.imap_unordered(_validate_worker, worker_args),
            total=len(worker_args),
        ):
            if ok:
                n_ok += 1
            else:
                n_bad += 1
                kata = next(k for k in katas if k.idx == idx)
                target = rejected_dir / kata.dir.name
                if kata.dir.exists():
                    kata.dir.rename(target)
                (target / "_failure.log").write_text(why)

    print(f"\nValidation complete:")
    print(f"  kept:     {n_ok}")
    print(f"  rejected: {n_bad}  (in {rejected_dir}/)")
    print(f"  pass rate: {100.0 * n_ok / max(1, n_ok + n_bad):.1f}%")

    # Summary by category / complexity for the kept set
    by_complexity: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for d in args.output_dir.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        by_complexity[meta.get("complexity") or "?"] = by_complexity.get(meta.get("complexity") or "?", 0) + 1
        by_category[meta.get("category") or "?"] = by_category.get(meta.get("category") or "?", 0) + 1

    print("\nKept by complexity:")
    for k, v in sorted(by_complexity.items()):
        print(f"  {k:>15} {v}")
    print("Kept by category:")
    for k, v in sorted(by_category.items()):
        print(f"  {k:>20} {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
