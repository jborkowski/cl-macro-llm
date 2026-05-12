#!/usr/bin/env -S /Users/jonatan/projects/macro-gym/.venv/bin/python -u
"""mine_rejected.py — promote viable katas out of _rejected/ pools.

Pipeline:
  1. Walk cl-ds/_rejected/ and creative/_rejected/ — 2209 directories.
  2. Filter by target category (default: capture-management, validation,
     debugging, resource-management — the user-prioritized gaps).
  3. Static-IO scan: reject any kata whose setup.lisp contains tokens
     that suggest filesystem/network/eval at load time.
  4. Copy candidates to a single staging dir with `mined-` prefix.
  5. Validate via macro-gym's MacroGrader: the reference_defmacro must
     self-grade at reward=1.0. This automatically enforces the grader
     rule that top-level macro forms (WHEN/UNLESS/DO/...) get unwound
     until the head is a special operator — bad katas surface here.
  6. Move accepted katas to data/katas-v2/<category>/mined-<orig-id>/,
     extending metadata.json with `source: "mined-cl-ds-rejected"` and
     a truncated _failure.log snippet for audit.
  7. Emit per-phase stats.

Usage:
  python scripts/mine_rejected.py --dry-run
  python scripts/mine_rejected.py --limit 50 --categories capture-management
  python scripts/mine_rejected.py            # full run, default categories
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path("/Users/jonatan/projects/cl-macro-llm")
REJECTED_DIRS = [
    REPO_ROOT / "data/grpo-sanity/katas/cl-ds/_rejected",
    REPO_ROOT / "data/grpo-sanity/katas/creative/_rejected",
]
OUT_ROOT = REPO_ROOT / "data/katas-v2"

DEFAULT_CATEGORIES = {
    "capture-management",
    "validation",
    "debugging",
    "resource-management",
}

IO_BANLIST = (
    "with-open-file",
    "with-open-stream",
    "read-from-string",
    "run-program",
    "sb-ext:",
    "delete-file",
    "rename-file",
    "make-pathname",
    "probe-file",
    "socket-",
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen, don't write or grade")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N candidates (post-category-filter)")
    ap.add_argument("--categories", type=str,
                    default=",".join(sorted(DEFAULT_CATEGORIES)),
                    help="comma-separated target categories")
    ap.add_argument("--pool-size", type=int, default=6,
                    help="SBCL workers for validation pool")
    ap.add_argument("--verbose", action="store_true",
                    help="log every per-kata decision")
    return ap.parse_args()


def static_io_check(setup_text):
    """Return the first banned token found, or None if clean."""
    for token in IO_BANLIST:
        if token in setup_text:
            return token
    return None


def load_failure_log(kata_dir):
    log_path = kata_dir / "_failure.log"
    if not log_path.exists():
        return ""
    try:
        return log_path.read_text(errors="replace")[:800]
    except Exception:
        return ""


def collect_candidates(target_categories, limit, verbose):
    """Walk _rejected/ pools, apply category + static-IO filters."""
    candidates = []
    counters = {
        "total": 0,
        "missing_metadata": 0,
        "wrong_category": 0,
        "io_rejected": 0,
        "missing_files": 0,
        "passed_static": 0,
    }
    io_reject_tokens = {}

    for rejected_root in REJECTED_DIRS:
        if not rejected_root.is_dir():
            continue
        for kata_dir in sorted(rejected_root.iterdir()):
            if not kata_dir.is_dir():
                continue
            counters["total"] += 1
            meta_path = kata_dir / "metadata.json"
            setup_path = kata_dir / "setup.lisp"
            tests_path = kata_dir / "tests.lisp"
            if not meta_path.exists():
                counters["missing_metadata"] += 1
                continue
            if not (setup_path.exists() and tests_path.exists()):
                counters["missing_files"] += 1
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                counters["missing_metadata"] += 1
                continue
            cat = meta.get("category", "?")
            if cat not in target_categories:
                counters["wrong_category"] += 1
                continue
            setup_text = setup_path.read_text(errors="replace")
            banned = static_io_check(setup_text.lower())
            if banned:
                counters["io_rejected"] += 1
                io_reject_tokens[banned] = io_reject_tokens.get(banned, 0) + 1
                if verbose:
                    print(f"  IO-reject {kata_dir.name} ({cat}): {banned!r}")
                continue
            counters["passed_static"] += 1
            candidates.append((kata_dir, meta))
            if limit is not None and len(candidates) >= limit:
                return candidates, counters, io_reject_tokens
    return candidates, counters, io_reject_tokens


def stage_candidates(candidates, staging):
    """Copy each candidate to staging/mined-<pool>-<orig-id>/.

    Pool prefix prevents collisions between cl-ds/_rejected/ and
    creative/_rejected/ which can share basenames.
    """
    staged = []
    for kata_dir, meta in candidates:
        pool = kata_dir.parent.parent.name  # "cl-ds" or "creative"
        new_id = f"mined-{pool}-{kata_dir.name}"
        dst = staging / new_id
        shutil.copytree(kata_dir, dst)
        staged.append((new_id, dst, kata_dir, meta))
    return staged


def validate(staged, staging, pool_size, verbose):
    """Grade each staged kata; return (accepted, rejected_for_grading)."""
    import os
    os.environ["MACRO_GYM_KATA_ROOT"] = str(staging)
    from macro_gym import MacroGrader

    accepted, rejected = [], []
    grader = MacroGrader(pool_size=pool_size)
    try:
        for new_id, dst, orig_dir, meta in staged:
            ref = meta.get("reference_defmacro", "")
            if not ref:
                rejected.append((new_id, dst, orig_dir, meta, "no-reference"))
                continue
            try:
                r = grader.grade(new_id, ref)
            except Exception as e:
                rejected.append((new_id, dst, orig_dir, meta, f"exception: {e}"))
                continue
            reward = r.get("reward", -99)
            if reward == 1.0:
                accepted.append((new_id, dst, orig_dir, meta, r))
                if verbose:
                    print(f"  ✅ {new_id} ({meta.get('category')})")
            else:
                rejected.append((new_id, dst, orig_dir, meta,
                                 f"reward={reward} passed={r.get('passed')}/{r.get('total')} "
                                 f"sim={r.get('semantic_eq_score','?')}"))
                if verbose:
                    print(f"  ❌ {new_id} reward={reward} sim={r.get('semantic_eq_score')}")
    finally:
        grader.close()
    return accepted, rejected


def promote(accepted):
    """Move accepted katas from staging to data/katas-v2/<category>/."""
    promoted = []
    for new_id, staging_dst, orig_dir, meta, _grade_result in accepted:
        category = meta.get("category", "uncategorized")
        final_dir = OUT_ROOT / category / new_id
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            # Already promoted before — skip with warning
            print(f"  WARN: already exists, skipping: {final_dir}")
            continue
        shutil.move(str(staging_dst), str(final_dir))

        # Extend metadata: add source + truncated failure-log snippet
        new_meta = dict(meta)
        new_meta["source"] = "mined-cl-ds-rejected"
        new_meta["mined_from"] = str(orig_dir.relative_to(REPO_ROOT))
        failure_snippet = load_failure_log(orig_dir).strip()
        if failure_snippet:
            new_meta["original_failure_log_excerpt"] = failure_snippet[:600]
        (final_dir / "metadata.json").write_text(
            json.dumps(new_meta, indent=2) + "\n"
        )
        promoted.append((new_id, category, str(final_dir)))
    return promoted


def main():
    args = parse_args()
    target_categories = set(c.strip() for c in args.categories.split(","))

    print(f"=== mine_rejected.py ===")
    print(f"target categories: {sorted(target_categories)}")
    print(f"dry-run: {args.dry_run}  limit: {args.limit}  pool: {args.pool_size}")
    print()

    print("Phase 1 — collect candidates (category + static-IO)")
    candidates, counters, io_reject_tokens = collect_candidates(
        target_categories, args.limit, args.verbose
    )
    print(f"  scanned:           {counters['total']}")
    print(f"  missing metadata:  {counters['missing_metadata']}")
    print(f"  missing files:     {counters['missing_files']}")
    print(f"  wrong category:    {counters['wrong_category']}")
    print(f"  IO-rejected:       {counters['io_rejected']}  {dict(io_reject_tokens)}")
    print(f"  candidates:        {counters['passed_static']}")
    print()

    if not candidates:
        print("No candidates to validate. Exiting.")
        return

    if args.dry_run:
        print("DRY-RUN — would now grade and promote. Per-category breakdown:")
        per_cat = {}
        for _, meta in candidates:
            per_cat[meta.get("category", "?")] = per_cat.get(meta.get("category", "?"), 0) + 1
        for cat, n in sorted(per_cat.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:24s} {n}")
        return

    print(f"Phase 2 — stage {len(candidates)} candidates to /tmp/")
    with tempfile.TemporaryDirectory(prefix="mine-rejected-") as staging_str:
        staging = Path(staging_str)
        staged = stage_candidates(candidates, staging)
        print(f"  staged: {len(staged)} → {staging}")
        print()

        print(f"Phase 3 — grade via macro-gym (pool={args.pool_size})")
        accepted, rejected = validate(staged, staging, args.pool_size, args.verbose)
        print(f"  accepted: {len(accepted)}")
        print(f"  rejected: {len(rejected)}")
        print()

        if accepted:
            print(f"Phase 4 — promote {len(accepted)} accepted katas to {OUT_ROOT}/")
            promoted = promote(accepted)
            print(f"  promoted: {len(promoted)}")
        else:
            promoted = []

        per_cat_accepted = {}
        for _, _, _, meta, _ in accepted:
            cat = meta.get("category", "?")
            per_cat_accepted[cat] = per_cat_accepted.get(cat, 0) + 1
        print()
        print("=== summary ===")
        print(f"  total candidates passing static filters: {len(candidates)}")
        print(f"  passed grader at reward=1.0:             {len(accepted)}")
        print(f"  acceptance rate:                         "
              f"{100*len(accepted)/max(1,len(candidates)):.1f}%")
        print(f"  promoted to katas-v2/:                   {len(promoted)}")
        print("  per-category accepted:")
        for cat, n in sorted(per_cat_accepted.items(), key=lambda kv: -kv[1]):
            print(f"    {cat:24s} {n}")

        if rejected and args.verbose:
            print("\n  sample rejections (first 10):")
            for new_id, _, _, meta, reason in rejected[:10]:
                print(f"    {new_id} ({meta.get('category')}): {reason}")


if __name__ == "__main__":
    sys.exit(main() or 0)
