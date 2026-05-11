#!/usr/bin/env python3
"""Apply safety + grading + parallelism patches to macro-gym.

Run after `git clone https://github.com/jborkowski/macro-gym` (and before
the trainer's first MacroEnv step). Idempotent — each patch is guarded by
a string check, so re-running on already-patched files is a no-op.

Takes the path to either `macro-gym/` (the package root) or directly to
`macro-gym/lisp/server.lisp` for backward compat. Patches both server.lisp
and macro_gym/env.py.

Patches applied:

  P1 — server.lisp: sb-ext:with-timeout 5 around macroexpand-1
        Prevents a hostile or buggy model output (e.g. (defmacro foo ()
        (sleep 30))) from wedging the SBCL service forever. Hits the
        timeout in 5 s and returns "ERROR: macroexpand-1 timeout (5s)".

  P2 — server.lisp: optionally switch macroexpand-1 → sb-walker:macroexpand-all
        Stricter equivalence check (walks tree). Disabled by default;
        enable with --use-macroexpand-all.

  P3 — server.lisp: pre-compile defmacro source before installing
        Catches malformed input (unbalanced parens, undefined helpers)
        BEFORE the body runs during expansion.

  P4 — server.lisp: lower compiler debug at server start
        (restrict-compiler-policy 'debug 0) + 'speed 3
        Faster compile, cleaner expansions.

  P5 — env.py: per-instance SBCLService (was global singleton)
        Each MacroEnv spawns its OWN SBCL subprocess. With ThreadPoolExecutor
        in the reward fn, this unlocks true parallel macro expansion across
        the pod's CPU cores (64 on this A100 host). Singleton design used
        to serialize ALL macro evals through one SBCL.

Usage:
  python3 scripts/cloud/patch_macro_gym.py /workspace/macro-gym
  python3 scripts/cloud/patch_macro_gym.py /workspace/macro-gym/lisp/server.lisp

Exit code 0 if any patch applied OR everything already in place;
non-zero only if the file structure doesn't match what we expect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ─── per-patch definitions ───────────────────────────────────────────


P1_GUARD = "sb-ext:with-timeout"
P1_OLD = """(actual (handler-case (macroexpand-1 input)
                           (error (c) (format nil "ERROR: ~a" c))))"""
P1_NEW_TPL = """(actual (handler-case (sb-ext:with-timeout {timeout} ({expand_fn} input))
                           (sb-ext:timeout (c) (declare (ignore c)) "ERROR: {expand_fn} timeout ({timeout}s)")
                           (error (c) (format nil "ERROR: ~a" c))))"""


P3_GUARD = "(defun install-macro"
P3_OLD = """(defun install-macro (defmacro-form)
  "Eval the defmacro form and return the macro name."
  (eval defmacro-form)
  (cadr defmacro-form))"""
P3_NEW = """(defun install-macro (defmacro-form)
  "Eval the defmacro form and return the macro name."
  ;; Pre-compile in a throwaway context first: catches malformed defmacro
  ;; (unbalanced parens, undefined helpers, ill-typed forms) before any
  ;; macro body would run during expansion. Signals an error on bad input
  ;; that the outer handler-case turns into a -0.1 reward, not a hang.
  (handler-case
      (sb-ext:with-timeout 5
        (compile nil `(lambda () ,defmacro-form)))
    (sb-ext:timeout (c)
      (declare (ignore c))
      (error "install-macro: compile timeout (5s)")))
  (eval defmacro-form)
  (cadr defmacro-form))"""


P4_GUARD = "restrict-compiler-policy"
# Inserted right after the (in-package :macro-gym) form near the top.
P4_OLD = "(in-package :macro-gym)"
P4_NEW = """(in-package :macro-gym)

;; Tighten compiler policy so expansions don't carry debug instrumentation
;; and compile/load is fast — model-generated macros land in this server
;; thousands of times per training run.
(sb-ext:restrict-compiler-policy 'debug 0)
(sb-ext:restrict-compiler-policy 'speed 3)"""


# P5 — env.py: per-instance SBCLService instead of global singleton.
# The original `reset()` and `step()` both call `get_service()` which
# returns a process-wide singleton. That serialises ALL macro evals
# across the entire trainer. Replacing with per-instance SBCLService
# enables true parallel macro expansion across the host's cores.
P5_GUARD = "self._sbcl = SBCLService()"
# Upstream `step()` calls `sbcl = get_service()` (singleton). Replace with
# per-instance lazy-init (matching upstream's lazy pattern, just not shared).
P5_OLD_STEP = """        sbcl = get_service()
        result = sbcl.eval_macro(self.kata_id, action)"""
P5_NEW_STEP = """        # Per-instance SBCL — each MacroEnv has its own subprocess so
        # ThreadPoolExecutor in the reward fn can parallelise macro
        # evaluation across the host's cores (was singleton, bottleneck).
        if self._sbcl is None:
            self._sbcl = SBCLService()
            self._sbcl.start()
        result = self._sbcl.eval_macro(self.kata_id, action)"""
# Upstream close() is one-liner calling the global shutdown.
P5_OLD_CLOSE = """    def close(self):
        shutdown_service()"""
P5_NEW_CLOSE = """    def close(self):
        # Per-instance teardown — only stop OUR SBCL, not the global one.
        if self._sbcl is not None:
            try:
                self._sbcl.stop()
            except Exception:
                pass
            self._sbcl = None"""


# ─── apply ───────────────────────────────────────────────────────────


def _apply(src: str, *, guard: str, old: str, new: str, name: str) -> tuple[str, bool]:
    if guard in src and "ERROR: macroexpand timeout" not in src and name == "P1":
        # P1 guard is a substring of the new code we'd add — if it's there
        # but our expected new line isn't, treat as a different patch from
        # an earlier attempt. Reapply by removing and re-adding is unsafe;
        # let the operator look manually.
        return src, False
    if name == "P1" and "ERROR: macroexpand timeout" in src:
        return src, False  # already applied
    if name == "P2" and "macroexpand-all input" in src:
        return src, False  # already applied
    if name == "P3" and "compile nil `(lambda () ,defmacro-form)" in src:
        return src, False  # already applied
    if name == "P4" and P4_GUARD in src:
        return src, False  # already applied
    if old not in src:
        print(f"  xx {name}: anchor not found — upstream may have refactored",
              file=sys.stderr)
        sys.exit(2)
    return src.replace(old, new, 1), True


def _patch_env_py(env_py: Path) -> list[str]:
    """P5: per-instance SBCLService in MacroEnv. Returns list of failures.

    Upstream MacroEnv calls a process-global `get_service()` singleton in
    step() and `shutdown_service()` in close(). We rewrite both so each
    MacroEnv owns its own SBCLService — this is the only thing that lets
    the reward fn's ThreadPoolExecutor actually run in parallel."""
    failures: list[str] = []
    if not env_py.is_file():
        return [f"P5: {env_py} not found"]

    src = env_py.read_text()
    original = src

    if P5_GUARD in src:
        print("  P5: already applied")
        return []

    if P5_OLD_STEP in src:
        src = src.replace(P5_OLD_STEP, P5_NEW_STEP, 1)
        print("  ✓ P5.step: per-instance SBCL in MacroEnv.step()")
    else:
        failures.append("P5.step: get_service() anchor not found in step()")

    if P5_OLD_CLOSE in src:
        src = src.replace(P5_OLD_CLOSE, P5_NEW_CLOSE, 1)
        print("  ✓ P5.close: per-instance SBCL teardown in close()")
    else:
        # close() shape may have changed — print a note but don't fail;
        # step() is the critical path.
        print("  ~ P5.close: anchor not found (non-critical)")

    if src != original:
        env_py.write_text(src)
        print(f"  wrote {env_py}")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path,
                    help="path to macro-gym/ (package root) OR macro-gym/lisp/server.lisp")
    ap.add_argument("--use-macroexpand-all", action="store_true", default=False,
                    help="apply P2 (switch to sb-walker:macroexpand-all for "
                         "stricter grading; requires (require :sb-walker))")
    ap.add_argument("--timeout-seconds", type=int, default=5,
                    help="P1 timeout window (default 5)")
    ap.add_argument("--no-env-patch", action="store_true",
                    help="skip P5 (env.py per-instance SBCL); apply only server.lisp patches")
    args = ap.parse_args()

    # Resolve target to server.lisp + (optional) env.py
    target = args.target.expanduser().resolve()
    if target.is_dir():
        macro_gym_root = target
        p = macro_gym_root / "lisp" / "server.lisp"
        env_py = macro_gym_root / "macro_gym" / "env.py"
    elif target.is_file() and target.name == "server.lisp":
        p = target
        env_py = target.parent.parent / "macro_gym" / "env.py"
    else:
        print(f"  xx target must be macro-gym/ dir or .../lisp/server.lisp; got {target}",
              file=sys.stderr)
        return 1

    if not p.is_file():
        print(f"  xx server.lisp not found at {p}", file=sys.stderr)
        return 1

    src = p.read_text()
    original = src
    # `sb-ext:macroexpand-all` doesn't exist; the correct symbol is
    # `sb-walker:macroexpand-all` and it requires loading the sb-walker
    # contrib. Default is the conservative macroexpand-1, matching the
    # original grading semantics.
    expand_fn = "sb-walker:macroexpand-all" if args.use_macroexpand_all else "macroexpand-1"

    # Apply each patch; collect failures but keep going so partial progress
    # writes through and the operator sees the full picture.
    failures: list[str] = []

    # P1 + P2 share the same anchor (the macroexpand-1 call) — applied together.
    p1_new = P1_NEW_TPL.format(timeout=args.timeout_seconds, expand_fn=expand_fn)
    if "ERROR: macroexpand-1 timeout" in src or "ERROR: sb-ext:macroexpand-all timeout" in src:
        print("  P1+P2: already applied")
    elif P1_OLD in src:
        src = src.replace(P1_OLD, p1_new, 1)
        print(f"  ✓ P1+P2: wrapped {expand_fn} in with-timeout "
              f"{args.timeout_seconds}s")
    else:
        failures.append("P1+P2: macroexpand-1 anchor not found")

    # P3 — install-macro with pre-compile timeout.
    if "compile nil `(lambda () ,defmacro-form)" in src:
        print("  P3: already applied")
    elif P3_OLD in src:
        src = src.replace(P3_OLD, P3_NEW, 1)
        print(f"  ✓ P3: pre-compile guard around install-macro")
    else:
        failures.append("P3: install-macro anchor not found")

    # P4 — compiler policy at server init.
    if P4_GUARD in src:
        print("  P4: already applied")
    elif P4_OLD in src:
        src = src.replace(P4_OLD, P4_NEW, 1)
        print(f"  ✓ P4: compiler policy debug=0 speed=3")
    else:
        failures.append("P4: (in-package :macro-gym) anchor not found")

    if src != original:
        p.write_text(src)
        print(f"  wrote {p}")
    else:
        print(f"  no server.lisp changes needed")

    # P5 — env.py patches
    if not args.no_env_patch:
        env_failures = _patch_env_py(env_py)
        failures.extend(env_failures)

    if failures:
        for f in failures:
            print(f"  xx {f}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
