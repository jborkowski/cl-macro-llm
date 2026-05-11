# Debugging Common Lisp macros

How to inspect what a macro actually does — at the REPL, in a script, or
inside the GRPO reward path.

## SBCL toolkit (drop into `~/.sbclrc`)

```lisp
;; macro research tools (all built-in)
;;   (macroexpand-1 '(form))             ; one level
;;   (macroexpand   '(form))             ; full recursive
;;   (sb-cltl2:macroexpand-all '(form))  ; whole-tree expansion
;;   (describe 'symbol)                  ; docstring, type, source
;;   (inspect object)                    ; interactive browser
;;   (disassemble 'fn)                   ; native asm
;;   (time (form))                       ; wall-clock, GC, CPU
;;   (require :sb-sprof)                 ; sampling profiler
;;   (sb-sprof:with-profiling (:report :graph) ...)  ; call graph
```

Quick reference, narrowed to what's actually needed when iterating on a
macro:

| Form | When to reach for it |
|---|---|
| `(macroexpand-1 '(foo …))` | Most common. One level — see what *your* macro produces, ignoring downstream expansion. Same thing the GRPO reward fn calls. |
| `(macroexpand '(foo …))` | Recursive — see the fully-expanded form a compiler will see. Use when one macro layers on another. |
| `(sb-cltl2:macroexpand-all '(form))` | Like `macroexpand` but walks every subform too. Use to debug nested expansions inside `let`, `cond`, etc. |
| `(describe 'foo)` | Lookup who defined this symbol, where, with what arglist. Cheap before you start guessing. |
| `(disassemble #'foo)` | Last resort — when expansion looks fine but runtime is wrong. |
| `(time (form))` | Reality check on perf claims. |
| `(sb-sprof:with-profiling …)` | When you suspect a macro's expansion is producing a hot loop. |

## How a kata is graded (what the model is optimising against)

The macro-gym kata + reward path mirrors `cl_ds_to_katas.py`'s validator:

```lisp
(in-package :cl-user)

;; 1. install the candidate defmacro the model produced
(defmacro my-macro (...) ...)

;; 2. run macroexpand-1 on the reference call form
(let* ((input    '(my-macro foo bar))
       (expected '(... reference expansion ...))
       (actual   (macroexpand-1 input)))
  (if (equal actual expected)
      "MATCH"        ; reward ≈ 1.0
      "MISMATCH"))   ; reward ≤ 0
```

Two reasons the comparison can fail spuriously (false negatives) — both
already canonicalised by `scripts/validate_creative_macros.py:_normalize`
on the dataset side, and by SBCL's own reader on the `cl_ds_to_katas`
side, so neither matters for current training:

- **Reader shorthand.** `'X` and `(QUOTE X)`, `#'X` and `(FUNCTION X)`,
  `'(a b c)` and `(QUOTE (A B C))` print differently but are `EQUAL`.
- **Empty-list / NIL.** `()` and `NIL` are the same object; SBCL's
  printer always emits `NIL` (e.g. `(LAMBDA NIL body)`), but expansions
  written by humans use `()`.

## Reproducing one kata locally

If you suspect a specific kata is broken (or want to know what the model
is producing for it):

```bash
KATA=/workspace/katas/cl-ds/cl-ds-0042
ls $KATA                                  # metadata.json setup.lisp tests.lisp
cat $KATA/setup.lisp                      # the reference defmacro

# Round-trip it through SBCL — same path the trainer takes:
sbcl --script $KATA/setup.lisp <<'EOF'
(let ((*print-pretty* nil) (*print-readably* nil))
  (format t "~&__ACTUAL__~S~%" (macroexpand-1 (car (read-from-string "<input from tests.lisp>")))))
EOF
```

## Inspecting what GRPO is actually generating

The reward fn dumps 4 rollouts to JSONL every `LOG_SAMPLES_EVERY` steps
(default 25). To find the worst katas at any point:

```bash
uv run scripts/cloud/triage_katas.py \
  --samples-dir grpo-output/full \
  --top 10
```

Output ranks katas by mean reward across all dumped rollouts and shows
one sample completion per kata — useful for "why does the model
keep flubbing `with-foo-bar`?" without having to ssh into the pod.

## Forcing a sample dump on demand

```bash
echo '{"sample_now": true}' > grpo-output/full/control.json
# wait one training step, then:
ls grpo-output/full/samples-step-*.jsonl | tail -1 | xargs cat | head -40
```
