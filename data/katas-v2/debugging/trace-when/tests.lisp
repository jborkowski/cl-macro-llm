;;; tests.lisp — hand-crafted, CL idiom
;;; instruction: Write `trace-when` — emit a trace line and run the
;;; body, but only when a runtime condition is true.
;;; category: debugging | technique: conditional-trace | complexity: basic
;;; quality_score: 0.9
;;; Note: expected expansions show the post-unwind form (WHEN-multi → IF+PROGN).

(
 ((trace-when *verbose* (process item))
  . (IF *VERBOSE*
        (PROGN (FORMAT *TRACE-OUTPUT* "[trace]~%") (PROCESS ITEM))))

 ((trace-when (> n 100) (log :hot n) (flush))
  . (IF (> N 100)
        (PROGN (FORMAT *TRACE-OUTPUT* "[trace]~%") (LOG :HOT N) (FLUSH))))

 ((trace-when (debug-mode-p) (dump-state))
  . (IF (DEBUG-MODE-P)
        (PROGN (FORMAT *TRACE-OUTPUT* "[trace]~%") (DUMP-STATE))))
)
