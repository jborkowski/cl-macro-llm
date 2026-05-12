;;; tests.lisp — adversarial pair (run when test is nil)
;;; instruction: Write `run-if-unset` — run body ONLY when test is
;;; unset (NIL). Multi-form body must execute in order.
;;; category: control-flow | technique: conditional | complexity: basic
;;; quality_score: 0.95

(
 ((run-if-unset *cache* (warm-cache))
  . (IF *CACHE* NIL (WARM-CACHE)))

 ((run-if-unset (already-initialized-p) (init-step-1) (init-step-2))
  . (IF (ALREADY-INITIALIZED-P) NIL (PROGN (INIT-STEP-1) (INIT-STEP-2))))
)
