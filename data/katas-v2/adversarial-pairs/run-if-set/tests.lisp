;;; tests.lisp — adversarial pair (run when test is non-nil)
;;; instruction: Write `run-if-set` — run body ONLY when test is set
;;; (non-nil). Single body or multiple forms; use a special-form
;;; expansion so the unwound shape is direct.
;;; category: control-flow | technique: conditional | complexity: basic
;;; quality_score: 0.95

(
 ((run-if-set *flag* (do-the-thing))
  . (IF *FLAG* (DO-THE-THING)))

 ((run-if-set (ready-p) (start) (announce))
  . (IF (READY-P) (PROGN (START) (ANNOUNCE))))
)
