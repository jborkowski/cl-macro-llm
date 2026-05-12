;;; tests.lisp — hand-crafted, CL idiom
;;; instruction: Write `check-args` — signal an error if the predicate
;;; is false. Optional message string.
;;; category: validation | technique: precondition | complexity: basic
;;; quality_score: 0.9
;;; Note: expected expansions show the post-unwind form (WHEN → IF).

(
 ((check-args (> n 0))
  . (IF (NOT (> N 0)) (ERROR "bad arguments")))

 ((check-args (and (plusp x) (plusp y)) "x and y must be positive")
  . (IF (NOT (AND (PLUSP X) (PLUSP Y))) (ERROR "x and y must be positive")))

 ((check-args (stringp s) "expected a string")
  . (IF (NOT (STRINGP S)) (ERROR "expected a string")))
)
