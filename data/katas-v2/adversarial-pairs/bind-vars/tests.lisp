;;; tests.lisp — adversarial pair (parallel binding)
;;; instruction: Write `bind-vars` — parallel let bindings. Each RHS
;;; sees the OUTER environment, not earlier bindings in this form.
;;; category: capture-management | technique: parallel-binding | complexity: basic
;;; quality_score: 0.95

(
 ((bind-vars ((x 1) (y 2)) (+ x y))
  . (LET ((X 1) (Y 2)) (+ X Y)))

 ((bind-vars ((a (outer-a)) (b (outer-b))) (combine a b))
  . (LET ((A (OUTER-A)) (B (OUTER-B))) (COMBINE A B)))
)
