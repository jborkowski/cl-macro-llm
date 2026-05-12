;;; tests.lisp — adversarial pair (sequential binding)
;;; instruction: Write `bind-vars-star` — sequential let bindings. Each
;;; RHS may reference earlier bindings in this same form.
;;; category: capture-management | technique: sequential-binding | complexity: basic
;;; quality_score: 0.95

(
 ((bind-vars-star ((x 1) (y (* x 2))) (+ x y))
  . (LET* ((X 1) (Y (* X 2))) (+ X Y)))

 ((bind-vars-star ((a (read-a)) (b (validate a))) (use a b))
  . (LET* ((A (READ-A)) (B (VALIDATE A))) (USE A B)))
)
