;;; tests.lisp — generated from j14i/cl-ds row 283
;;; instruction: Write a Common Lisp macro `hypot` that computes the Euclidean hypotenuse sqrt(a^2 + b^2). Bind each argument to a gensym
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((hypot (foo) (bar)) . (LET ((#:G1 (FOO)) (#:G2 (BAR))) (SQRT (+ (* #:G1 #:G1) (* #:G2 #:G2))))))
