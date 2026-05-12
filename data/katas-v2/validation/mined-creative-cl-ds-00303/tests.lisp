;;; tests.lisp — generated from j14i/cl-ds row 303
;;; instruction: Write a Common Lisp macro `assert-non-negative` that evaluates an expression once, signals an error if it is negative, a
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((assert-non-negative score) . (LET ((#:G1 SCORE)) (WHEN (MINUSP #:G1) (ERROR "Expected non-negative, got ~A" #:G1)) #:G1)))
