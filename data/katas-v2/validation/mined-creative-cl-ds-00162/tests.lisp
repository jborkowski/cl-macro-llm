;;; tests.lisp — generated from j14i/cl-ds row 162
;;; instruction: Write a Common Lisp macro `ensure-integer` that evaluates an expression once; if the value is an integer return it, othe
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((ensure-integer (read)) . (LET ((#:G1 (READ))) (IF (INTEGERP #:G1) #:G1 (ERROR "Not an integer: ~A" #:G1)))))
