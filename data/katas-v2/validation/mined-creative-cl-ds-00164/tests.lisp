;;; tests.lisp — generated from j14i/cl-ds row 164
;;; instruction: Write a Common Lisp macro `check-not-zero` that evaluates an expression once; if the value is zero, signal an error; oth
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((check-not-zero (denominator x)) . (LET ((#:G1 (DENOMINATOR X))) (WHEN (ZEROP #:G1) (ERROR "Value must not be zero")) #:G1)))
