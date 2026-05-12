;;; tests.lisp — generated from j14i/cl-ds row 305
;;; instruction: Write a Common Lisp macro `ensure-not-nil` that evaluates an expression once, signals an error if the value is nil, and 
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((ensure-not-nil (find-user id)) . (LET ((#:G1 (FIND-USER ID))) (WHEN (NULL #:G1) (ERROR "Expected non-nil value")) #:G1)))
