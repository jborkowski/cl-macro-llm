;;; tests.lisp — generated from j14i/cl-ds row 225
;;; instruction: Write a Common Lisp macro `assert-string` that evaluates a value once and signals an error if it is not a string. Return
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((assert-string name) . (LET ((#:G1 NAME)) (UNLESS (STRINGP #:G1) (ERROR "Expected a string, got ~S" #:G1)) #:G1)))
