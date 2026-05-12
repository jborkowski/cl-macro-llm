;;; tests.lisp — generated from j14i/cl-ds row 229
;;; instruction: Write a Common Lisp macro `ensure-symbol` that signals an error if its argument is not a symbol; otherwise returns the a
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((ensure-symbol name) . (LET ((#:G1 NAME)) (UNLESS (SYMBOLP #:G1) (ERROR "Expected a symbol, got ~S" #:G1)) #:G1)))
