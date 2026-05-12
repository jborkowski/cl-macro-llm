;;; tests.lisp — generated from j14i/cl-ds row 414
;;; instruction: Write a Common Lisp macro `assert-non-nil` that evaluates an expression once; if the value is nil it signals an error me
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-non-nil (find x list)) . (LET ((#:G1 (FIND X LIST))) (UNLESS #:G1 (ERROR "Assertion failed: ~S is nil" '(FIND X LIST))) #:G1)))
