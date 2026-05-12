;;; tests.lisp — generated from j14i/cl-ds row 467
;;; instruction: Write a Common Lisp macro `assert-with-context` that asserts an expression is non-nil and, on failure, signals an error 
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-with-context (numberp x) "x must be a number") . (LET ((#:G1 (NUMBERP X))) (UNLESS #:G1 (ERROR "Assertion failed: ~S~%Context: ~A" '(NUMBERP X) "x must be a number")) #:G1)))
