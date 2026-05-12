;;; tests.lisp — generated from j14i/cl-ds row 468
;;; instruction: Write a Common Lisp macro `assert-type` that evaluates an expression once, checks that it satisfies a given type, and si
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-type (read) integer) . (LET ((#:G1 (READ))) (UNLESS (TYPEP #:G1 'INTEGER) (ERROR "Type assertion failed: ~S => ~S, not of type ~S" '(READ) #:G1 'INTEGER)) #:G1)))
