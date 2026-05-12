;;; tests.lisp — generated from j14i/cl-ds row 498
;;; instruction: Write a Common Lisp macro `assert-instance` that evaluates a value once, signals an error unless it is TYPEP of the name
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((assert-instance obj account) . (LET ((#:G1 OBJ)) (UNLESS (TYPEP #:G1 'ACCOUNT) (ERROR "~A is not an instance of ~A" #:G1 'ACCOUNT)) #:G1)))
