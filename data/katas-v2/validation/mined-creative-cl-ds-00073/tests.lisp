;;; tests.lisp — generated from j14i/cl-ds row 73
;;; instruction: Write a Common Lisp macro `assert-equal` that evaluates two expressions each once and signals an error if they are not E
;;; category: validation | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((assert-equal (compute) expected) . (LET ((#:G1 (COMPUTE)) (#:G2 EXPECTED)) (UNLESS (EQUAL #:G1 #:G2) (ERROR "Expected ~S to equal ~S" #:G1 #:G2)) #:G1)))
