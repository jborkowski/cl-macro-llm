;;; tests.lisp — generated from j14i/cl-ds row 228
;;; instruction: Write a Common Lisp macro `assert-length` that evaluates a sequence and an expected length once each, signals an error i
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((assert-length pair 2) . (LET ((#:G1 PAIR) (#:G2 2)) (UNLESS (= (LENGTH #:G1) #:G2) (ERROR "Expected length ~S, got ~S" #:G2 (LENGTH #:G1))) #:G1)))
