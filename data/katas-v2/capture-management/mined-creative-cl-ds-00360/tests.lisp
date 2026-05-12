;;; tests.lisp — generated from j14i/cl-ds row 360
;;; instruction: Write a Common Lisp predicate macro `power-of-two-p` that tests whether an integer expression is a positive power of two
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((power-of-two-p (compute)) . (LET ((#:G1 (COMPUTE))) (AND (PLUSP #:G1) (ZEROP (LOGAND #:G1 (1- #:G1)))))))
