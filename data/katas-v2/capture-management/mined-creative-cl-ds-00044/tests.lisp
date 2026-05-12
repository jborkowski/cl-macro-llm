;;; tests.lisp — generated from j14i/cl-ds row 44
;;; instruction: Write a Common Lisp macro `xor2` that returns true when exactly one of two expressions is truthy. Each expression must b
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((xor2 p q) . (LET ((#:G1 P) (#:G2 Q)) (AND (OR #:G1 #:G2) (NOT (AND #:G1 #:G2))))))
