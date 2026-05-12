;;; tests.lisp — generated from j14i/cl-ds row 47
;;; instruction: Write a Common Lisp macro `random-between` that returns a random integer offset between `lo` (inclusive) and `hi` (exclu
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((random-between 1 100) . (LET ((#:G1 1) (#:G2 100)) (+ #:G1 (RANDOM (- #:G2 #:G1))))))
