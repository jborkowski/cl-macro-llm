;;; tests.lisp — generated from j14i/cl-ds row 379
;;; instruction: Write a Common Lisp macro `square-difference` that computes (a - b) squared, evaluating both `a` and `b` exactly once an
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((square-difference x (offset)) . (LET* ((#:G1 X) (#:G2 (OFFSET)) (#:G3 (- #:G1 #:G2))) (* #:G3 #:G3))))
