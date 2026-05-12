;;; tests.lisp — generated from j14i/cl-ds row 208
;;; instruction: Write a Common Lisp macro `in-range-p` that returns true iff x is between lo and hi inclusive, evaluating each of the th
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((in-range-p (compute) 0 100) . (LET ((#:G1 (COMPUTE)) (#:G2 0) (#:G3 100)) (AND (>= #:G1 #:G2) (<= #:G1 #:G3)))))
