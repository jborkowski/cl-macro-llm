;;; tests.lisp — generated from j14i/cl-ds row 165
;;; instruction: Write a Common Lisp macro `validate-range` that evaluates an expression and two bounds exactly once each; if the value i
;;; category: validation | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((validate-range (read-input) 0 100) . (LET ((#:G1 (READ-INPUT)) (#:G2 0) (#:G3 100)) (UNLESS (AND (>= #:G1 #:G2) (<= #:G1 #:G3)) (ERROR "Value ~A out of range [~A, ~A]" #:G1 #:G2 #:G3)) #:G1)))
