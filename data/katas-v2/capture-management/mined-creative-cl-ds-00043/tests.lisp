;;; tests.lisp — generated from j14i/cl-ds row 43
;;; instruction: Write a Common Lisp macro `safe-divide` that divides its first argument by its second, returning 0 when the divisor is z
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((safe-divide total count) . (LET ((#:G1 COUNT)) (IF (ZEROP #:G1) 0 (/ TOTAL #:G1)))))
