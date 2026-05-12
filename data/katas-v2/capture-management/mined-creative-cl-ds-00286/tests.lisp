;;; tests.lisp — generated from j14i/cl-ds row 286
;;; instruction: Write a Common Lisp macro `safe-div` that divides a by b, returning 0 if b is zero. The divisor b must be evaluated only
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((safe-div x (denominator)) . (LET ((#:G1 (DENOMINATOR))) (IF (ZEROP #:G1) 0 (/ X #:G1)))))
