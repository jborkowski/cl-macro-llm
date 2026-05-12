;;; tests.lisp — generated from j14i/cl-ds row 380
;;; instruction: Write a Common Lisp macro `clamp-to-min` that returns `val` if it is greater than or equal to `min`, otherwise returns `
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((clamp-to-min (compute) (floor-val)) . (LET ((#:G1 (COMPUTE)) (#:G2 (FLOOR-VAL))) (IF (< #:G1 #:G2) #:G2 #:G1))))
