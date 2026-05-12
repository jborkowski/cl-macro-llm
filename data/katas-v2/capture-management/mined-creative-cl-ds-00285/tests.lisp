;;; tests.lisp — generated from j14i/cl-ds row 285
;;; instruction: Write a Common Lisp macro `tri-max` that returns the largest of three expressions. Each argument may be a side-effecting
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((tri-max (x) (y) (z)) . (LET ((#:G1 (X)) (#:G2 (Y)) (#:G3 (Z))) (IF (>= #:G1 #:G2) (IF (>= #:G1 #:G3) #:G1 #:G3) (IF (>= #:G2 #:G3) #:G2 #:G3)))))
