;;; tests.lisp — generated from j14i/cl-ds row 377
;;; instruction: Write a Common Lisp macro `max-of-three` that returns the largest of three expressions, evaluating each exactly once. Us
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((max-of-three (f) (g) (h)) . (LET ((#:G1 (F)) (#:G2 (G)) (#:G3 (H))) (IF (> #:G1 #:G2) (IF (> #:G1 #:G3) #:G1 #:G3) (IF (> #:G2 #:G3) #:G2 #:G3)))))
