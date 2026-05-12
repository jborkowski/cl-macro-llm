;;; tests.lisp — generated from j14i/cl-ds row 39
;;; instruction: Write a Common Lisp macro `max-of` that returns the larger of two expressions without calling the built-in max. Each sub
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((max-of (f) (g)) . (LET ((#:G1 (F)) (#:G2 (G))) (IF (> #:G1 #:G2) #:G1 #:G2))))
