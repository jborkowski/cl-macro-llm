;;; tests.lisp — generated from j14i/cl-ds row 40
;;; instruction: Write a Common Lisp macro `min-of` that returns the smaller of two expressions without calling the built-in min. Both su
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((min-of x y) . (LET ((#:G1 X) (#:G2 Y)) (IF (< #:G1 #:G2) #:G1 #:G2))))
