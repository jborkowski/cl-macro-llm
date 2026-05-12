;;; tests.lisp — generated from j14i/cl-ds row 41
;;; instruction: Write a Common Lisp macro `abs-of` that returns the absolute value of its argument without invoking abs. The argument ex
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((abs-of (delta)) . (LET ((#:G1 (DELTA))) (IF (MINUSP #:G1) (- #:G1) #:G1))))
