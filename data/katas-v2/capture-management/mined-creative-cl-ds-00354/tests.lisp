;;; tests.lisp — generated from j14i/cl-ds row 354
;;; instruction: Write a Common Lisp macro `hypot-of` that returns the hypotenuse sqrt(a^2 + b^2) of two expressions while evaluating eac
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((hypot-of (dx) (dy)) . (LET ((#:G1 (DX)) (#:G2 (DY))) (SQRT (+ (* #:G1 #:G1) (* #:G2 #:G2))))))
