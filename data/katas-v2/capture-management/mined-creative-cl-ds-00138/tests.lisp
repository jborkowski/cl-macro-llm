;;; tests.lisp — generated from j14i/cl-ds row 138
;;; instruction: Write a Common Lisp macro `clampf` that clamps a place to lie between LO and HI. LO and HI must each be evaluated exactl
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((clampf x 0 100) . (LET ((#:G1 0) (#:G2 100)) (SETF X (MAX #:G1 (MIN #:G2 X))))))
