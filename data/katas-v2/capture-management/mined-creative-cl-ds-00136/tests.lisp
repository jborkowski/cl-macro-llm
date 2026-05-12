;;; tests.lisp — generated from j14i/cl-ds row 136
;;; instruction: Write a Common Lisp macro `swapf` that swaps the values of two places using a hidden gensym temporary so the user cannot
;;; category: capture-management | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((swapf x y) . (LET ((#:G1 X)) (SETF X Y) (SETF Y #:G1))))
