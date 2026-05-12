;;; tests.lisp — generated from j14i/cl-ds row 355
;;; instruction: Write a Common Lisp macro `swap-via-temp` that swaps the values of two places using a hidden gensym temporary so the use
;;; category: capture-management | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((swap-via-temp x y) . (LET ((#:G1 X)) (SETF X Y Y #:G1))))
