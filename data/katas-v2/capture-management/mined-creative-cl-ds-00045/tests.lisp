;;; tests.lisp — generated from j14i/cl-ds row 45
;;; instruction: Write a Common Lisp macro `swap-vars` that exchanges the values of two variables through a hidden temporary binding the 
;;; category: capture-management | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((swap-vars x y) . (LET ((#:G1 X)) (SETF X Y) (SETF Y #:G1))))
