;;; tests.lisp — generated from j14i/cl-ds row 290
;;; instruction: Write a Common Lisp macro `swap-places` that exchanges the values of two setf-able places using a single gensym to hold 
;;; category: capture-management | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((swap-places x y) . (LET ((#:G1 X)) (SETF X Y) (SETF Y #:G1))))
